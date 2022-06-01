import logging
from copy import deepcopy

import albumentations as A
import numpy as np
from PIL import Image

import openpifpaf.transforms
from openpifpaf.transforms import Preprocess

import time

LOG = logging.getLogger(__name__)


class AlbumentationsComposeWrapperMeta(type(Preprocess), type(A.BaseCompose)):
    pass


class AlbumentationsComposeWrapper(Preprocess, A.Compose, metaclass=AlbumentationsComposeWrapperMeta):
    """
    Wrapper class for albumentations compose transform.
    Please note if you are using this to add more functionality from albumentations,
    the call method has been greatly simplified and may lack functionality.

    Paste in annotations from previous image into current image
    """

    def __init__(self, *args, apply_copy_paste=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_copy_paste = apply_copy_paste
        self.previous_image_data = None
        self.calls = 0


    def update_previous_image(self, image, masks, bboxes, anns):
        """
            Masks and bboxes have been edited with cropping and padding.
            Annotations are the original annotations coming into the transformation function.
        """
        self.previous_image_data = dict(
            previous_image=image,
            previous_masks=masks,
            previous_bboxes=bboxes,
            previous_anns=anns,
        )

    def convert_copy_paste_to_anns(self, unedited_image_data, original_annotations):
        """ Convert masks and bboxes back to annotations """
        annotations = []
        # add annotations for current image to new annotations list
        annotations.extend(self.reformat_annotations(original_annotations, unedited_image_data['bboxes']))
        # add annotations for previous image to new annotations list
        assert self.previous_image_data is not None
        annotations.extend(self.reformat_annotations(
            self.previous_image_data['previous_anns'],
            self.previous_image_data['previous_bboxes']
        ))
        return annotations

    @staticmethod
    def reformat_annotations(annotations, bboxes):
        """ Update annotations with new bboxes. Note the reformat function removes segmentation details. """
        new_annotations = []
        for ix, bbox in enumerate(bboxes):
            if len(bbox) == 6:
                [x, y, w, h, category_id, ann_idx] = bbox
            elif len(bbox) == 7:
                [x, y, w, h, category_id, _, ann_idx] = bbox
            else:
                raise ValueError('Incorrect bounding box format')
            assert category_id == annotations[ann_idx]['category_id']  # check bounding box and ann for same object
            new_annotations.append(dict(
                bbox=np.array([x, y, w, h]),
                category_id=category_id,
                bbox_original=annotations[ix]['bbox_original'],
                iscrowd=annotations[ix]['iscrowd'],
                image_id=annotations[ix]['image_id'],
                id=annotations[ix]['id'],
            ))
        return new_annotations

    @staticmethod
    def adjust_meta_img_dimensions(image, anns, meta):
        # adjust image dimensions in coco instance
        current_image_id = anns[0]['image_id']
        current_image_width, current_image_height = image.size
        meta['coco_instance'].imgs[current_image_id]['height'] = current_image_height
        meta['coco_instance'].imgs[current_image_id]['width'] = current_image_width

    def __call__(self, image, anns, meta):
        self.calls += 1
        # LOG.debug('Applying albumentations transforms')
        t1 = time.time()

        self.adjust_meta_img_dimensions(image, anns, meta)  # may be unnecessary if not applied with other transforms

        # convert target segmentations to masks
        all_masks, all_bboxes = [], []
        for ix, ann in enumerate(anns):
            mask = meta['ann_to_mask'](ann)
            bbox = ann['bbox'].tolist() + [ann['category_id']] + [ix]
            all_masks.append(mask)
            all_bboxes.append(bbox)

        # pack outputs into a dict
        data = {
            'image': np.asarray(image),
            'masks': all_masks,
            'bboxes': all_bboxes,
        }
        t2 = time.time()
        # apply usual transforms
        data = A.Compose.__call__(self, **data)
        updated_annotations = self.reformat_annotations(anns, data['bboxes'])
        cp_output_data = None
        t3 = time.time()
        t4 = t5 = t6 = t3
        if self.apply_copy_paste:
            if self.previous_image_data is not None:
                # LOG.debug('Applying albumentations copy paste transform')
                cp_data = dict(
                    image=data['image'],
                    bboxes=[bbox + (bbox[-1],) for bbox in data['bboxes']],
                    masks=deepcopy(data['masks']),
                    paste_image=self.previous_image_data['previous_image'],
                    paste_masks=self.previous_image_data['previous_masks'],
                    paste_bboxes=[bbox + (len(data['masks']) + bbox[-1],) for bbox in self.previous_image_data['previous_bboxes']],
                )
                cp_transform = A.Compose(
                    [openpifpaf.transforms.CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1)],
                    bbox_params=A.BboxParams(format="coco")
                )
                cp_output_data = cp_transform(**cp_data)
                # add annotation details back in
                t4 = time.time()
                updated_annotations = self.reformat_annotations(
                    anns + self.previous_image_data['previous_anns'],
                    cp_output_data['bboxes']
                )
                t5 = time.time()

                # useful debug statement
                # from openpifpaf.transforms.copy_paste.visualize import display_instances
                # import matplotlib.pyplot as plt
                # if len(cp_output_data['bboxes']) == len(cp_output_data['masks']):
                #     display_instances(cp_output_data['image'], np.array([x[:4] for x in cp_output_data['bboxes']]),
                #                       np.moveaxis(np.array(cp_output_data['masks']), 0, -1),
                #                       np.array([i for i in range(len(cp_output_data['masks']))]),
                #                       np.array(['c1'] * len(cp_output_data['masks'])))
                #     plt.show()
                # else:
                #     LOG.debug('Bboxes and masks length does not match')

            # save current image details for next copy paste augmentation
            self.update_previous_image(
                data['image'],
                data['masks'],
                data['bboxes'],
                anns
            )
            t6 = time.time()

        if cp_output_data is not None:
            data = cp_output_data
        t7 = time.time()
        r_func = lambda x: round(x, 4)
        LOG.info(f't1->t2: {r_func(t2-t1)}, t2->t3: {r_func(t3-t2)}, t3->t4: {r_func(t4-t3)}, t4->t5: {r_func(t5-t4)}, '
                 f't5->t6: {r_func(t6-t5)}, t6->t7: {r_func(t7-t6)}, total: {r_func(t7-t1)}'
                 f'len annos: {len(updated_annotations)}, '
                 f'image id: {updated_annotations[0]["image_id"] if len(updated_annotations) > 0 else -1}')
        return Image.fromarray(data['image']), updated_annotations, meta

