import logging
from copy import deepcopy

import albumentations as A
import numpy as np
from PIL import Image

import openpifpaf.transforms
from openpifpaf.transforms import Preprocess

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
            [x, y, w, h, category_id, mask_idx] = bbox
            # TODO careful if some data gets lost and bboxes are removed during transformations
            assert category_id == annotations[mask_idx]['category_id']  # check bounding box and ann for same object
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
        LOG.debug('Applying albumentations transforms')

        self.adjust_meta_img_dimensions(image, anns, meta)  # may be unnecessary if not applied with other transforms

        # convert target segmentations to masks
        # bboxes are expected to be (y1, x1, y2, x2, category_id)
        # TODO keep original bboxes as bbox original, done by normalise annotations as well,
        #  need to add in convert back to annotations
        all_masks, all_bboxes = [], []
        crowd_annotation_indices = []  # stores indices of crowd annotations to avoid running copy paste augmentation
        for ix, ann in enumerate(anns):
            mask = meta['ann_to_mask'](ann)
            bbox = ann['bbox'].tolist() + [ann['category_id']] + [ix]
            all_masks.append(mask)
            all_bboxes.append(bbox)
            if type(ann['segmentation']) != list:
                crowd_annotation_indices.append(ix)

        # pack outputs into a dict
        data = {
            'image': np.asarray(image),
            'masks': all_masks,
            'bboxes': all_bboxes,
        }

        # apply usual transforms
        data = A.Compose.__call__(self, **data)
        updated_annotations = self.reformat_annotations(anns, data['bboxes'])
        cp_output_data = None

        if self.apply_copy_paste:
            LOG.debug('Applying albumentations copy paste transform')

            # apply copy paste augmentation on non crowd annotations only
            # TODO 3/4 masks are empty but there are 2 bbox existing?
            cp_masks, cp_bboxes = [], []
            bbox_ix = 0
            for ix, mask in enumerate(data['masks']):
                # mask is empty and no bbox exists OR mask is associated with crowd annotation (SKIP in both cases)
                if len(np.unique(mask)) == 1 or ix in crowd_annotation_indices:
                    pass
                else:
                    cp_masks.append(mask)
                    cp_bboxes.append(data['bboxes'][bbox_ix])  # get equivalent bbox for mask
                    bbox_ix += 1
            assert len(cp_bboxes) == len(cp_masks)

            if self.previous_image_data is not None:
                cp_data = dict(
                    image=data['image'],
                    bboxes=deepcopy(cp_bboxes),
                    masks=deepcopy(cp_masks),
                    paste_image=self.previous_image_data['previous_image'],
                    paste_masks=self.previous_image_data['previous_masks'],
                    paste_bboxes=self.previous_image_data['previous_bboxes'],
                )
                cp_transform = A.Compose(
                    [openpifpaf.transforms.CopyPaste(blend=True, sigma=1, pct_objects_paste=1, p=1)],
                    bbox_params=A.BboxParams(format="coco")
                )
                cp_output_data = cp_transform(**cp_data)
                updated_annotations = []
                # add annotations from current image
                updated_annotations.extend(self.reformat_annotations(anns, cp_bboxes))
                # add annotations from previous image
                updated_annotations.extend(self.reformat_annotations(
                    self.previous_image_data['previous_anns'], self.previous_image_data['previous_bboxes']
                ))

                # useful debug statement
                from openpifpaf.transforms.copy_paste.visualize import display_instances
                import matplotlib.pyplot as plt
                f, ax = plt.subplots(1, 2, figsize=(16, 16))
                empty = np.array([])
                display_instances(cp_output_data['image'], empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax[0])
                plt.show()

            # save current image details for next copy paste augmentation
            self.update_previous_image(
                data['image'],
                cp_masks,
                cp_bboxes,
                anns
            )

        if cp_output_data is not None:
            data = cp_output_data

        return Image.fromarray(data['image']), updated_annotations, meta

