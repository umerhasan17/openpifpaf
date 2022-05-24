import logging
from copy import deepcopy
from typing import Dict

import albumentations as A
import numpy as np
from PIL import Image

import openpifpaf.transforms
from openpifpaf.transforms import Preprocess

LOG = logging.getLogger(__name__)


class AlbumentationsComposeWrapperMeta(type(Preprocess), type(A.BaseCompose)):
    pass


class AlbumentationsComposeWrapper(A.Compose):
    """
    Wrapper class for albumentations compose transform.
    Please note if you are using this to add more functionality from albumentations,
    the call method has been greatly simplified and may lack functionality.

    Paste in annotations from previous image into current image
    """

    def reformat_annotations(self, annotations, bboxes):
        """ Update annotations with new bboxes """
        new_annotations = []
        for ix, bbox in enumerate(bboxes):
            [x, y, w, h, category_id, _] = bbox
            assert category_id == annotations[ix]['category_id']  # check bounding box and ann for same object
            new_annotations.append(dict(
                bbox=np.array([x, y, w, h]),
                category_id=category_id,
                bbox_original=annotations[ix]['bbox_original'],
                iscrowd=annotations[ix]['iscrowd'],
                image_id=annotations[ix]['image_id'],
                id=annotations[ix]['id'],
            ))

        return new_annotations

    def convert_albumentations_to_anns(self, original_annotations):


    def convert_output_to_anns(self, transformed_output: Dict, unedited_image_data: Dict, original_annotations,
                               copy_paste_applied=True):
        """ Convert masks and bboxes back to annotations """
        annotations = []
        # add annotations for current image to new annotations list
        annotations.extend(self.reformat_annotations(original_annotations, unedited_image_data['bboxes']))
        # add annotations for previous image to new annotations list
        if copy_paste_applied:
            assert self.previous_image_data is not None
            annotations.extend(self.reformat_annotations(
                self.previous_image_data['previous_anns'],
                self.previous_image_data['previous_bboxes']
            ))

        return annotations

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

    def __call__(self, image, anns, meta):
        LOG.debug('Applying albumentations transform')
        # convert target segmentations to masks
        # bboxes are expected to be (y1, x1, y2, x2, category_id)
        # TODO keep original bboxes as bbox original, done by normalise annotations as well, need to add in convert back to annotations
        crowd_annotations = []
        masks = []
        bboxes = []
        # adjust image dimensions in coco instance
        current_image_id = anns[0]['image_id']
        current_image_width, current_image_height = image.size
        # meta['coco_instance'].imgs[current_image_id]['height'] = current_image_height
        # meta['coco_instance'].imgs[current_image_id]['width'] = current_image_width
        for ix, ann in enumerate(anns):
            if type(ann['segmentation']) == list:
                masks.append(meta['ann_to_mask'](ann))
                bboxes.append(ann['bbox'].tolist() + [ann['category_id']] + [ix])
            else:
                # avoid processing crowd annotations with copy paste augmentation
                crowd_annotations.append(ann)

        # pack outputs into a dict
        data = {
            'image': np.asarray(image),
            'masks': masks,
            'bboxes': bboxes,
        }

        # assert len(self.transforms) == 1
        # t = self.transforms[0]
        # assert isinstance(t, openpifpaf.transforms.CopyPaste)

        for t in self.transforms:
            if isinstance(t, openpifpaf.transforms.CopyPaste):
                unedited_image_data = deepcopy(data)  # save unedited image as previous image
                if self.previous_image_data is not None:
                    LOG.debug(
                        f'Previous image exists, applying copy paste, copy paste arguments: '
                        f'{data["image"].shape}, {len(data["masks"])}, '
                        f'{data["masks"][0].shape}, {data["bboxes"]}'
                    )
                    data['bboxes'] = [list(x) for x in data['bboxes']]
                    self.previous_image_data['previous_bboxes'] = [list(x) for x in self.previous_image_data['previous_bboxes']]
                    data = t(**dict(paste_image=self.previous_image_data['previous_image'],
                                    paste_masks=self.previous_image_data['previous_masks'],
                                    paste_bboxes=self.previous_image_data['previous_bboxes'],
                                    **data))
                    transformed_anns = self.convert_output_to_anns(
                        data, unedited_image_data, anns, copy_paste_applied=True
                    )

                    # useful debug statement
                    from openpifpaf.transforms.copy_paste.visualize import display_instances
                    import matplotlib.pyplot as plt
                    f, ax = plt.subplots(1, 2, figsize=(16, 16))
                    empty = np.array([])
                    display_instances(data['image'], empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax[0])
                    plt.show()
                else:
                    transformed_anns = self.convert_output_to_anns(data, unedited_image_data, anns, copy_paste_applied=False)

                self.update_previous_image(
                    unedited_image_data['image'],
                    unedited_image_data['masks'],
                    unedited_image_data['bboxes'],
                    anns
                )
            else:
                # apply other transformations

                # def convert_coco_bbox_to_albumentations_bbox(bbox):


                # extra_bbox_details = [x[-2:] for x in data['bboxes']]
                raw_bbox = [x[4:] for x in data['bboxes']]
                # data['bboxes'] = raw_bbox
                data = t(**data)
                # data['bboxes'] = [x + extra_bbox_details[i] for i, x in enumerate(data['bboxes'])]

        return Image.fromarray(data['image']), transformed_anns + crowd_annotations, meta


class AlbumentationsComposeCopyPasteWrapper(AlbumentationsComposeWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_image_data = None

    def convert_copy_paste_to_anns(self, unedited_image_data, original_annotations):

