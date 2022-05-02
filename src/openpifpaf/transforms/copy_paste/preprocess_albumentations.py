import logging
from typing import Dict

import albumentations as A
import numpy as np
from PIL import Image

import openpifpaf.transforms
from openpifpaf.transforms import Preprocess
from copy import deepcopy

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_image_data = None

    def convert_output_to_anns(self, transformed_output: Dict, unedited_image_data: Dict, original_annotations,
                               copy_paste_applied=True):
        """ Convert masks and bboxes back to annotations """
        annotations = []
        for ix, bbox in enumerate(unedited_image_data['bboxes']):
            [x, y, w, h, category_id, _] = bbox
            assert category_id == original_annotations[ix]['category_id']  # check bounding box and ann for same object
            annotations.append(dict(
                bbox=np.array([x, y, w, h]),  # TODO maybe this is x1, y1, x2, y2
                category_id=category_id,
                bbox_original=original_annotations[ix]['bbox_original'],
                iscrowd=original_annotations[ix]['iscrowd'],
                image_id=original_annotations[ix]['image_id'],
                id=original_annotations[ix]['id'],
            ))

        if copy_paste_applied:
            assert self.previous_image_data is not None
            previous_annotations = self.previous_image_data['previous_anns']
            for ix, bbox in enumerate(self.previous_image_data['previous_bboxes']):
                [x, y, w, h, category_id, _] = bbox
                assert category_id == previous_annotations[ix][
                    'category_id']  # check bounding box and ann for same object
                annotations.append(dict(
                    bbox=np.array([x, y, w, h]),  # TODO maybe this is x1, y1, x2, y2
                    category_id=category_id,
                    bbox_original=previous_annotations[ix]['bbox_original'],
                    iscrowd=previous_annotations[ix]['iscrowd'],
                    image_id=previous_annotations[ix]['image_id'],
                    id=previous_annotations[ix]['id'],
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
        current_image_height, current_image_width = image.size
        meta['coco_instance'].imgs[current_image_id]['height'] = current_image_height
        meta['coco_instance'].imgs[current_image_id]['width'] = current_image_width
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

        for idx, t in enumerate(self.transforms):
            if isinstance(t, openpifpaf.transforms.CopyPaste):
                if self.previous_image_data is not None:
                    LOG.debug(
                        f'Copy paste arguments: {data["image"].shape}, {len(data["masks"])}, '
                        f'{data["masks"][0].shape}, {data["bboxes"]}')
                    unedited_image_data = deepcopy(data)  # save unedited image as previous image
                    data = t(**dict(paste_image=self.previous_image_data['previous_image'],
                                    paste_masks=self.previous_image_data['previous_masks'],
                                    paste_bboxes=self.previous_image_data['previous_bboxes'],
                                    **data))
                    transformed_anns = self.convert_output_to_anns(
                        data, unedited_image_data, anns, copy_paste_applied=True
                    )
                    self.update_previous_image(
                        unedited_image_data['image'],
                        unedited_image_data['masks'],
                        unedited_image_data['bboxes'],
                        anns
                    )

                    # useful debug statement
                    # from openpifpaf.transforms.copy_paste.visualize import display_instances
                    # import matplotlib.pyplot as plt
                    # f, ax = plt.subplots(1, 2, figsize=(16, 16))
                    # empty = np.array([])
                    # display_instances(data['image'], empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax[0])
                    # plt.show()

                    return Image.fromarray(data['image']), transformed_anns, meta
            else:
                data = t(**data)
                LOG.debug(f'Albumentations transforms image dimensions: {data["image"].shape}')

        self.update_previous_image(data['image'], data['masks'], data['bboxes'], anns)
        transformed_anns = self.convert_output_to_anns(data, data, anns, copy_paste_applied=False)
        return Image.fromarray(data['image']), transformed_anns + crowd_annotations, meta
