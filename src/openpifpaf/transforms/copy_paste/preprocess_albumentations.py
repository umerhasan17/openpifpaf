import logging
from typing import Dict

import albumentations as A
import numpy as np
from PIL import Image

import openpifpaf.transforms
from openpifpaf.transforms import Preprocess
from copy import deepcopy

LOG = logging.getLogger(__name__)


def convert_output_to_anns(transformed_output: Dict):
    """ Convert masks and bboxes back to annotations """
    annotations = []
    for bbox in transformed_output['bboxes']:
        [x, y, w, h, category_id, _] = bbox
        annotations.append(dict(
            bbox=[x, y, w, h],  # TODO maybe this is x1, y1, x2, y2
            category_id=category_id
        ))

    return annotations


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

    def update_previous_image(self, image, paste_masks, paste_bboxes):
        update_data = dict(
            paste_image=image,
            paste_masks=paste_masks,
            paste_bboxes=paste_bboxes
        )

        self.previous_image_data = update_data

    def __call__(self, image, anns, meta):
        LOG.debug('Applying albumentations transform')
        # convert target segmentations to masks
        # bboxes are expected to be (y1, x1, y2, x2, category_id)
        # TODO keep original bboxes as bbox original, done by normalise annotations as well, need to add in convert back to annotations
        masks = []
        bboxes = []
        for ix, obj in enumerate(anns):
            masks.append(meta['ann_to_mask'](obj))
            bboxes.append(obj['bbox'].tolist() + [obj['category_id']] + [ix])

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
                    data = t(**dict(data, **self.previous_image_data))
                    transformed_anns = convert_output_to_anns(data)
                    self.update_previous_image(
                        unedited_image_data['image'],
                        unedited_image_data['masks'],
                        unedited_image_data['bboxes'],
                    )
                    return Image.fromarray(data['image']), transformed_anns, meta
            else:
                data = t(**data)

        self.update_previous_image(data['image'], data['masks'], data['bboxes'])
        transformed_anns = convert_output_to_anns(data)
        return Image.fromarray(data['image']), transformed_anns, meta
