import random

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

import openpifpaf

from openpifpaf.transforms.copy_paste.copy_paste_coco import CocoDetectionCP
from openpifpaf.transforms.copy_paste.copy_paste_utils import CopyPaste#
from openpifpaf.transforms.copy_paste.visualize import display_instances


def visualise_example(img_data):
    f, ax = plt.subplots(1, 2, figsize=(16, 16))

    image = img_data['image']
    masks = img_data['masks']
    bboxes = img_data['bboxes']

    empty = np.array([])
    display_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax[0])

    if len(bboxes) > 0:
        boxes = np.stack([b[:4] for b in bboxes], axis=0)
        box_classes = np.array([b[-2] for b in bboxes])
        mask_indices = np.array([b[-1] for b in bboxes])
        show_masks = np.stack(masks, axis=-1)[..., mask_indices]
        class_names = {k: data.coco.cats[k]['name'] for k in data.coco.cats.keys()}
        display_instances(image, boxes, show_masks, box_classes, class_names, show_bbox=True, ax=ax[1])
    else:
        display_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax[1])

    plt.show()


if __name__ == '__main__':
    random.seed(3)

    transform = openpifpaf.transforms.AlbumentationsComposeWrapper([
        # A.RandomScale(scale_limit=(-0.9, 1), p=1),  # LargeScaleJitter from scale of 0.1 to 2
        A.PadIfNeeded(256, 256, border_mode=0),  # pads with image in the center, not the top left like the paper
        A.RandomCrop(256, 256),
        CopyPaste(blend=True, sigma=1, pct_objects_paste=1, p=1)  # pct_objects_paste is a guess
    ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
    )

    data = CocoDetectionCP(
        '../../../data-mscoco/images/train2017/',
        '../../../data-mscoco/annotations/instances_train2017.json',
        transform
    )

    x = data[6]
    visualise_example(x)
    # for i in range(10, 15):
    #     visualise_example(data[i])
    # visualise_example(data[200])

    print('done')
