from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def display_kp_markers(image, anns, coco):
    for ann in anns:
        for i in range(0, len(ann['keypoints']), 3):
            x, y = ann['keypoints'][i], ann['keypoints'][i+1]
            plt.plot(x, y, marker='v', color='red')
    plt.imshow(np.asarray(image))
    coco.showAnns(anns, draw_bbox=True)


def display_img_anns(image, anns, meta, show=False, seperate_categories=False):
    anns = deepcopy(anns)
    if type(image) == torch.Tensor:
        t = T.ToPILImage()
        image = t(image)
    coco = meta['coco']
    # for ann in anns:
    #     ann['keypoints'] = ann['keypoints'].tolist()
    if seperate_categories:
        ann_dict = defaultdict(list)
        for ann in anns:
            ann_dict[ann['category_id']].append(ann)
        for k, v in ann_dict.items():
            print('Category id: ', k)
            display_kp_markers(image, v, coco)
            plt.show()
            plt.clf()
    else:
        display_kp_markers(image, anns, coco)
        plt.savefig(f'model-input-images/{meta["image_id"]}.jpeg')
        if show:
            plt.show()
        plt.clf()
