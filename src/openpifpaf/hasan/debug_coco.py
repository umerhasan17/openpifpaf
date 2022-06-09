import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def display_img_anns(im, anns, meta):
    if type(im) == torch.Tensor:
        t = T.ToPILImage()
        im = t(im)
    plt.imshow(np.asarray(im))
    coco = meta['coco']
    coco.showAnns(anns, draw_bbox=True)
    # plt.show()
    plt.savefig(f'model-input-images/{meta["image_id"]}.jpeg')
