import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image

import openpifpaf


def make_prediction():
    predictor = openpifpaf.Predictor(checkpoint='cocofivekp.pkl.epoch134')
    with Image.open('../data-mscoco/images/train2017/000000184613.jpg') as pil_im:
        predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
    print(predictions)


def debug_data_target(data, targets):
    print(f'Data dim: {data.shape}, No targets: {len(targets)}')
    assert data.shape[0] == 1
    t = T.ToPILImage()
    (torch.squeeze(data)).show()
    for target in targets:
        t(target).show()


def debug_image_anns(image, anns):
    plt.imshow(image)
    for ann in anns:
        pass



# from openpifpaf.hasan.research import debug_data_target

if __name__ == '__main__':
    debug_data_target(torch.zeros(1, 3, 385, 385), torch.ones(1, 5, 5, 25, 25))

"""
# srun --gpu-bind=closest \
#   /bin/bash -c 'python3 -m openpifpaf.train --ddp \
#   --lr=0.0003 --momentum=0.95 --clip-grad-value=10 \
#   --epochs=150 \
#   --lr-decay 130 140 --lr-decay-epochs=10 \
#   --batch-size=8 \
#   --weight-decay=1e-5 \
#   --dataset=cocodetkptriplets --cocodet-upsample=2 \
#   --basenet=resnet50 \
#   --resnet-input-conv2-stride=2 --resnet-block5-dilation=2'
"""
