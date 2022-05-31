import os
import openpifpaf
from openpifpaf.plugins.coco import CocoDet
import albumentations as A
import uuid
from PIL import Image
import pickle
import numpy as np

from openpifpaf.plugins.coco.constants import COCO_CATEGORIES

if __name__ == '__main__':
    os.chdir('/home/albion/code/epfl/pdm/openpifpaf/src')
    enc = openpifpaf.encoder.CifDet(openpifpaf.headmeta.CifDet('cifdet', 'cocodet', COCO_CATEGORIES))

    cp_class = openpifpaf.transforms.Compose([
        openpifpaf.transforms.NormalizeAnnotations(),
        openpifpaf.transforms.AlbumentationsComposeWrapper([
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=(0.5, 1.0), p=1),
            A.Blur(),
            A.RandomRotate90(),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
            A.RandomCrop(512, 512, always_apply=True),  # TODO area of interest
        ], apply_copy_paste=True, bbox_params=A.BboxParams(format="coco")),
        openpifpaf.transforms.MinSize(min_side=4.0),
        openpifpaf.transforms.UnclippedArea(threshold=0.75),
        openpifpaf.transforms.TRAIN_TRANSFORM,
        openpifpaf.transforms.Encoders([enc]),
    ])



    # cp_class = openpifpaf.transforms.Compose([
    #     openpifpaf.transforms.AlbumentationsComposeWrapper([
    #         A.HorizontalFlip(p=1),
    #     ], bbox_params=A.BboxParams(format="coco")),
    #     openpifpaf.transforms.AlbumentationsComposeCopyPasteWrapper([
    #         openpifpaf.transforms.CopyPaste(blend=True, sigma=1, pct_objects_paste=1, p=1),
    #     ], bbox_params=A.BboxParams(format="coco"))
    # ])

    # cp_class = A.Compose([
    #         # A.RandomScale(scale_limit=(-0.9, 1), p=1),  # LargeScaleJitter from scale of 0.1 to 2
    #         # A.CropAndPad(px=512),
    #         # A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
    #         A.HorizontalFlip(p=1),
    #         # A.RandomCrop(512, 512, always_apply=True),  # TODO area of interest
    #         # pads with image in the center, not the top left like the paper
    #         openpifpaf.transforms.CopyPaste(blend=True, sigma=1, pct_objects_paste=1, p=1)
    #         # pct_objects_paste is a guess
    #     ], bbox_params=A.BboxParams(format="coco"))

    def get_test_details(id1):
        # root_dir = 'code/epfl/pdm/openpifpaf/src/'
        root_dir = 'cp_test_images/'
        im1 = Image.open(f'{root_dir}cp_test_img_{id1}.jpg')
        with open(f'{root_dir}cp_test_anns_{id1}.pkl', 'rb') as f:
            ann1 = pickle.load(f)
        with open(f'{root_dir}cp_test_meta_{id1}.pkl', 'rb') as f:
            meta1 = pickle.load(f)
        return im1, ann1, meta1


    # first image
    # details = get_test_details('eb335416-da9e-11ec-8fb2-afd08f3669a7')
    # input_data = dict(
    #     image=np.asarray(details[0]),
    #     bboxes=[ann['bbox'].tolist() + [ann['category_id']] + [ix] for ix, ann in enumerate(details[1])],
    #     masks=[details[2]['ann_to_mask'](ann) for ann in details[1]]
    # )
    # cp_class(**input_data)
    test_image_ids = [x.rsplit('_')[-1][:-4] for x in os.listdir('cp_test_images') if x.endswith('.jpg')]
    for cur_id in test_image_ids:
        # if cur_id.startswith('6a0') or cur_id.startswith('1ef'):
        print(cp_class(*get_test_details(cur_id)))

    # cp_class(*get_test_details('eb335416-da9e-11ec-8fb2-afd08f3669a7'))
    # # second image
    # cp_class(*get_test_details('0cf10558-da9f-11ec-8fb2-afd08f3669a7'))

    # test_coco_det = CocoDet()
    # loader = test_coco_det.train_loader()
    # loader_iter = iter(loader)
    # x = next(loader_iter)
    # print(x)


"""
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
openpifpaf 0.13.3 requires torch==1.11.0, but you have torch 1.9.0 which is incompatible.
openpifpaf 0.13.3 requires torchvision==0.12.0, but you have torchvision 0.10.0 which is incompatible.
"""

"""
display_instances(self.previous_image_data['previous_image'], np.array([x[:4] for x in self.previous_image_data['previous_bboxes']]), np.moveaxis(np.array(self.previous_image_data['previous_masks']), 0, -1), np.array([0, 1, 2, 3]), np.array(['c1', 'c2', 'c3', 'c4']))

display_instances(data['image'], np.array([x[:4] for x in data['bboxes']]), np.moveaxis(np.array(data['masks']), 0, -1), np.array([0, 1, 2, 3]), np.array(['c1', 'c2', 'c3', 'c4']))

display_instances(data['image'], np.array([x[:4] for x in cp_bboxes]), np.moveaxis(np.array(cp_masks), 0, -1), np.array([i for i in range(len(cp_masks))]), np.array(['c1'] * len(cp_masks)))

display_instances(cp_output_data['image'], np.array([x[:4] for x in cp_output_data['bboxes']]), np.moveaxis(np.array(cp_output_data['masks']), 0, -1), np.array([i for i in range(len(cp_output_data['masks']))]), np.array(['c1'] * len(cp_output_data['masks'])))

"""


"""
        # extra_bbox_details = [x[-2:] for x in data['bboxes']]
        # raw_bbox = [x[4:] for x in data['bboxes']]
        # data['bboxes'] = raw_bbox
        # data = t(**data)
        # data['bboxes'] = [x + extra_bbox_details[i] for i, x in enumerate(data['bboxes'])]

"""


"""
        masks, bboxes, crowd_masks, crowd_bboxes, all_masks, all_bboxes = [], [], [], [], [], []
        crowd_annotation_indices = []  # stores indices of crowd annotations to avoid running copy paste augmentation
        for ix, ann in enumerate(anns):
            mask = meta['ann_to_mask'](ann)
            bbox = ann['bbox'].tolist() + [ann['category_id']] + [ix]
            all_masks.append(mask)
            all_bboxes.append(bbox)
            if type(ann['segmentation']) == list:
                masks.append(mask)
                bboxes.append(bbox)
            else:
                crowd_masks.append(mask)
                crowd_bboxes.append(bbox)
                crowd_annotation_indices.append(ix)

"""