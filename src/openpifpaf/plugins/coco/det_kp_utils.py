"""
Util functions for object detection with triplet keypoints
"""

import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# Useful constants
from openpifpaf.transforms.det_kp_transforms import create_keypoints_from_bbox

DETKP_SKELETON = [(1, 2)]
DETKP_KEYPOINTS = ['top_left', 'bottom_right']
DETKP_POSE = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
DETKP_SIGMAS = [1.0, 1.0]
DETKP_SCORE_WEIGHTS = [1.0, 1.0]


def create_detection_keypoints_annotations(detection_annos):
    kp_info = dict(description='COCO 2017 detection dataset formatted as keypoint triplets', version='1.0', year=2022,
                   date_created='2022/06/01')
    kp_annotations = [
        dict(
            segmentation=da['segmentation'],
            num_keypoints=2,
            area=da['area'],
            iscrowd=da['iscrowd'],
            keypoints=create_keypoints_from_bbox(da['bbox']),
            image_id=da['image_id'],
            bbox=da['bbox'],
            category_id=da['category_id'],
            id=da['id']
        )
        for da in detection_annos['annotations']
    ]
    kp_categories = [
        dict(
            supercategory=dc['supercategory'], id=dc['id'], name=dc['name'],
            keypoints=DETKP_KEYPOINTS,
            skeleton=DETKP_SKELETON
        )
        for dc in detection_annos['categories']
    ]

    return dict(
        info=kp_info,
        licenses=detection_annos['licenses'],
        images=detection_annos['images'],
        annotations=kp_annotations,
        categories=kp_categories,
    )


def visualise_test(ann_file):
    root_dir = '../../data-mscoco/images/val2017/'
    coco = COCO(ann_file)
    with open(ann_file) as f:
        anns = json.load(f)

    for i in range(10):
        test_image_dict = anns['images'][i]
        test_annos = [ann for ann in anns['annotations'] if ann['image_id'] == test_image_dict['id']]
        test_image = Image.open(os.path.join(root_dir + test_image_dict['file_name']))
        # display_img_anns(test_image, test_annos, coco)


def create_det_keypoint_annotation_file(root_dir, detection_ann_file, name_template='test_template_'):
    with open(os.path.join(root_dir, detection_ann_file)) as f:
        detection_annos = json.load(f)

    detection_tripkp_annos = create_detection_keypoints_annotations(detection_annos)

    with open(os.path.join(root_dir, name_template + detection_ann_file), 'w') as outfile:
        json.dump(detection_tripkp_annos, outfile)


def create_det_keypoint_test_anno_file(root_dir):
    with open(os.path.join(root_dir, 'detection_five_kp_instances_train2017.json')) as f:
        train_annos = json.load(f)

    num_images = 50
    images, annotations = [], []
    for i in range(num_images):
        cur_img = train_annos['images'][i]
        image_id = cur_img['id']
        cur_annos = [ann for ann in train_annos['annotations'] if
                     ann['image_id'] == image_id and ann['category_id'] == 1]
        images.append(cur_img)
        annotations.extend(cur_annos)

    with open(os.path.join(root_dir, 'detection_five_kp_test_person_only.json'), 'w') as outfile:
        json.dump(dict(
            info=dict(description='Test COCO 2017 detection dataset formatted as 5kp',
                      version='1.0', year=2022, date_created='2022/06/01'),
            licenses=train_annos['licenses'],
            images=images,
            annotations=annotations,
            categories=train_annos['categories'],
        ), outfile)


def create_class_agnostic_detection_annos(root_dir, detection_ann_file):
    with open(os.path.join(root_dir, detection_ann_file)) as f:
        detection_annos = json.load(f)

    new_detection_annos_list = []

    for ann in detection_annos['annotations']:
        ann['category_id'] = 1
        new_detection_annos_list.append(ann)

    detection_annos['annotations'] = new_detection_annos_list
    detection_annos['categories'] = detection_annos['categories'][:1]
    detection_annos['categories'][0] = dict(supercategory='object', id=1, name='object')
    with open(os.path.join(root_dir, 'detection_class_agnostic_' + detection_ann_file), 'w') as outfile:
        json.dump(detection_annos, outfile)


if __name__ == '__main__':
    import json

    anno_root_dir = '../../data-mscoco/annotations/'
    # create_det_keypoint_annotation_file(anno_root_dir, 'instances_val2017.json', name_template='detection_cornernet_')
    # create_det_keypoint_annotation_file(anno_root_dir, 'instances_train2017.json', name_template='detection_cornernet_')
    # create_class_agnostic_detection_annos(anno_root_dir, 'detection_cornernet_instances_train2017.json')
    # create_class_agnostic_detection_annos(anno_root_dir, 'detection_cornernet_instances_val2017.json')
    # f(anno_root_dir, 'detection_triplet_kp_instances_train2017.json')
    # f(anno_root_dir, 'detection_triplet_kp_instances_val2017.json')


    # visualise_test(anno_root_dir + 'instances_val2017.json')

    """
    --cocokp-train-annotations=data-mscoco/annotations/detection_triplet_kp_instances_train2017.json
    --cocokp-val-annotations=data-mscoco/annotations/detection_triplet_kp_instances_val2017.json
    """

    # create_det_keypoint_test_anno_file(anno_root_dir)

    with open(anno_root_dir + 'detection_class_agnostic_detection_cornernet_instances_val2017.json', 'r') as f:
        test_annos = json.load(f)

    print('Done')
