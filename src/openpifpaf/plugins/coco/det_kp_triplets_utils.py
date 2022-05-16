"""
Util functions for object detection with triplet keypoints
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


# Useful constants
TRIPLET_SKELETON = [(1, 2), (2, 3)]
TRIPLET_KEYPOINTS = ['top_left', 'center', 'bottom_right']
DUMMY_TRIPLET_HFLIP = {x: x for x in TRIPLET_KEYPOINTS}
DUMMY_TRIPLET_POSE = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
DUMMY_TRIPLET_SIGMAS = [0.0, 0.0, 0.0]
DUMMY_TRIPLET_SCORE_WEIGHTS = [1.0, 1.0, 1.0]


def create_detection_triplet_keypoints_annotations(detection_annos):
    def create_bbox_triplet_keypoints(bbox: list):
        assert len(bbox) == 4
        [x, y, w, h] = bbox
        # keypoints are represented as 3 values (x coordinate, y coordinate, visibility flag=2)
        return [x, y, 2, x + w / 2, y + h / 2, 2, x + w, y + h, 2]

    kp_info = dict(description='COCO 2017 detection dataset formatted as keypoint triplets', version='1.0', year=2022,
                   date_created='2022/06/01')
    kp_annotations = [
        dict(
            segmentation=da['segmentation'],
            num_keypoints=3,
            area=da['area'],
            iscrowd=da['iscrowd'],
            keypoints=create_bbox_triplet_keypoints(da['bbox']),
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
            keypoints=TRIPLET_KEYPOINTS,
            skeleton=TRIPLET_SKELETON
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
        plt.imshow(np.asarray(test_image))
        coco.showAnns(test_annos, draw_bbox=True)
        plt.show()


def create_triplet_keypoint_annotation_file(root_dir, detection_ann_file):
    with open(os.path.join(root_dir, detection_ann_file)) as f:
        detection_annos = json.load(f)

    detection_tripkp_annos = create_detection_triplet_keypoints_annotations(detection_annos)

    with open(os.path.join(root_dir, 'detection_triplet_kp_' + detection_ann_file), 'w') as outfile:
        json.dump(detection_tripkp_annos, outfile)


if __name__ == '__main__':
    import json

    anno_root_dir = '../../data-mscoco/annotations/'
    create_triplet_keypoint_annotation_file(anno_root_dir, 'instances_val2017.json')
    create_triplet_keypoint_annotation_file(anno_root_dir, 'instances_train2017.json')

    # visualise_test('json_test_anns.json')

    """
    --cocokp-train-annotations=data-mscoco/annotations/detection_triplet_kp_instances_train2017.json
    --cocokp-val-annotations=data-mscoco/annotations/detection_triplet_kp_instances_val2017.json
    """

    print('Done')
