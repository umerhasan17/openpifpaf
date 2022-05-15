"""
Util functions for object detection with triplet keypoints
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO


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
            keypoints=['top_left', 'center', 'bottom_right'],
            skeleton=[[1, 2], [2, 3]]
        )
        for dc in detection_annos['categories']
    ]

    return dict(
        info=kp_info,
        images=detection_annos['images'],
        annotations=kp_annotations,
        categories=kp_categories,
    )


def display_image_with_bbox_kp(img, anns):
    fig = plt.figure(1, figsize=(20, 20))

    for ann in anns:
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        ax1 = fig.add_subplot(121)  # left side
        ax1.imshow(img)
        ax1.add_patch(
            Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, alpha=0.3, facecolor="blue", edgecolor="red", hatch='x'))

    # ax2 = fig.add_subplot(122)  # right side
    # ax2.imshow(res_img)
    # ax2 = plot_keypoints(draw_non_visible,keypoints,k_vis,ax2)
    plt.show()


def visualise_test(ann_file):
    root_dir = '../../data-mscoco/images/val2017/'
    coco = COCO(ann_file)
    with open(ann_file) as f:
        anns = json.load(f)
    test_image_dict = anns['images'][0]
    test_annos = [ann for ann in anns['annotations'] if ann['image_id'] == test_image_dict['id']]
    test_image = Image.open(os.path.join(root_dir + test_image_dict['file_name']))

    plt.imshow(np.asarray(test_image))
    coco.showAnns(test_annos, draw_bbox=True)
    plt.show()

    # plt.imshow(np.asarray(im))
    # # plt.savefig(f"{img_id}_unannotated.jpg", bbox_inches="tight", pad_inches=0)
    # metas['show_anns'](coco_annotation, draw_bbox=True)
    # plt.savefig(f"{img_id}_1.jpg", bbox_inches="tight", pad_inches=0)
    # plt.clf()


if __name__ == '__main__':
    import json

    with open('../../data-mscoco/annotations/instances_val2017.json') as f:
        detection_annos = json.load(f)

    with open('../../data-mscoco/annotations/person_keypoints_val2017.json') as f:
        keypoints_annos = json.load(f)

    detection_tripkp_annos = create_detection_triplet_keypoints_annotations(detection_annos)

    with open('json_test_anns.json', 'w') as outfile:
        json.dump(detection_tripkp_annos, outfile)

    visualise_test('json_test_anns.json')

    print('Done')
