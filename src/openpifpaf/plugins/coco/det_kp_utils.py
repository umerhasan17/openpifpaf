"""
Util functions for object detection with triplet keypoints
"""

import os

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import itertools

# Useful constants
from openpifpaf.hasan.debug_coco import display_img_anns

DETKP_SKELETON = [(1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)]
DETKP_KEYPOINTS = ['top', 'left', 'bottom', 'right', 'center']
DETKP_HFLIP = {
    'top': 'top',
    'left': 'right',
    'bottom': 'bottom',
    'right': 'left',
    'center': 'center',
}
DETKP_POSE = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
DETKP_SIGMAS = [1.0, 1.0, 1.0, 1.0, 1.0]
DETKP_SCORE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0]


def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.int32)
    return bbox


def _get_extreme_points(pts):
    l, t = min(pts[:, 0]), min(pts[:, 1])
    r, b = max(pts[:, 0]), max(pts[:, 1])
    # 3 degrees
    thresh = 0.02
    w = r - l + 1
    h = b - t + 1

    pts = np.concatenate([pts[-1:], pts, pts[:1]], axis=0)
    t_idx = np.argmin(pts[:, 1])
    t_idxs = [t_idx]
    tmp = t_idx + 1
    while tmp < pts.shape[0] and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
        t_idxs.append(tmp)
        tmp += 1
    tmp = t_idx - 1
    while tmp >= 0 and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
        t_idxs.append(tmp)
        tmp -= 1
    tt = [(max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) // 2, t]

    b_idx = np.argmax(pts[:, 1])
    b_idxs = [b_idx]
    tmp = b_idx + 1
    while tmp < pts.shape[0] and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
        b_idxs.append(tmp)
        tmp += 1
    tmp = b_idx - 1
    while tmp >= 0 and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
        b_idxs.append(tmp)
        tmp -= 1
    bb = [(max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) // 2, b]

    l_idx = np.argmin(pts[:, 0])
    l_idxs = [l_idx]
    tmp = l_idx + 1
    while tmp < pts.shape[0] and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
        l_idxs.append(tmp)
        tmp += 1
    tmp = l_idx - 1
    while tmp >= 0 and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
        l_idxs.append(tmp)
        tmp -= 1
    ll = [l, (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) // 2]

    r_idx = np.argmax(pts[:, 0])
    r_idxs = [r_idx]
    tmp = r_idx + 1
    while tmp < pts.shape[0] and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
        r_idxs.append(tmp)
        tmp += 1
    tmp = r_idx - 1
    while tmp >= 0 and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
        r_idxs.append(tmp)
        tmp -= 1
    rr = [r, (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) // 2]

    return np.array([tt, ll, bb, rr])


def generate_extreme_points_file(root_dir, person_only=True):
    ANN_PATH = root_dir + 'annotations/instances_{}2017.json'
    OUT_PATH = root_dir + 'annotations/instances_person_extreme_{}2017.json'
    IMG_DIR = root_dir + '{}2017/'
    DEBUG = False
    SPLITS = ['val', 'train']
    for split in SPLITS:
        data = json.load(open(ANN_PATH.format(split), 'r'))
        coco = COCO(ANN_PATH.format(split))
        img_ids = coco.getImgIds()
        num_images = len(img_ids)
        tot_box = 0
        print('num_images', num_images)
        # person only
        if person_only:
            anns_all = [ann for ann in data['annotations'] if ann['category_id'] == 1]
            data['categories'] = [category for category in data['categories'] if category['id'] == 1]
        else:
            anns_all = data['annotations']
        for category in data['categories']:
            category['keypoints'] = DETKP_KEYPOINTS
            category['skeleton'] = DETKP_SKELETON
        for i, ann in enumerate(anns_all):
            tot_box += 1
            bbox = ann['bbox']
            seg = ann['segmentation']
            if type(seg) == list:
                if len(seg) == 1:
                    pts = np.array(seg[0]).reshape(-1, 2)
                else:
                    pts = []
                    for v in seg:
                        pts += v
                    pts = np.array(pts).reshape(-1, 2)
            else:
                mask = coco.annToMask(ann) * 255
                tmp = np.where(mask > 0)
                pts = np.asarray(tmp).transpose()[:, ::-1].astype(np.int32)
            extreme_points = _get_extreme_points(pts).astype(np.int32)
            # top, left, bottom, right, center
            anns_all[i]['keypoints'] = list(itertools.chain.from_iterable([[x, y, 2] for (x, y) in extreme_points.copy().tolist()])) + [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, 2]
            anns_all[i]['num_keypoints'] = 5
            if DEBUG:
                img_id = ann['image_id']
                img_info = coco.loadImgs(ids=[img_id])[0]
                img_path = IMG_DIR.format(split) + img_info['file_name']
                img = cv2.imread(img_path)
                if type(seg) == list:
                    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], (255, 0, 0))
                else:
                    mask = mask.reshape(img.shape[0], img.shape[1], 1)
                img = (0.4 * img + 0.6 * mask).astype(np.uint8)
                bbox = _coco_box_to_bbox(ann['bbox'])
                cl = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
                for j in range(extreme_points.shape[0]):
                    cv2.circle(img, (extreme_points[j, 0], extreme_points[j, 1]),
                               5, cl[j], -1)
                cv2.imshow('img', img)
                cv2.waitKey()
        print('tot_box', tot_box)
        data['annotations'] = anns_all
        json.dump(data, open(OUT_PATH.format(split), 'w'))


def create_detection_keypoints_annotations(detection_annos, humans_only=True):
    def create_bbox_keypoints(bbox: list):
        assert len(bbox) == 4
        [x, y, w, h] = bbox
        # keypoints are represented as 3 values (x coordinate, y coordinate, visibility flag=2)
        return [
            x, y, 2,
            x + w, y, 2,
            x + w / 2, y + h / 2, 2,
            x, y + h, 2,
            x + w, y + h, 2
        ]

    kp_info = dict(description='COCO 2017 detection dataset formatted as keypoint triplets', version='1.0', year=2022,
                   date_created='2022/06/01')
    kp_annotations = [
        dict(
            segmentation=da['segmentation'],
            num_keypoints=5,
            area=da['area'],
            iscrowd=da['iscrowd'],
            keypoints=create_bbox_keypoints(da['bbox']),
            image_id=da['image_id'],
            bbox=da['bbox'],
            category_id=da['category_id'],
            id=da['id']
        )
        for da in detection_annos['annotations'] if (humans_only and da['category_id'] == 1) or not humans_only
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

    for i in range(10, 20):
        test_image_dict = anns['images'][i]
        test_annos = [ann for ann in anns['annotations'] if ann['image_id'] == test_image_dict['id']]
        test_image = Image.open(os.path.join(root_dir + test_image_dict['file_name']))
        display_img_anns(test_image, test_annos, dict(coco=coco, image_id=test_image_dict['id']))

def create_det_keypoint_annotation_file(root_dir, detection_ann_file, output_file_name_template='detection_five_kp_'):
    with open(os.path.join(root_dir, detection_ann_file)) as f:
        detection_annos = json.load(f)

    detection_tripkp_annos = create_detection_keypoints_annotations(detection_annos)

    with open(os.path.join(root_dir, output_file_name_template + detection_ann_file), 'w') as outfile:
        json.dump(detection_tripkp_annos, outfile)


def create_det_person_only_annotation_file(root_dir, detection_ann_file, output_file_name_template='detection_person_'):
    with open(os.path.join(root_dir, detection_ann_file)) as f:
        detection_annos = json.load(f)

    person_anns = []

    for ann in detection_annos['annotations']:
        if ann['category_id'] == 1:
            person_anns.append(ann)

    detection_annos['annotations'] = person_anns
    image_ids = set([ann['image_id'] for ann in detection_annos['annotations']])
    detection_annos['images'] = [img_dict for img_dict in detection_annos['images'] if img_dict['id'] in image_ids]

    with open(os.path.join(root_dir, output_file_name_template + detection_ann_file), 'w') as outfile:
        json.dump(detection_annos, outfile)


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


if __name__ == '__main__':
    import json

    anno_root_dir = '../../data-mscoco/annotations/'
    output_file_name_template = 'detection_five_kp_humans_'
    # create_det_keypoint_annotation_file(anno_root_dir, 'instances_val2017.json', output_file_name_template=output_file_name_template)
    # create_det_keypoint_annotation_file(anno_root_dir, 'instances_train2017.json', output_file_name_template=output_file_name_template)
    # create_det_person_only_annotation_file(anno_root_dir, 'instances_val2017.json')
    # generate_extreme_points_file('../../data-mscoco/')


    # visualise_test(anno_root_dir + 'instances_person_extreme_val2017.json')

    """
    --cocokp-train-annotations=data-mscoco/annotations/detection_triplet_kp_instances_train2017.json
    --cocokp-val-annotations=data-mscoco/annotations/detection_triplet_kp_instances_val2017.json
    """

    # create_det_keypoint_test_anno_file(anno_root_dir)

    with open(anno_root_dir + 'detection_person_instances_val2017.json', 'r') as f:
        test_annos = json.load(f)

    print('Done')
