import matplotlib.pyplot as plt
import numpy as np

from ..annotation import Annotation, AnnotationCrowd, AnnotationDet
from .preprocess import Preprocess


class ToAnnotations(Preprocess):
    """Convert inputs to annotation objects."""

    def __init__(self, converters):
        self.converters = converters

    def __call__(self, image, anns, meta):
        anns = [
            ann
            for converter in self.converters
            for ann in converter(anns)
        ]
        return image, anns, meta


class ToKpAnnotations:
    """Input to keypoint annotations."""

    def __init__(self, categories, keypoints_by_category, skeleton_by_category):
        self.keypoints_by_category = keypoints_by_category
        self.skeleton_by_category = skeleton_by_category
        self.categories = categories

    def __call__(self, anns):
        return [
            Annotation(
                self.keypoints_by_category[ann['category_id']],
                self.skeleton_by_category[ann['category_id']],
                categories=self.categories,
            )
            .set(
                ann['keypoints'],
                category_id=ann['category_id'],
                fixed_score='',
                fixed_bbox=ann.get('bbox'),
            )
            for ann in anns
            if not ann['iscrowd'] and np.any(ann['keypoints'][2::3] > 0.0)
        ]


class KpToDetAnnotations:
    """ Converting triplet keypoints to detection annotations """
    def __init__(self, categories):
        self.categories = categories

    def __call__(self, anns):
        def create_bbox(ann_keypoints):
            assert len(ann_keypoints) == 5
            x, y = ann_keypoints[0][:2]
            w = ann_keypoints[1][0] - ann_keypoints[0][0]
            h = ann_keypoints[3][1] - ann_keypoints[0][1]
            assert w >= 0 and h >= 0
            return np.array([x, y, w, h])

        return [
            AnnotationDet(categories=self.categories)
                .set(
                ann['category_id'],
                None,
                create_bbox(ann['keypoints']),
            )
            for ann in anns
            if not ann['iscrowd'] and np.any(ann['bbox'])
        ]


class ToDetAnnotations:
    """Input to detection annotations."""

    def __init__(self, categories):
        self.categories = categories

    def __call__(self, anns):
        return [
            AnnotationDet(categories=self.categories)
            .set(
                ann['category_id'],
                None,
                ann['bbox'],
            )
            for ann in anns
            if not ann['iscrowd'] and np.any(ann['bbox'])
        ]


class ToCrowdAnnotations:
    """Input to crowd annotations."""

    def __init__(self, categories):
        self.categories = categories

    def __call__(self, anns):
        return [
            AnnotationCrowd(categories=self.categories)
            .set(
                ann.get('category_id', 1),
                ann['bbox'],
            )
            for ann in anns
            if ann['iscrowd']
        ]
