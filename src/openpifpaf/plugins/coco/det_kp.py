import argparse

import torch

import openpifpaf
from .constants import (
    COCO_CATEGORIES,
)
from .dataset import CocoDataset
from .det_kp_utils import (
    DETKP_SKELETON,
    DETKP_KEYPOINTS,
    DETKP_POSE,
    DETKP_SIGMAS,
    DETKP_SCORE_WEIGHTS, DETKP_HFLIP,
)

try:
    import pycocotools.coco

    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass


class CocoDetKpTriplets(openpifpaf.datasets.DataModule, openpifpaf.Configurable):
    _test2017_annotations = None
    _testdev2017_annotations = None
    _test2017_image_dir = 'data-mscoco/images/test2017/'

    # cli configurable
    train_annotations = 'data-mscoco/annotations/instances_person_extreme_train2017.json'
    val_annotations = 'data-mscoco/annotations/instances_person_extreme_val2017.json'
    # train_annotations = 'data-mscoco/annotations/detection_five_kp_test_person_only.json'
    # val_annotations = train_annotations
    eval_annotations = val_annotations
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017/'
    # val_image_dir = train_image_dir
    eval_image_dir = val_image_dir

    square_edge = 385
    with_dense = False
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 0.1

    eval_annotation_filter = True
    eval_long_edge = 641
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                      keypoints=DETKP_KEYPOINTS,
                                      sigmas=DETKP_SIGMAS,
                                      pose=DETKP_POSE,
                                      draw_skeleton=DETKP_SKELETON,
                                      score_weights=DETKP_SCORE_WEIGHTS)
        caf = openpifpaf.headmeta.Caf('caf', 'cocokp',
                                      keypoints=DETKP_KEYPOINTS,
                                      sigmas=DETKP_SIGMAS,
                                      pose=DETKP_POSE,
                                      skeleton=DETKP_SKELETON)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module CocoDetKpTriplets')

        group.add_argument('--cocodettkp-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--cocodettkp-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--cocodettkp-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--cocodettkp-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')

        group.add_argument('--cocodettkp-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.with_dense
        group.add_argument('--cocodettkp-with-dense',
                           default=False, action='store_true',
                           help='train with dense connections')
        assert not cls.extended_scale
        group.add_argument('--cocodettkp-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--cocodettkp-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--cocodettkp-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--cocodettkp-no-augmentation',
                           dest='cocodettkp_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--cocodettkp-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--cocodettkp-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--cocodettkp-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--cocodettkp-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')

        # evaluation
        eval_set_group = group.add_mutually_exclusive_group()
        eval_set_group.add_argument('--cocodettkp-eval-test2017', default=False, action='store_true')
        eval_set_group.add_argument('--cocodettkp-eval-testdev2017', default=False, action='store_true')

        assert cls.eval_annotation_filter
        group.add_argument('--cocodettkp-no-eval-annotation-filter',
                           dest='coco_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--cocodettkp-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--cocodettkp-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--cocodettkp-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocodettkp specific
        cls.train_annotations = args.cocodettkp_train_annotations
        cls.val_annotations = args.cocodettkp_val_annotations
        cls.train_image_dir = args.cocodettkp_train_image_dir
        cls.val_image_dir = args.cocodettkp_val_image_dir

        cls.square_edge = args.cocodettkp_square_edge
        cls.with_dense = args.cocodettkp_with_dense
        cls.extended_scale = args.cocodettkp_extended_scale
        cls.orientation_invariant = args.cocodettkp_orientation_invariant
        cls.blur = args.cocodettkp_blur
        cls.augmentation = args.cocodettkp_augmentation
        cls.rescale_images = args.cocodettkp_rescale_images
        cls.upsample_stride = args.cocodettkp_upsample
        cls.min_kp_anns = args.cocodettkp_min_kp_anns
        cls.bmin = args.cocodettkp_bmin

        # evaluation
        cls.eval_annotation_filter = args.coco_eval_annotation_filter
        if args.cocodettkp_eval_test2017:
            cls.eval_image_dir = cls._test2017_image_dir
            cls.eval_annotations = cls._test2017_annotations
            cls.annotation_filter = False
        if args.cocodettkp_eval_testdev2017:
            cls.eval_image_dir = cls._test2017_image_dir
            cls.eval_annotations = cls._testdev2017_annotations
            cls.annotation_filter = False
        cls.eval_long_edge = args.coco_eval_long_edge
        cls.eval_orientation_invariant = args.coco_eval_orientation_invariant
        cls.eval_extended_scale = args.coco_eval_extended_scale

        if (args.cocodettkp_eval_test2017 or args.cocodettkp_eval_testdev2017) \
                and not args.write_predictions and not args.debug:
            raise Exception('have to use --write-predictions for this dataset')

    def _preprocess(self):
        encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]
        if len(self.head_metas) > 2:
            encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.4 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(DETKP_KEYPOINTS, DETKP_HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.4],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.KpToDetAnnotations(COCO_CATEGORIES),
                openpifpaf.transforms.ToCrowdAnnotations(COCO_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [openpifpaf.metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=100,
            category_ids=[],
            iou_type='bbox',
        )]
