import logging
from copy import deepcopy
from collections import deque
import albumentations as A
import numpy as np
from PIL import Image
import random
import openpifpaf.transforms
from openpifpaf.transforms import Preprocess

import time

LOG = logging.getLogger(__name__)


class AlbumentationsComposeWrapperMeta(type(Preprocess), type(A.BaseCompose)):
    pass


class AlbumentationsComposeWrapper(Preprocess, A.Compose, metaclass=AlbumentationsComposeWrapperMeta):
    """
    Wrapper class for albumentations compose transform.
    Please note if you are using this to add more functionality from albumentations,
    the call method has been greatly simplified and may lack functionality.

    Paste in annotations from previous image into current image
    """

    def __init__(self, *args, max_annos=50, default_transformations=None, apply_copy_paste=True,
                 max_previous_images=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_copy_paste = apply_copy_paste
        self.previous_image_data = deque()
        self.max_annos = max_annos
        self.max_previous_images = max_previous_images
        self.default_transformations = default_transformations
        if max_annos > 0 and self.apply_copy_paste:
            assert default_transformations is not None
        self.num_images = 8
        self.total_times = 0
        self.calls = 0

    def update_previous_image_data(self, data, anns):
        """
            Masks and bboxes have been edited with cropping and padding.
            Annotations are the original annotations coming into the transformation function.
        """
        data['previous_anns'] = anns
        self.previous_image_data.append(data)
        if len(self.previous_image_data) > self.max_previous_images:
            self.previous_image_data.popleft()

    @staticmethod
    def reformat_annotations(annotations, bboxes):
        """ Update annotations with new bboxes. Note the reformat function removes segmentation details. """
        if len(bboxes) == 0:
            return []
        ann_idx = 5 if len(bboxes[0]) == 6 else 6
        ann_idxs = [bbox[ann_idx] for bbox in bboxes]
        new_annotations = [annotations[idx] for idx in ann_idxs]
        for ix, new_ann in enumerate(new_annotations):
            bbox = bboxes[ix]
            assert bbox[4] == new_ann['category_id']
            new_ann['bbox'] = np.array(bbox[:4])
        return new_annotations

    def copy_paste_augmentation(self, image, anns, meta):
        t1 = time.time()
        # convert target segmentations to masks
        all_masks, all_bboxes = [], []
        for ix, ann in enumerate(anns):
            mask = meta['masks'][ix]
            bbox = ann['bbox'].tolist() + [ann['category_id']] + [ix]
            all_masks.append(mask)
            all_bboxes.append(bbox)

        # pack outputs into a dict
        data = {
            'image': np.asarray(image),
            'masks': all_masks,
            'bboxes': all_bboxes,
        }
        t2 = time.time()
        # apply usual transforms
        data = A.Compose.__call__(self, **data)
        cp_output_data = None
        t3 = time.time()
        t4 = t5 = t6 = t3
        if self.apply_copy_paste:
            if len(self.previous_image_data) > 0:
                copy_paste_image_data = random.choice(self.previous_image_data)
                # LOG.debug('Applying albumentations copy paste transform')
                cp_data = dict(
                    image=data['image'],
                    bboxes=[bbox + (bbox[-1],) for bbox in data['bboxes']],
                    masks=data['masks'],
                    paste_image=copy_paste_image_data['image'],
                    paste_masks=copy_paste_image_data['masks'],
                    paste_bboxes=[bbox + (len(data['masks']) + bbox[-1],) for bbox in
                                  copy_paste_image_data['bboxes']],
                )
                cp_transform = A.Compose(
                    [openpifpaf.transforms.CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1)],
                    bbox_params=A.BboxParams(format="coco")
                )
                cp_output_data = cp_transform(**cp_data)
                # add annotation details back in
                t4 = time.time()
                updated_annotations = self.reformat_annotations(
                    anns + copy_paste_image_data['previous_anns'],
                    cp_output_data['bboxes']
                )
                t5 = time.time()

                # useful debug statement
                # from openpifpaf.transforms.copy_paste.visualize import display_instances
                # import matplotlib.pyplot as plt
                # try:
                #     new_mask_idxs = [x[5] for x in cp_output_data['bboxes']]
                #     COCO_CATEGORIES = [
                #         'person',
                #         'bicycle',
                #         'car',
                #         'motorcycle',
                #         'airplane',
                #         'bus',
                #         'train',
                #         'truck',
                #         'boat',
                #         'traffic light',
                #         'fire hydrant',
                #         'street sign',
                #         'stop sign',
                #         'parking meter',
                #         'bench',
                #         'bird',
                #         'cat',
                #         'dog',
                #         'horse',
                #         'sheep',
                #         'cow',
                #         'elephant',
                #         'bear',
                #         'zebra',
                #         'giraffe',
                #         'hat',
                #         'backpack',
                #         'umbrella',
                #         'shoe',
                #         'eye glasses',
                #         'handbag',
                #         'tie',
                #         'suitcase',
                #         'frisbee',
                #         'skis',
                #         'snowboard',
                #         'sports ball',
                #         'kite',
                #         'baseball bat',
                #         'baseball glove',
                #         'skateboard',
                #         'surfboard',
                #         'tennis racket',
                #         'bottle',
                #         'plate',
                #         'wine glass',
                #         'cup',
                #         'fork',
                #         'knife',
                #         'spoon',
                #         'bowl',
                #         'banana',
                #         'apple',
                #         'sandwich',
                #         'orange',
                #         'broccoli',
                #         'carrot',
                #         'hot dog',
                #         'pizza',
                #         'donut',
                #         'cake',
                #         'chair',
                #         'couch',
                #         'potted plant',
                #         'bed',
                #         'mirror',
                #         'dining table',
                #         'window',
                #         'desk',
                #         'toilet',
                #         'door',
                #         'tv',
                #         'laptop',
                #         'mouse',
                #         'remote',
                #         'keyboard',
                #         'cell phone',
                #         'microwave',
                #         'oven',
                #         'toaster',
                #         'sink',
                #         'refrigerator',
                #         'blender',
                #         'book',
                #         'clock',
                #         'vase',
                #         'scissors',
                #         'teddy bear',
                #         'hair drier',
                #         'toothbrush',
                #         'hair brush',
                #     ]
                #     display_instances(cp_output_data['image'], np.array([x[:4] for x in cp_output_data['bboxes']]),
                #                       np.moveaxis(np.array([cp_output_data['masks'][i] for i in new_mask_idxs]), 0, -1),
                #                       np.array([i for i in range(len(cp_output_data['bboxes']))]),
                #                       np.array([COCO_CATEGORIES[category_id-1] for category_id in list(map(lambda x: x[4], cp_output_data['bboxes']))]),
                #                       call_number=self.calls)
                # except Exception as e:
                #     print('Visualisation failed: ', e)
            else:
                updated_annotations = self.reformat_annotations(anns, data['bboxes'])
            # save current image details for next copy paste augmentation
            self.update_previous_image_data(data, anns)
            t6 = time.time()
        else:
            updated_annotations = self.reformat_annotations(anns, data['bboxes'])

        if cp_output_data is not None:
            data = cp_output_data
        t7 = time.time()
        total_time = t7 - t1
        self.total_times += total_time
        r_func = lambda x: round(x, 4)
        # if total_time >= 0.9:
        meta['masks'] = data['masks']
        LOG.debug(
            f't1->t2: {r_func(t2 - t1)}, t2->t3: {r_func(t3 - t2)}, t3->t4: {r_func(t4 - t3)}, t4->t5: {r_func(t5 - t4)}, '
            f't5->t6: {r_func(t6 - t5)}, t6->t7: {r_func(t7 - t6)}, total: {r_func(total_time)}, '
            f'len annos: {len(updated_annotations)}, '
            f'image id: {updated_annotations[0]["image_id"] if len(updated_annotations) > 0 else -1}')
        return Image.fromarray(data['image']), updated_annotations, meta

    def __call__(self, image, anns, meta):
        self.calls += 1
        # if self.calls % self.num_images == 0:
        #     LOG.info(f'Total time for {self.num_images}: {self.total_times}')
        #     self.total_times = 0
        # LOG.debug('Applying albumentations transforms')
        t1 = time.time()
        if len(anns) > self.max_annos:
            image, anns, meta = self.default_transformations(image, anns, meta)
            LOG.debug(f'Default transformations took {round(time.time() - t1, 4)} seconds.')
            return image, anns, meta

        # LOG.debug('Applying albumentations transforms')

        # may be unnecessary if not applied with other transforms
        # necessary if applying transformations which depend on metadata after this transformation
        # self.adjust_meta_img_dimensions(image, anns, meta)
        try:
            return self.copy_paste_augmentation(image, anns, meta)
        except Exception as e:
            LOG.warning(f'Exception during copy paste augmentation, using default transformations instead: {e}')
            return self.default_transformations(image, anns, meta)
