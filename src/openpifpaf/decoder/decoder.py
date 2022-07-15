import argparse
import logging
import multiprocessing
import sys
import time
from typing import List

import torch

from .. import annotation, visualizer

LOG = logging.getLogger(__name__)


class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


class Decoder:
    """Generate predictions from image or field inputs.

    When creating a new generator, the main implementation goes into `__call__()`.
    """
    default_worker_pool = None
    torch_decoder = True

    def __init__(self):
        self.priority = 0.0  # reference priority for single image CifCaf
        self.worker_pool = self.default_worker_pool

        if self.worker_pool is None or self.worker_pool == 0:
            self.worker_pool = DummyPool()
        if isinstance(self.worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', self.worker_pool)
            assert not sys.platform.startswith('win'), (
                'not supported, use --decoder-workers=0 '
                'on windows'
            )

            # The new default for multiprocessing is 'spawn' for py38 on Mac.
            # This is not compatible with our configuration system.
            # For now, try to use 'fork'.
            # TODO: how to make configuration 'spawn' compatible
            multiprocessing_context = multiprocessing.get_context('fork')
            self.worker_pool = multiprocessing_context.Pool(self.worker_pool)

        self.last_decoder_time = 0.0
        self.last_nn_time = 0.0

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    @classmethod
    def factory(cls, head_metas) -> List['Decoder']:
        """Create instances of an implementation."""
        raise NotImplementedError

    def __call__(self, fields, *, initial_annotations=None) -> List[annotation.Base]:
        """For single image, from fields to annotations."""
        raise NotImplementedError

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ('worker_pool',)
        }

    @classmethod
    def fields_batch(cls, model, image_batch, *, device=None):
        """From image batch to field batch."""
        start = time.time()

        def apply(f, items):
            """Apply f in a nested fashion to all items that are not list or tuple."""
            if items is None:
                return None
            if isinstance(items, (list, tuple)):
                return [apply(f, i) for i in items]
            return f(items)

        with torch.no_grad():
            if device is not None:
                image_batch = image_batch.to(device, non_blocking=True)

            with torch.autograd.profiler.record_function('model'):
                heads = model(image_batch)

            # to numpy
            with torch.autograd.profiler.record_function('tonumpy'):
                if cls.torch_decoder:
                    heads = apply(lambda x: x.cpu(), heads)
                else:
                    heads = apply(lambda x: x.cpu().numpy(), heads)

        # index by frame (item in batch)
        head_iter = apply(iter, heads)
        heads = []
        while True:
            try:
                heads.append(apply(next, head_iter))
            except StopIteration:
                break

        LOG.debug('nn processing time: %.1fms', (time.time() - start) * 1000.0)
        return heads

    def batch(self, model, image_batch, *, device=None, hflip=False, gt_anns_batch=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batch = self.fields_batch(model, image_batch, device=device)

        if hflip:
            # take horizontal flipped image and generate fields
            # then average the fields with the original fields before decoding
            hflip_image_batch = torch.flip(image_batch, [-1])
            hflip_fields_batch = self.fields_batch(model, hflip_image_batch, device=device)
            image_batch = hflip_image_batch
            for i, fields in enumerate(fields_batch):
                for j, field_set in enumerate(fields):
                    hflip_field_set = hflip_fields_batch[i][j]
                    # deal with vector offsets for x regression field
                    fields_shape = field_set.shape
                    offset_tensor = torch.arange(fields_shape[3]).repeat(fields_shape[0], fields_shape[2], 1)
                    # remove offset
                    hflip_field_set[:, 2, :, :] = hflip_field_set[:, 2, :, :].subtract(offset_tensor)
                    # horizontally flip all fields
                    hflip_field_set = torch.flip(hflip_field_set, [-1])
                    # negate x regression field
                    hflip_field_set[:, 2, :, :] = torch.neg(hflip_field_set[:, 2, :, :])
                    # add back offset
                    hflip_field_set[:, 2, :, :] = hflip_field_set[:, 2, :, :].add(offset_tensor)

                    import matplotlib.pyplot as plt
                    from PIL import Image
                    from mpl_toolkits.axes_grid1 import make_axes_locatable

                    # def field_plot_debug(cur_field, field_num=0, axarr=None, indices=None):
                    #     for i, (v1, v2) in enumerate(indices):
                    #         cur_img_arr = cur_field[field_num, i, :, :].detach().cpu().numpy()
                    #         cur_img = axarr[v1, v2].imshow(cur_img_arr)
                    #         axarr[v1, v2].set_xlabel(f'Field number {i}')
                    #         divider = make_axes_locatable(axarr[v1, v2])
                    #         cax = divider.append_axes('right', size='5%', pad=0.05)
                    #         f.colorbar(cur_img, cax=cax, orientation='vertical')

                    # def compare_field_plot_debug(f1, f2, class_num=0, field_index=2, difference=False):
                    #     if difference:
                    #         plt.imshow(f1.subtract(f2)[class_num, field_index, :, :].detach().cpu().numpy())
                    #         plt.colorbar()
                    #     else:
                    #         f, axarr = plt.subplots(1, 2, figsize=(15, 15))
                    #         plots = [(0, f1), (1, f2)]
                    #         for arr_idx, field in plots:
                    #             cur_img = axarr[arr_idx].imshow(field[class_num, field_index, :, :].detach().cpu().numpy())
                    #             divider = make_axes_locatable(axarr[arr_idx])
                    #             cax = divider.append_axes('right', size='5%', pad=0.05)
                    #             f.colorbar(cur_img, cax=cax, orientation='vertical')
                    #     plt.show()
                    #     plt.clf()
                    #
                    # compare_field_plot_debug(field_set, hflip_field_set, difference=True)
                    # compare_field_plot_debug(field_set, hflip_field_set, field_index=1, difference=True)
                    # compare_field_plot_debug(field_set, hflip_field_set, field_index=3, difference=True)
                    # compare_field_plot_debug(field_set, hflip_field_set, field_index=4, difference=True)
                    # compare_field_plot_debug(field_set, hflip_field_set, field_index=5, difference=True)



                    # f, axarr = plt.subplots(3, 2, figsize=(15, 15))
                    # field_plot_debug(field_set, axarr=axarr, indices=[(x, y) for x in range(3) for y in range(2)])
                    # plt.show()
                    # plt.clf()
                    # f, axarr = plt.subplots(3, 2, figsize=(15, 15))
                    # field_plot_debug(hflip_field_set, axarr=axarr, indices=[(x, y) for x in range(3) for y in range(2)])
                    # plt.show()
                    # plt.clf()

                    # take an average of both fields
                    field_set = field_set.add(hflip_field_set)
                    field_set = torch.div(field_set, 2)
                    fields_batch[i][j] = field_set

        self.last_nn_time = time.perf_counter() - start_nn

        if gt_anns_batch is None:
            gt_anns_batch = [None for _ in fields_batch]

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]
            gt_anns_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch, gt_anns_batch))
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug('time: nn = %.1fms, dec = %.1fms',
                  self.last_nn_time * 1000.0,
                  self.last_decoder_time * 1000.0)
        return result

    def _mappable_annotations(self, fields, debug_image, gt_anns):
        if debug_image is not None:
            visualizer.Base.processed_image(debug_image)
        if gt_anns is not None:
            visualizer.Base.ground_truth(gt_anns)

        return self(fields)
