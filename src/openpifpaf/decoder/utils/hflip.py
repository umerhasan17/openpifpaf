"""
Helper methods for horizontally flipping field representations of the image during evaluation.
"""

import torch


def hflip_average_fields_batch(fields_batch, hflip_fields_batch, head_metas):
    """ Entrypoint function for horizontal flipping. """
    hflip_funcs = []
    for head_meta in head_metas:
        if head_meta.name == 'cifdet':
            hflip_func = hflip_average_cifdet_fields_batch
        else:
            raise ValueError(f'Unsupported head meta for hflip: {head_meta.name}.')
        hflip_funcs.append(hflip_func)

    for i, current_batch in enumerate(fields_batch):
        assert len(current_batch) == len(head_metas)
        for j, field_set in enumerate(current_batch):
            # Additional processing for hflip field set specific to heads used.
            hflip_field_set = hflip_funcs[j](hflip_fields_batch[i][j])

            # Take an average of both fields for final fields batch prediction.
            field_set = field_set.add(hflip_field_set)
            field_set = torch.div(field_set, 2)
            fields_batch[i][j] = field_set

    return fields_batch


def hflip_handle_reg_x_offset(hflip_field_set, offset_field_index=2):
    """ Handle the set of x regression fields that the cifdet head produces. """
    # Horizontally flip field to perform offset (reverse the operation of flipping all fields)
    hflip_field_set[:, offset_field_index, :, :] = torch.flip(hflip_field_set[:, offset_field_index, :, :], [-1])
    # Deal with vector offsets for x regression field
    fields_shape = hflip_field_set.shape
    offset_tensor = torch.arange(fields_shape[3]).repeat(fields_shape[0], fields_shape[2], 1)
    # Remove offset
    hflip_field_set[:, offset_field_index, :, :] = hflip_field_set[:, offset_field_index, :, :].subtract(offset_tensor)
    # Horizontally flip field again
    hflip_field_set[:, offset_field_index, :, :] = torch.flip(hflip_field_set[:, offset_field_index, :, :], [-1])
    # Negate x regression field
    hflip_field_set[:, offset_field_index, :, :] = torch.neg(hflip_field_set[:, offset_field_index, :, :])
    # Add back offset
    hflip_field_set[:, offset_field_index, :, :] = hflip_field_set[:, offset_field_index, :, :].add(offset_tensor)
    return hflip_field_set


def hflip_average_cifdet_fields_batch(hflip_field_set):
    """ Function returns the horizontally flipped set of cifdet fields used in object detection tasks. """
    hflip_field_set = torch.flip(hflip_field_set, [-1])
    hflip_field_set = hflip_handle_reg_x_offset(hflip_field_set)
    return hflip_field_set
