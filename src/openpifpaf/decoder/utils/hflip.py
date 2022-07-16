import torch


def hflip_average_fields_batch(fields_batch, hflip_fields_batch, head_metas):
    hflip_funcs = []
    for head_meta in head_metas:
        if head_meta.name == 'cifdet':
            hflip_func = hflip_average_cifdet_fields_batch
        elif head_meta.name == 'cif':
            hflip_func = hflip_average_cif_fields_batch
        elif head_meta.name == 'caf':
            hflip_func = hflip_average_caf_fields_batch
        else:
            raise ValueError(f'Unsupported head meta for hflip: {head_meta.name}.')
        hflip_funcs.append(hflip_func)

    for i, current_batch in enumerate(fields_batch):
        assert len(current_batch) == len(head_metas)
        for j, field_set in enumerate(current_batch):
            # additional processing for hflip field set specific to heads used
            hflip_field_set = hflip_funcs[j](hflip_fields_batch[i][j])

            # take an average of both fields
            field_set = field_set.add(hflip_field_set)
            field_set = torch.div(field_set, 2)
            fields_batch[i][j] = field_set

    return fields_batch


def hflip_average_cifdet_fields_batch(hflip_field_set):
    # deal with vector offsets for x regression field
    fields_shape = hflip_field_set.shape
    offset_tensor = torch.arange(fields_shape[3]).repeat(fields_shape[0], fields_shape[2], 1)
    # remove offset
    hflip_field_set[:, 2, :, :] = hflip_field_set[:, 2, :, :].subtract(offset_tensor)
    # horizontally flip all fields
    hflip_field_set = torch.flip(hflip_field_set, [-1])
    # negate x regression field
    hflip_field_set[:, 2, :, :] = torch.neg(hflip_field_set[:, 2, :, :])
    # add back offset
    hflip_field_set[:, 2, :, :] = hflip_field_set[:, 2, :, :].add(offset_tensor)
    return hflip_field_set


def hflip_average_cif_fields_batch(hflip_field_set):
    return hflip_field_set


def hflip_average_caf_fields_batch(hflip_field_set):
    return hflip_field_set

