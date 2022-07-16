import torch

"""
Calculated using the below line. Code from specific plugins cannot be imported into a generic decoder. 
Change indices if the keypoint structure changes. 
During hflipping it will be necessary to average the left eye with the hflipped right eye (they point to the same keypoint).
It is not possible to average together fields for the left eye and the hflpped left eye again.

print({COCO_KEYPOINTS.index(i): COCO_KEYPOINTS.index(HFLIP[i]) for i in HFLIP})

The caf indices are selected manually to perform correct averaging i.e. average fields for left_ankle-left knee 
association with right_ankle-right_knee association in the hflipped image.
"""

human_pose_hflip_cif_indices = {1: 2, 3: 4, 5: 6, 7: 8, 9: 10, 11: 12, 13: 14, 15: 16}
human_pose_hflip_caf_indices = {0: 2, 1: 3, 5: 6, 8: 9, 10: 11, 13: 14, 15: 16, 17: 18}


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

            # TODO remove this debug code
            # if hflip_funcs[j] == hflip_average_caf_fields_batch:
            #     hflip_field_set = field_set
            #     # for swp_idx in range(field_set.shape[0]):
            #     # for swp_idx in [4, 7, 12]:
            #     #     hflip_field_set[swp_idx, :, :, :] = field_set[swp_idx, :, :, :]
            # if hflip_funcs[j] == hflip_average_cif_fields_batch:
            #     hflip_field_set[:, 2, :, :] = field_set[:, 2, :, :]
            #     run_debug(field_set, hflip_field_set)

            # take an average of both fields
            field_set = field_set.add(hflip_field_set)
            field_set = torch.div(field_set, 2)
            fields_batch[i][j] = field_set

    return fields_batch


def hflip_handle_reg_x_offset(hflip_field_set, offset_field_index=2):
    # horizontally flip field to perform offset (reverse the operation of flipping all fields)
    hflip_field_set[:, offset_field_index, :, :] = torch.flip(hflip_field_set[:, offset_field_index, :, :], [-1])
    # deal with vector offsets for x regression field
    fields_shape = hflip_field_set.shape
    offset_tensor = torch.arange(fields_shape[3]).repeat(fields_shape[0], fields_shape[2], 1)
    # remove offset
    hflip_field_set[:, offset_field_index, :, :] = hflip_field_set[:, offset_field_index, :, :].subtract(offset_tensor)
    # horizontally flip field again
    hflip_field_set[:, offset_field_index, :, :] = torch.flip(hflip_field_set[:, offset_field_index, :, :], [-1])
    # negate x regression field
    hflip_field_set[:, offset_field_index, :, :] = torch.neg(hflip_field_set[:, offset_field_index, :, :])
    # add back offset
    hflip_field_set[:, offset_field_index, :, :] = hflip_field_set[:, offset_field_index, :, :].add(offset_tensor)
    return hflip_field_set


def hflip_handle_reg_y_offset(hflip_field_set, offset_field_index=3):
    # horizontally flip field to perform offset (reverse the operation of flipping all fields)
    # hflip_field_set[:, offset_field_index, :, :] = torch.flip(hflip_field_set[:, offset_field_index, :, :], [-1])
    # deal with vector offsets for y regression field
    fields_shape = hflip_field_set.shape
    offset_tensor = torch.transpose(torch.arange(fields_shape[2]).repeat(fields_shape[0], fields_shape[3], 1), 1, 2)
    # remove offset
    hflip_field_set[:, offset_field_index, :, :] = hflip_field_set[:, offset_field_index, :, :].subtract(offset_tensor)
    # horizontally flip all fields
    # hflip_field_set[:, offset_field_index, :, :] = torch.flip(hflip_field_set[:, offset_field_index, :, :], [-1])
    # negate y regression field
    # hflip_field_set[:, offset_field_index, :, :] = torch.neg(hflip_field_set[:, offset_field_index, :, :])
    # add back offset
    hflip_field_set[:, offset_field_index, :, :] = hflip_field_set[:, offset_field_index, :, :].add(offset_tensor)
    return hflip_field_set


def hflip_average_cifdet_fields_batch(hflip_field_set):
    hflip_field_set = torch.flip(hflip_field_set, [-1])
    hflip_field_set = hflip_handle_reg_x_offset(hflip_field_set)
    return hflip_field_set


def hflip_swap_keypoints(hflip_field_set, swap_indices):
    for field_idx, swap_field_idx in swap_indices.items():
        # operations have to be on same line for correct swapping
        hflip_field_set[field_idx, :, :, :], hflip_field_set[swap_field_idx, :, :, :] = \
            hflip_field_set[swap_field_idx, :, :, :], hflip_field_set[field_idx, :, :, :]
    return hflip_field_set


def hflip_swap_caf_vectors(hflip_field_set, swap_indices):
    return hflip_field_set


def hflip_average_cif_fields_batch(hflip_field_set):
    hflip_field_set = torch.flip(hflip_field_set, [-1])
    hflip_field_set = hflip_handle_reg_x_offset(hflip_field_set)
    # hflip_field_set = hflip_handle_reg_y_offset(hflip_field_set)
    hflip_field_set = hflip_swap_keypoints(hflip_field_set, swap_indices=human_pose_hflip_cif_indices)
    return hflip_field_set


def hflip_average_caf_fields_batch(hflip_field_set):
    hflip_field_set = hflip_handle_reg_x_offset(hflip_field_set, offset_field_index=2)
    hflip_field_set = hflip_handle_reg_x_offset(hflip_field_set, offset_field_index=4)
    hflip_field_set = hflip_swap_keypoints(hflip_field_set, swap_indices=human_pose_hflip_caf_indices)
    # 4? left_hip - right_hip 7? left_shoulder - right_shoulder 12
    hflip_field_set = hflip_swap_caf_vectors(hflip_field_set, swap_indices=[4, 7, 12])
    return hflip_field_set


def run_debug(field_set, hflip_field_set):
    import matplotlib.pyplot as plt
    from PIL import Image
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def field_plot_debug(cur_field, field_num=0, f=None, axarr=None, indices=None):
        for i, (v1, v2) in enumerate(indices):
            cur_img_arr = cur_field[field_num, i, :, :].detach().cpu().numpy()
            cur_img = axarr[v1, v2].imshow(cur_img_arr)
            axarr[v1, v2].set_xlabel(f'Field number {i}')
            divider = make_axes_locatable(axarr[v1, v2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(cur_img, cax=cax, orientation='vertical')

    def compare_field_plot_debug(f1, f2, class_num=0, field_index=2, difference=False):
        if difference:
            plt.imshow(f1.subtract(f2)[class_num, field_index, :, :].detach().cpu().numpy())
            plt.colorbar()
        else:
            f, axarr = plt.subplots(1, 2, figsize=(15, 15))
            plots = [(0, f1), (1, f2)]
            for arr_idx, field in plots:
                cur_img = axarr[arr_idx].imshow(field[class_num, field_index, :, :].detach().cpu().numpy())
                divider = make_axes_locatable(axarr[arr_idx])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                f.colorbar(cur_img, cax=cax, orientation='vertical')
        plt.show()
        plt.clf()

    # compare_field_plot_debug(field_set, hflip_field_set, difference=True)
    # compare_field_plot_debug(field_set, hflip_field_set, field_index=1, difference=True)
    # compare_field_plot_debug(field_set, hflip_field_set, field_index=3, difference=True)
    # compare_field_plot_debug(field_set, hflip_field_set, field_index=4, difference=True)
    # compare_field_plot_debug(field_set, hflip_field_set, field_index=5, difference=True)

    def cif_subplot(current_field_set):
        f, axarr = plt.subplots(3, 2, figsize=(15, 15))
        field_plot_debug(current_field_set, f=f, axarr=axarr, indices=[(x, y) for x in range(3) for y in range(2)][:5])
        plt.show()
        plt.clf()

    cif_subplot(field_set)
    cif_subplot(hflip_field_set)
    compare_field_plot_debug(field_set, hflip_field_set, difference=True)