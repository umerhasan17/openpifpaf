import numpy as np

NUM_KEYPOINTS = 2


def create_keypoints_from_bbox(bbox, num_keypoints=NUM_KEYPOINTS):
    assert len(bbox) == 4
    [x, y, w, h] = bbox
    if num_keypoints == 2:
        kps = np.array([[x, y, 2], [x + w, y + h, 2]])
    elif num_keypoints == 3:
        kps = np.array([[x, y, 2], [x + (w / 2), y + (h / 2), 2], [x + w, y + h, 2]])
    elif num_keypoints == 5:
        kps = np.array(
            [[x, y, 2], [x + w, y, 2], [x + (w / 2), y + (h / 2), 2], [x, y + h, 2], [x + w, y + h, 2]]
        )
    else:
        raise ValueError('Invalid number of keypoints')

    return kps
