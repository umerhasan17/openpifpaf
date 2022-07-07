from collections import defaultdict

from .preprocess import Preprocess


class Encoders(Preprocess):
    """Preprocess operation that runs encoders."""
    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, image, anns, meta):
        anns = [enc(image, anns, meta) for enc in self.encoders]
        meta['head_indices'] = [enc.meta.head_index for enc in self.encoders]
        return image, anns, meta


class DetKpEncoders(Preprocess):
    """Preprocess operation that runs encoder for current annotation category."""

    def __init__(self, encoders):
        self.encoders = encoders
        self.n = len(self.encoders) // 2

    def __call__(self, image, anns, meta):
        ann_dict = defaultdict(list)
        for ann in anns:
            ann_dict[ann['category_id'] - 1].append(ann)
        encoded_anns = []
        meta['head_indices'] = []

        # encode cifs
        for i in range(self.n):
            encoded_anns.append(self.encoders[i](image, ann_dict[i], meta))
            meta['head_indices'].append(self.encoders[i].meta.head_index)

        # encode cafs
        for i in range(self.n):
            encoded_anns.append(self.encoders[i + self.n](image, ann_dict[i], meta))
            meta['head_indices'].append(self.encoders[i + self.n].meta.head_index)

        return image, encoded_anns, meta
