
TRAIN_COMMAND = [
    'python3', '-m', 'openpifpaf.train',
    '--dataset=cocokp',
    '--lr=1e-3',
    '--momentum=0.9',
    '--epochs=1',
    '--batch-size=1',
    '--basenet=resnet18', '--resnet-no-pretrain',
    '--cocokp-upsample=2',
    '--cocokp-square-edge=97',
    '--cocokp-train-annotations', 'tests/coco/train1.json',
    '--cocokp-train-image-dir', 'tests/coco/images/',
    '--cocokp-val-annotations', 'tests/coco/train1.json',
    '--cocokp-val-image-dir', 'tests/coco/images/',
]


if __name__ == '__main__':
    print(' '.join(TRAIN_COMMAND))