import json
import os


def main():
    with open('data-mscoco/annotations/person_keypoints_val2017.json', 'r') as f:
        data1 = json.load(f)

    with open('data-mscoco/annotations/instances_val2017.json', 'r') as f:
        data = json.load(f)

    x = [ann for ann in data['annotations'] if ann['category_id'] == 1]

    for i, ann in enumerate(data1['annotations']):
        if ann['bbox'] != x[i]['bbox']:
            print(f'False, {i}')

    print(data)


if __name__ == '__main__':
    os.chdir('/home/albion/')
    main()
