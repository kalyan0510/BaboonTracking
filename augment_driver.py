import random

import cv2 as cv
import numpy as np
from tqdm import tqdm
from pathlib import Path

from augmentors.augmenter import Augmenter
from augmentors.flip import FlipAugmentor
from augmentors.shuffle_object import ShuffleObjectAugmentor
from augmentors.translate import TranslateAugmentor
from ps_hepers.helpers import get_png_files, load_png, imshow


def preview_annotation(image, annotations):
    image = image.copy()
    (h, w) = image.shape[:2]
    to_pixel_space = lambda x: (x * np.asarray([1, w, h, w, h])).astype(int)
    for anno in annotations:
        # continue
        a = to_pixel_space(anno)
        if ((a[2] - a[4] // 2) > 0 and (a[2] + a[4] // 2) < image.shape[0] and (a[1] - a[3] // 2) > 0 and (
                a[1] + a[3] // 2) < image.shape[1]):
            image[a[2] - a[4] // 2: a[2] + a[4] // 2, a[1] - a[3] // 2, ...] = 255
            image[a[2] - a[4] // 2: a[2] + a[4] // 2, a[1] + a[3] // 2, ...] = 255
            image[a[2] - a[4] // 2, a[1] - a[3] // 2: a[1] + a[3] // 2, ...] = 255
            image[a[2] + a[4] // 2, a[1] - a[3] // 2: a[1] + a[3] // 2, ...] = 255
    imshow(image)


def read_annotations_file(file_path):
    """list of annotations of format : <object-class> <x> <y> <width> <height>"""
    with open(file_path) as file:
        lines = file.readlines()
        lines = np.asarray([list(map(float, line.split(' '))) for line in lines])
    return lines


def save_annotation_file(file_path, annotations):
    with open(file_path, 'w') as f:
        for anno in annotations:
            f.write("%d %f %f %f %f\n" % (anno[0], anno[1], anno[2], anno[3], anno[4]))


if __name__ == '__main__':
    base_path = 'W:\\CV\\Baboons\\yolo annotations\\all\\obj_train_data\\'
    save_path = 'W:\\CV\\Baboons\\AugmentedImages\\'
    Path(save_path + 'obj_train_data').mkdir(parents=True, exist_ok=True)

    image_paths = get_png_files(base_path)[::-1]
    # image_paths = [image_paths[i] for i in [0,44,42]]
    annotation_paths = [p[:-4] + '.txt' for p in image_paths]
    # print(image_paths, annotation_paths)
    # zipped_imgs = zip(image_paths, load_png([base_path + image_path for image_path in image_paths]))
    all_image_paths = []
    bar = tqdm(range(len(image_paths)))
    for i in bar:
        img = load_png([base_path + image_paths[i]], [0, 1, 2])[0]
        annotations = read_annotations_file(base_path + annotation_paths[i])
        augmentor = Augmenter()
        augmentor.add_augmentor(FlipAugmentor(axis=FlipAugmentor.FLIP_HORIZONTAL))
        augmentor.add_augmentor(
            TranslateAugmentor(axis=TranslateAugmentor.TRANSLATE_HORIZONTAL, max_trans_ratio=0.3, max_aug=6))
        augmentor.add_augmentor(
            TranslateAugmentor(axis=TranslateAugmentor.TRANSLATE_VERTICAL, max_trans_ratio=0.2, max_aug=1))
        augmentor.add_augmentor(ShuffleObjectAugmentor(max_obj_shuffle=3, max_aug=4))
        # preview_annotation(img, annotations)
        augmentation_data = augmentor.augment(img, annotations)
        bar.set_description("Augmentations per image: %s, mean: %.3f, total: %d\t" % (
        len(augmentation_data), len(all_image_paths) / (i + 1), len(all_image_paths)))
        for aug_im, anno, aug_name in augmentation_data:
            preview_annotation(aug_im, anno)
            # cv.imwrite(save_path + "RGB\\" + path.split('.png')[0] + '_RGB.png', rgb)
            path = save_path + 'obj_train_data\\' + image_paths[i][:-4] + '_' + aug_name
            img_path = path + image_paths[i][-4:]
            all_image_paths.append(image_paths[i][:-4] + '_' + aug_name + image_paths[i][-4:])
            anno_path = path + '.txt'
            # print(img_path)
            cv.imwrite(img_path, aug_im[:, :, [2, 1, 0]])
            save_annotation_file(anno_path, anno)
    # random.shuffle(all_image_paths)
    with open(save_path + 'train.txt', 'w') as f:
        for image_path in all_image_paths:
            f.write("data/obj_train_data/%s\n" % image_path)

"""     ====================  DONE  ====================     """
# h,w = (img.shape[:2])
# img[0:h//2, 0:w//2] = [255, 0, 0]
# img[h//2:h, 0:w//2] = [0, 255, 0]
# img[0:h//2, w//2:w] = [0, 0, 255]
# img[h//2:h, w//2:w] = [255, 255, 0]
# annotations = [np.asarray([0,0.5,0.5,0.1,0.1]), np.asarray([0,0.1,0.5,0.1,0.1]), np.asarray([0,0.5,0.1,0.1,0.1])]
