import cv2
import cv2 as cv
import os

import numpy as np

from ps_hepers.helpers import imshow, load_png


def select_color_range(img, c_range=[255, 255, 255], tolerance=10):
    mask = np.zeros(img.shape[:2]) == 0
    for i in range(3):
        mask = mask & ((c_range[i] - tolerance < img[:, :, i]) & (img[:, :, i] < c_range[i] + tolerance))
    return mask


def get_connected_segment(im_mask, point):
    new_mask = im_mask.copy()
    new_mask[...] = False

    def expand(i, j):
        if 0 <= i < im_mask.shape[0] and 0 <= j < im_mask.shape[1] and new_mask[i, j] == False and im_mask[i, j]:
            new_mask[i, j] = True
            print(i, j)
            expand(i + 1, j)
            expand(i - 1, j)
            expand(i, j + 1)
            expand(i, j - 1)

    expand(point[0], point[1])
    return new_mask


if __name__ == '__main__':
    zipped_imgs = zip(["wallpaper_new"], load_png(['wallpaper.png']))
    print(zipped_imgs)
    for (path, img) in zipped_imgs:
        new_img = img.copy()
        mask = select_color_range(img, [120, 120, 120], 50)
        # new_mask = get_connected_segment(mask, (10,10))
        labels, labeled_mask = cv2.connectedComponents(mask.astype(np.uint8), None, 4, cv.CV_16U)
        print(labels)
        new_img[labeled_mask == labeled_mask[10, 10]] = [0, 0, 0]
        imshow(new_img)
        cv.imwrite(path + '.png', np.zeros((1920, 1080, 3)))
