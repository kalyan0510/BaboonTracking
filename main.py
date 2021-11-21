import cv2 as cv
import os

from ps_hepers.helpers import get_png_files, load_png

if __name__ == '__main__':
    base_path = 'W:\\CV\\Baboons\\Images for Annotation\\'
    save_path = 'W:\\CV\\Baboons\\Images for Annotation\\Split\\'
    image_paths = get_png_files(base_path)
    zipped_imgs = zip(image_paths, load_png([base_path + image_path for image_path in image_paths]))

    for (path, img) in zipped_imgs:
        rgb = img[:, :1279, :]
        d = img[:, 1280:, :]
        cv.imwrite(save_path + "RGB\\" + path.split('.png')[0] + '_RGB.png', rgb)
        cv.imwrite(save_path + "Depth\\" + path.split('.png')[0] + '_Depth.png', d)
