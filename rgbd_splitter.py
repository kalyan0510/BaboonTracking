import sys

import cv2 as cv
from tqdm import tqdm

from ps_hepers.helpers import get_png_files, load_png, imshow


def split_images(base_path, save_path):
    image_paths = get_png_files(base_path)
    zipped_imgs = zip(image_paths, load_png([base_path + image_path for image_path in image_paths]))

    for (path, img) in zipped_imgs:
        rgb = img[:, :1279, :]
        d = img[:, 1280:, :]
        cv.imwrite(save_path + "RGB\\" + path.split('.png')[0] + '_RGB.png', rgb)
        cv.imwrite(save_path + "Depth\\" + path.split('.png')[0] + '_Depth.png', d)


def read_fps(cap):
    (major_ver, minor_ver, subminor_ver) = cv.__version__.split('.')
    return cap.get(cv.cv.CV_CAP_PROP_FPS) if (int(major_ver) < 3) else cap.get(cv.CAP_PROP_FPS)


def split_video(input_path, save_path, is_depth):
    cap = cv.VideoCapture(input_path)
    ret, frame = cap.read()
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(save_path, fourcc, read_fps(cap), (frame.shape[1] // 2, frame.shape[0]))
    total_f_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(total_f_count)):
        ret, frame = cap.read()
        if ret:
            b = frame[:, frame.shape[1] // 2:, :] if is_depth else frame[:, :frame.shape[1] // 2, :]
            out.write(b)
        else:
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    args = sys.argv[1:]
    base_path = 'W:\\CV\\Baboons\\Images for Annotation\\'
    save_path = 'W:\\CV\\Baboons\\Images for Annotation\\Split\\'
    isVid = lambda path: path.split('.')[-1] in ['avi', 'mp4', 'mpg']
    if len(args) == 0:
        split_images(base_path, save_path)

    if isVid(args[0]) and isVid(args[1]):
        split_video(args[0], [args[1], args[0]][len(args) <= 1], [args[2], 'rgb'][len(args) <= 2] in ['D', 'd'])
    else:
        split_images(base_path, save_path)
