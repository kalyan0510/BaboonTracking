import numpy as np


class TranslateAugmentor:
    TRANSLATE_HORIZONTAL = 'horizontal'
    TRANSLATE_VERTICAL = 'vertical'
    TRANSLATE_BOTH = 'both'

    def __init__(self, axis=TRANSLATE_BOTH, max_trans_ratio=0.1, max_aug=6):
        self.axis = axis
        temp = np.array([self.TRANSLATE_HORIZONTAL, self.TRANSLATE_VERTICAL, self.TRANSLATE_BOTH])
        self.max_horizontal_aug = ((temp == axis) * np.array([max_aug, 0, max_aug // 2])).max()
        self.max_vertical_aug = ((temp == axis) * np.array([0, max_aug, max_aug - (max_aug // 2)])).max()
        # print(self.max_horizontal_aug, self.max_vertical_aug)
        self.max_trans_ratio = max_trans_ratio

    def translate_image(self, image, dist_ratio, axis):
        if dist_ratio > 0:
            if axis is self.TRANSLATE_HORIZONTAL:
                dist = int(dist_ratio * image.shape[1])
                return image[:, np.concatenate([np.arange(dist, image.shape[1]), np.arange(0, dist)], axis=0), ...]
            if axis is self.TRANSLATE_VERTICAL:
                dist = int(dist_ratio * image.shape[0])
                return image[np.concatenate([np.arange(dist, image.shape[0]), np.arange(0, dist)], axis=0), :, ...]
        if dist_ratio < 0:
            if axis is self.TRANSLATE_HORIZONTAL:
                dist = int(-dist_ratio * image.shape[1])
                return image[:, np.concatenate([np.arange(-dist, 0), np.arange(0, image.shape[1] - dist)], axis=0), ...]
            if axis is self.TRANSLATE_VERTICAL:
                dist = int(-dist_ratio * image.shape[0])
                return image[np.concatenate([np.arange(-dist, 0), np.arange(0, image.shape[0] - dist)], axis=0), :, ...]
        return image

    def translate_annotations(self, annotations, dist, axis):
        def correct_pos(pos):
            return (pos - 1) if pos > 1 else ((pos + 1) if pos < 0 else pos)

        def translate(anno):
            if axis is self.TRANSLATE_HORIZONTAL:
                return np.asarray([anno[0], correct_pos(anno[1] - dist), anno[2], anno[3], anno[4]])
            if axis is self.TRANSLATE_VERTICAL:
                return np.asarray([anno[0], anno[1], correct_pos(anno[2] - dist), anno[3], anno[4]])

        drop_crossings_filter = lambda a: (
                (a[2] - a[4] / 2) > 0 and (a[2] + a[4] / 2) < 1 and (a[1] - a[3] / 2) > 0 and (
                a[1] + a[3] / 2) < 1)

        return list(filter(drop_crossings_filter, [translate(anno) for anno in annotations]))

    def augment(self, image, annotations):
        augs = []
        for axis, max_aug in [(self.TRANSLATE_HORIZONTAL, self.max_horizontal_aug),
                              (self.TRANSLATE_VERTICAL, self.max_vertical_aug)]:
            [dist != 0 and augs.append((
                self.translate_image(image, dist, axis),
                self.translate_annotations(annotations, dist, axis)
            )) for dist in np.linspace(-self.max_trans_ratio, self.max_trans_ratio, max_aug + (max_aug % 2))[:(max_aug -(max_aug % 2))]]
        return augs
