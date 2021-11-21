class FlipAugmentor:
    FLIP_HORIZONTAL = 'horizontal'
    FLIP_VERTICAL = 'vertical'
    FLIP_BOTH = 'both'

    def __init__(self, axis=FLIP_BOTH, max_aug=2):
        self.axis = axis
        self.max_aug = max_aug

    def augment(self, image, annotations):
        augs = []
        if self.axis is self.FLIP_HORIZONTAL or self.axis is self.FLIP_BOTH:
            len(augs) < self.max_aug and augs.append((
                image[:, ::-1, ...],
                [[anno[0], 1.0 - anno[1], anno[2], anno[3], anno[4]] for anno in annotations]
            ))
        if self.axis is self.FLIP_VERTICAL or self.axis is self.FLIP_BOTH:
            len(augs) < self.max_aug and augs.append((
                image[::-1, :, ...],
                [[anno[0], anno[1], 1.0 - anno[2], anno[3], anno[4]] for anno in annotations]
            ))
        return augs
