class Augmenter:

    def __init__(self, augmenters=None):
        self.augmenters = [augmenters, []][augmenters is None]

    def add_augmentor(self, augmentor):
        self.augmenters.append(augmentor)

    def augment(self, image, annotations):
        assert len(self.augmenters) != 0
        augs = []
        for augmenter in self.augmenters:
            for i, a in augmenter.augment(image, annotations):
                augs.append((i, a, type(augmenter).__name__[:-9]+str(len(augs)+1)))
        return augs
