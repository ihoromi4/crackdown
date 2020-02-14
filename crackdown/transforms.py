import numpy as np
import albumentations
from albumentations import Crop, Resize
from albumentations.pytorch.transforms import ToTensor

__all__ = [
    'Compose',
    'ToTensor',
    'ToNumpy',
    'Crop',
    'Resize',
    'MaxPooling',
    'EMPTY',
]


class Compose(albumentations.Compose):
    def __call__(self, image):
        return super().__call__(image=image)['image']


class ToNumpy:
    def __call__(self, img):
        arr = np.array(img)

        if len(arr.shape) == 2:
            arr = arr[:, :, np.newaxis]

        return arr


class MaxPooling:
    def __init__(self, factor: int):
        assert isinstance(factor, int)

        self.factor = factor

    def __call__(self, arr):
        max_pool = self.factor
        width, height, channels = arr.shape
        arr = arr[:width // max_pool * max_pool, :height // max_pool * max_pool]
        width, height, channels = arr.shape
        return arr.reshape(width // max_pool, max_pool, height // max_pool, max_pool, -1).max(axis=(1, 3))


EMPTY = Compose([
    ToTensor(),
])
