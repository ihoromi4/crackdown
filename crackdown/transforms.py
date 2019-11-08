import numpy as np
from torchvision.transforms import *

__all__ = [
    'ToTensor',
    'ToNumpy',
    'Transpose',
    'Scale',
    'CenterCrop',
    'MaxPooling',
    'EMPTY',
    'downscale',
    'crop_center',
]


class ToNumpy:
    def __call__(self, img):
        arr = np.array(img)

        if len(arr.shape) == 2:
            arr = arr[:, :, np.newaxis]

        return arr


class Transpose:
    def __init__(self, order=(2, 0, 1)):
        self.order = order

    def __call__(self, arr):
        return np.transpose(arr, self.order)


class Scale:
    def __init__(self, factor: float = 1 / 255):
        self.factor = factor

    def __call__(self, arr):
        return arr * self.factor


class CenterCrop:
    def __init__(self, size: int):
        assert isinstance(size, int)

        self.size = size

    def __call__(self, arr):
        shape = np.array(arr.shape[:2])
        center = shape / 2
        half_size = self.size / 2
        left_up = (center - half_size).astype(np.int)
        right_down = (center + half_size).astype(np.int)
        return arr[tuple([slice(a, b) for a, b in zip(left_up, right_down)])]


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


EMPTY = ToTensor()


def downscale(shape: tuple, factor: float = 0.5):
    shape = np.array(shape)
    shape = (shape[:2] * factor).astype(np.int)

    return transforms.Compose([
        ToPILImage(),
        Resize(shape),
        ToNumpy(),
        Transpose((2, 0, 1)),
        Scale(),
    ])


def crop_center(size: int):
    return transforms.Compose([
        CenterCrop(size),
        Transpose((2, 0, 1)),
        Scale()
    ])
