import numpy as np
from torchvision.transforms import *

__all__ = [
    'ToNumpy',
    'Transpose',
]


class ToNumpy:
    def __call__(self, img):
        arr = np.array(img)

        if len(arr.shape) == 2:
            arr = arr[:, :, np.newaxis]

        return arr


class Transpose:
    def __init__(self, order):
        self.order = order

    def __call__(self, arr):
        return np.transpose(arr, self.order)


class Scale:
    def __init__(self, factor: float = 1 / 255):
        self.factor = factor

    def __call__(self, arr):
        return arr * self.factor


EMPTY = transforms.Compose([
    Transpose((2, 0, 1)),
    Scale(),
])


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
