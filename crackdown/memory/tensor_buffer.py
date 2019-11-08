from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'TensorBuffer',
]

BUFFER_TEMPLATE = (
    ('state', (1, 128, 128), torch.float32),
    ('action', (4,), torch.float32),
    ('reward', (1,), torch.float32),
    ('done', (1,), torch.float32),
)
BATCH_TEMPLATE = (
    ('state', -1),
    ('action', 0),
    ('state', 0),
    ('reward', 0),
)


class TensorBuffer(nn.Module):
    def __init__(self,
                 size: int,
                 buffer_template: tuple = BUFFER_TEMPLATE,
                 batch_template: tuple = BATCH_TEMPLATE):

        super().__init__()

        self.size = size
        self.index = 0

        self.buffer_template = buffer_template
        self.batch_template = batch_template
        self.buffer = None

        self.initialize_buffer()

    def initialize_buffer(self):
        self.buffer = nn.ParameterDict({
            name: nn.Parameter(torch.zeros((self.size,) + shape, dtype=dtype), False) for
            name, shape, dtype in self.buffer_template
        })

    def reset(self):
        self.index = 0

    def rolling(self, percent: float = 0.2):
        n_samples = int(len(self) * percent)
        assert n_samples > 0

        for tensor in self.buffer.values():
            tensor[:n_samples] = tensor[:self.index][-n_samples:]

        self.index = n_samples

    def __len__(self):
        return self.index

    def __getattr__(self, key):
        if ('buffer' in self.__dict__) and (key in self.buffer):
            return self.buffer[key][:self.index]

        return super().__getattr__(key)

    def put(self, *args, **kwargs):
        for sample, name in zip(args, np.take(self.buffer_template, 0, -1)):
            self.buffer[name][self.index] = sample

        for key, value in kwargs.items():
            self.buffer[key][self.index] = value

        self.index += 1

        if self.index >= self.size:
            self.rolling()

    def batch(self, size: int, template: list = None):
        template = template or self.batch_template

        min_ = np.take(template, 1, -1).astype(int).min()
        from_ = range(-min_, len(self))
        keys = np.random.choice(from_, size, replace=True)

        return [self.buffer[name][keys + index] for name, index in template]
