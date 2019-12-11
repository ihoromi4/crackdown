from typing import Union
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'TensorBuffer',
]

BUFFER_TEMPLATE = (
    ('state', (1, 128, 128), torch.float32),
    ('action', (4,), torch.float32),
    ('next_state', (1, 128, 128), torch.float32),
    ('reward', (1,), torch.float32),
    ('done', (1,), torch.float32),
)
BATCH_TEMPLATE = (
    ('state', 0),
    ('action', 0),
    ('next_state', 0),
    ('reward', 0),
    ('done', 0),
)


class TensorBuffer(nn.Module):
    def __init__(self,
                 size: int,
                 buffer_template: tuple = BUFFER_TEMPLATE,
                 batch_template: tuple = BATCH_TEMPLATE):

        assert isinstance(size, int), "expected size type is int, got %s" % type(size)
        assert isinstance(buffer_template, tuple), "expected buffer_template type is tuple, got %s" % type(buffer_template)
        assert isinstance(batch_template, tuple), "expected batch_template type is tuple, got %s" % type(batch_template)

        super().__init__()

        self.size = size
        self.buffer_template = buffer_template
        self.batch_template = batch_template

        self.index = 0
        self.buffer = nn.ParameterDict({})

        self.initialize_buffer()

    def initialize_buffer(self):
        self.buffer = nn.ParameterDict({
            name: nn.Parameter(torch.zeros((self.size,) + shape, dtype=dtype), False) for
            name, shape, dtype in self.buffer_template
        })

    def add_buffer(self, name: str, shape: Union[tuple, torch.Size], dtype: torch.dtype = torch.float32):
        assert isinstance(name, str), "expected name type is str, got %s" % type(name)
        assert isinstance(shape, (tuple, torch.Size)), "expected shape type is tuple or torch.Size, got %s" % type(shape)
        assert isinstance(dtype, torch.dtype), 'expected dtype type is torch.dtype, got %s' % type(dtype)

        tensor = torch.zeros((self.size,) + shape, dtype=dtype)
        self.buffer[name] = nn.Parameter(tensor)
        self.buffer_template += (name, shape, dtype)

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
        if key in super().__getattr__('buffer'):
            return super().__getattr__('buffer')[key][:self.index]

        return super().__getattr__(key)

    def put(self, *args, **kwargs):
        for sample, name in zip(args, np.take(self.buffer_template, 0, -1)):
            self.buffer[name][self.index] = sample

        for key, value in kwargs.items():
            self.buffer[key][self.index] = value

        self.index += 1

        if self.index >= self.size:
            self.rolling()

    def batch(self, size: int, template: tuple = None):
        template = template or self.batch_template

        min_ = np.take(template, 1, -1).astype(int).min()
        from_ = range(-min_, len(self))
        keys = np.random.choice(from_, size, replace=True)

        return [self.buffer[name][keys + index] for name, index in template]
