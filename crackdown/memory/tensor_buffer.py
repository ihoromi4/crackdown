import warnings
from typing import Union
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'TensorBuffer',
]

SAMPLE_BUFFER_TEMPLATE = (
    ('state', (1, 128, 128), torch.float32),
    ('action', (4,), torch.float32),
    ('next_state', (1, 128, 128), torch.float32),
    ('reward', (1,), torch.float32),
    ('done', (1,), torch.float32),
)
SAMPLE_BATCH_TEMPLATE = (
    ('state', 0),
    ('action', 0),
    ('next_state', 0),
    ('reward', 0),
    ('done', 0),
)


class TensorBuffer(nn.Module):
    """
    Reinforcement Learning Trajectory Container
    Collect any tensor data in named arrays. Sample batches of any saved data with any index shift.
    Save numerical data, boolean, numpy.ndarray, torch.Tensor with casting to torch.Tensor.

    Args:
        size (int): len of fixed-size internal tensor buffer
        buffer_template (tuple, optional): description (name, shape, dtype) of buffered data
        batch_template (tuple, optional): buffer values to sample from container by __getitem__() or sample()
        device (str|torch.device, optional): torch device tensors allocate to
        expand_buffer (bool, optional): add buffers dynamically in runtime by kwargs of put()
        extend_buffer (bool, optional): increase length of buffer by 2x on hit of length limit
    """

    def __init__(self,
                 size: int,
                 batch_template: tuple = (),
                 buffer_template: tuple = (),
                 device: Union[str, torch.device] = None,
                 expand_buffer: bool = True,
                 extend_buffer: bool = False):

        assert isinstance(size, int), "expected size type is int, got %s" % type(size)
        assert size > 0, "expected size > 0, got %s" % size
        assert isinstance(buffer_template, tuple), \
            "expected buffer_template type is tuple, got %s" % type(buffer_template)
        assert isinstance(batch_template, tuple), "expected batch_template type is tuple, got %s" % type(batch_template)
        assert isinstance(expand_buffer, bool), "expected expand_buffer type is bool, got %s" % type(expand_buffer)

        super().__init__()

        self.index = 0
        self.size = size
        self.batch_template = batch_template
        self.buffer_template = ()
        self.device = device
        self.expand_buffer = expand_buffer
        self.extend_buffer = extend_buffer
        self.buffer = nn.ParameterDict({})

        self._initialize_buffer(buffer_template)

    def _initialize_buffer(self, buffer_template: tuple) -> None:
        for args in buffer_template:
            self.add_buffer(*args)

    def add_buffer(self,
                   name: str,
                   shape: Union[tuple, torch.Size],
                   dtype: torch.dtype = torch.float32,
                   device: Union[str, torch.device] = None) -> None:

        """Add new buffer tensor array with specific name, shape and dtype"""

        assert isinstance(name, str), "expected name type is str, got %s" % type(name)
        assert isinstance(shape, (tuple, torch.Size)), \
            "expected shape type is tuple or torch.Size, got %s" % type(shape)
        assert isinstance(dtype, torch.dtype), 'expected dtype type is torch.dtype, got %s' % type(dtype)

        device = device or self.device
        tensor = torch.zeros((self.size,) + shape, dtype=dtype, device=device)
        self.buffer[name] = nn.Parameter(tensor, False)
        self.buffer_template += ((name, shape, dtype),)

    def reset(self) -> None:
        """Restart buffer filling with overriding"""

        self.index = 0

    def extend(self, factor: float = 2.0) -> None:
        """Extend length of the buffer by specified factor"""

        additional_size = int(self.size * (factor - 1.0))

        for name, values in self.buffer.items():
            _, *sample_shape = values.shape
            extension = torch.empty([additional_size] + sample_shape, dtype=values.dtype, device=values.device)
            extended = torch.cat([values, extension], dim=0)
            self.buffer[name] = nn.Parameter(extended, False)

        self.size += additional_size

    def rolling(self, percent: float = 0.2) -> None:
        """Restart buffer filling with overriding and store some percent of data"""

        n_samples = int(len(self) * percent)
        assert n_samples > 0

        for tensor in self.buffer.values():
            tensor[:n_samples] = tensor[:self.index][-n_samples:]

        self.index = n_samples

    def on_hit_length_limit(self) -> None:
        if self.extend_buffer:
            self.extend()
        else:
            self.rolling()

    def __repr__(self):
        repr_ = super().__repr__()

        bytes_size = sum([t.element_size() * t.nelement() for t in self.buffer.values()])

        return repr_ + "\n" + \
               'Length: %s/%s\n' % (len(self), self.size) + \
               "Total size: %s kb" % (bytes_size >> 10)

    def __len__(self) -> int:
        return self.index

    def __getattr__(self, key):
        if key in super().__getattr__('buffer'):
            return super().__getattr__('buffer')[key][:self.index]

        return super().__getattr__(key)

    def put(self, *args, **kwargs) -> None:
        """Put data sample to buffer"""

        def to_torch_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            elif isinstance(value, (int, float, bool)):
                return torch.tensor([value])
            else:
                return torch.tensor(value)

        assert len(args) <= len(self.buffer)

        # resolve values' names
        names = [name for name, *_ in self.buffer_template]
        named_args: dict = {name: sample for name, sample in zip(names, args)}
        kwargs.update(named_args)

        for key, value in kwargs.items():
            value = to_torch_tensor(value)

            if key not in self.buffer:
                if not self.expand_buffer:
                    raise IndexError('buffer does not contain %s key' % key)

                self.add_buffer(key, value.shape, value.dtype)

            self.buffer[key][self.index] = value

        self.index += 1

        if self.index >= self.size:
            self.on_hit_length_limit()

    def batch(self, size: int, template: tuple = None) -> list:
        warnings.warn(
            "batch is deprecated now and will be deleted",
            DeprecationWarning
        )

        return self.sample(size, template)

    def __getitem__(self, key: int) -> OrderedDict:
        """Get one buffer sample by batch_template"""

        assert isinstance(key, int)
        assert key < len(self)

        # check bounds
        template_shifts = [index for _, index, *_ in self.batch_template]
        min_shift = min(template_shifts)
        max_shift = max(template_shifts)
        assert key >= min_shift
        assert key < len(self) - max_shift

        return OrderedDict(((name, self.buffer[name][key + index]) for name, index in self.batch_template))

    def sample(self, size: int, template: tuple = None, device=None) -> OrderedDict:
        """Sample data batch from buffer"""

        template = self.batch_template if (template is None) else template

        assert isinstance(template, tuple), "expected template type is tuple, got %s" % type(template)
        assert len(template) > 0, "expected non empty template"

        template_shifts = [index for _, index, *_ in template]
        min_shift = min(template_shifts)
        max_shift = max(template_shifts)
        from_range = range(-min_shift, len(self) - max_shift)
        indexes = np.random.choice(from_range, size, replace=True)

        return OrderedDict(((name, self.buffer[name][indexes + shift].to(device)) for name, shift in template))
