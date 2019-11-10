from gym import spaces
import torch.nn as nn

from .multi_binary_head import MultiBinaryPolicyHead
from .discrete_head import DiscretePolicyHead
from .multi_discrete_head import MultiDiscretePolicyHead

__all__ = [
    'MultiBinaryPolicyHead',
    'DiscretePolicyHead',
    'MultiDiscretePolicyHead',
    'make_action_head',
]


def make_action_head(action_space) -> nn.Module:
    if isinstance(action_space, spaces.MultiBinary):
        return MultiBinaryPolicyHead(action_space)

    if isinstance(action_space, spaces.Discrete):
        return DiscretePolicyHead(action_space)

    if isinstance(action_space, spaces.MultiDiscrete):
        return MultiDiscretePolicyHead(1, action_space)

    raise TypeError('Unsupported action_space of type %s' % type(action_space))
