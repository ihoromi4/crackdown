from gym.spaces import Discrete
import torch
import torch.nn as nn
from torch import distributions

__all__ = [
    'DiscretePolicyHead',
]


class DiscretePolicyHead(nn.Module):
    def __init__(self, action_space: Discrete):
        super().__init__()

        assert isinstance(action_space, Discrete)
        self.action_space = action_space

    @property
    def input_shape(self):
        return torch.Size([-1, self.action_space.n])

    def forward(self, logits):
        return torch.sigmoid(logits)

    def distribution(self, logits):
        return distributions.Categorical(logits=logits)

    def sample(self, logits):
        dist = self.distribution(logits)
        return dist.sample().view((-1, 1))

    def simple(self, logits):
        probabilities = self.probabilities(logits)
        return torch.argmax(probabilities, dim=-1)

    def random(self, logits=None):
        n = 1 if (logits is None) else logits.shape[0]
        actions = [self.action_space.sample() for _ in range(n)]
        actions = [torch.tensor(a) for a in actions]
        return torch.stack(actions)
