from gym.spaces import MultiBinary
import torch
import torch.nn as nn
from torch import distributions

__all__ = [
    'MultiBinaryPolicyHead',
]


class MultiBinaryPolicyHead(nn.Module):
    def __init__(self, input_dim: int, action_space: MultiBinary):
        super().__init__()

        assert isinstance(action_space, MultiBinary)

        self.input_dim = input_dim
        self.action_space = action_space

        self.linear = nn.Linear(input_dim, action_space.n)

    def forward(self, x):
        logits = self.linear(x)
        return logits

    def probabilities(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)

    def distribution(self, x):
        logits = self.linear(x)
        dist = distributions.Bernoulli(logits=logits)
        return dist

    def sample(self, x):
        dist = self.distribution(x)
        return dist.sample()

    def simple(self, x):
        probabilities = self.probabilities(x)
        return torch.round(probabilities)

    def random(self, x=None):
        n = 1 if (x is None) else x.shape[0]
        actions = [self.action_space.sample() for _ in range(n)]
        actions = [torch.from_numpy(a) for a in actions]
        return torch.stack(actions)
