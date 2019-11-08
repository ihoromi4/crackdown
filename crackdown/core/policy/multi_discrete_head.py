from gym.spaces import MultiDiscrete
import torch
import torch.nn as nn
from torch import distributions

__all__ = [
    'MultiDiscretePolicyHead',
]


class MultiDiscretePolicyHead(nn.Module):
    def __init__(self, input_dim: int, action_space: MultiDiscrete):
        super().__init__()

        assert isinstance(action_space, MultiDiscrete)

        self.input_dim = input_dim
        self.action_space = action_space

        self.linears = nn.ModuleList([nn.Linear(input_dim, out_dim) for out_dim in action_space.nvec])

    def forward(self, x):
        logits = [linear(x) for linear in self.linears]
        return logits

    def probabilities(self, x):
        logits = [linear(x) for linear in self.linears]
        return [torch.softmax(logit, dim=-1) for logit in logits]

    def distribution(self, x):
        logits = [linear(x) for linear in self.linears]
        distributions_ = [distributions.Categorical(logits=logit) for logit in logits]
        return distributions_

    def sample(self, x):
        distributions = self.distribution(x)
        samples = [distribution.sample() for distribution in distributions]
        return torch.stack(samples, dim=-1)

    def simple(self, x):
        probabilities = self.probabilities(x)
        return torch.stack([torch.argmax(p, dim=-1) for p in probabilities], dim=-1)

    def random(self, x=None):
        n = 1 if (x is None) else x.shape[0]
        actions = [self.action_space.sample() for _ in range(n)]
        actions = [torch.from_numpy(a) for a in actions]
        return torch.stack(actions)
