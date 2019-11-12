from gym.spaces import MultiBinary
import torch
import torch.nn as nn
from torch import distributions

from ...utils import rsample_bernoulli

__all__ = [
    'MultiBinaryPolicyHead',
]


class MultiBinaryPolicyHead(nn.Module):
    def __init__(self, action_space: MultiBinary):
        super().__init__()

        assert isinstance(action_space, MultiBinary)

        self.action_space = action_space

    @property
    def input_shape(self):
        return torch.Size([-1, self.action_space.n])

    @property
    def output_shape(self):
        return torch.Size([-1, self.action_space.n])

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    @staticmethod
    def distribution(logits: torch.Tensor) -> distributions.Distribution:
        return distributions.Bernoulli(logits=logits)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(logits)
        return dist.entropy()

    def sample(self, logits: torch.Tensor = None, deterministic: bool = False) -> torch.Tensor:
        if logits is None:
            return self.random()
        if deterministic:
            probabilities = self.probabilities(logits)
            return torch.round(probabilities)
        else:
            dist = self.distribution(logits=logits)
            return dist.sample()

    def rsample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return rsample_bernoulli(probs)

    def random(self, n_samples: int = 1) -> torch.Tensor:
        actions = [self.action_space.sample() for _ in range(n_samples)]
        return torch.tensor(actions)
