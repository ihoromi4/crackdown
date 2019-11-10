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
        self.input_dim = action_space.n

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([-1, self.action_space.n])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([-1, 1])

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[-1] == self.input_dim

        return torch.softmax(logits, dim=-1)

    def distribution(self, logits: torch.Tensor) -> distributions.Distribution:
        assert logits.shape[-1] == self.input_dim

        return distributions.Categorical(logits=logits)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(logits)
        return dist.entropy()

    def sample(self, logits: torch.Tensor = None, deterministic: bool = False) -> torch.Tensor:
        if logits is None:
            return self.random()
        elif deterministic:
            assert logits.shape[-1] == self.input_dim

            probabilities = torch.softmax(logits, dim=-1)
            return torch.argmax(probabilities, dim=-1, keepdim=True)
        else:
            dist = self.distribution(logits)
            return dist.sample().view((-1, 1))

    def random(self, n_samples: int = 1) -> torch.Tensor:
        actions = [self.action_space.sample() for _ in range(n_samples)]
        actions = [torch.tensor(a) for a in actions]
        return torch.tensor(actions).view((n_samples, 1))
