import warnings
from typing import Union
import torch
from torch import nn

__all__ = [
    'QualityHead',
]


def get_module_output_dim(model: nn.Module) -> int:
    linear_layers = [l for l in model.modules() if isinstance(l, nn.Linear)]

    if len(linear_layers) == 0:
        raise NotImplemented('get_module_output_dim support only nn.Linear')

    return linear_layers[-1].out_features


class QualityHead(nn.Module):
    def __init__(self,
                 input_dim: int = 16,
                 output_dim: int = 0,
                 discount_factor: float = 0.95,
                 dueling_dqn: bool = False):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.discount_factor = discount_factor
        self.dueling_dqn = dueling_dqn

        self.action_quality_net = nn.Linear(input_dim, output_dim, bias=True)

        if dueling_dqn:
            self.value_net = nn.Linear(input_dim, 1, bias=True)

        self.criterion = nn.MSELoss()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        action_quality = self.action_quality_net(embedding)

        if self.dueling_dqn:
            value = self.value_net(embedding)
            action_quality_mean = torch.mean(action_quality, dim=-1, keepdim=True)
            action_quality = action_quality - action_quality_mean
            action_quality = action_quality + value

        return action_quality

    def update(self,
               embedding: torch.Tensor,
               action: torch.Tensor,
               reward: Union[float, torch.Tensor],
               next_embedding,
               next_quality=None,
               done: bool = False):

        action = action.long()
        quality = self.forward(embedding)
        quality = torch.gather(quality, 1, action)

        with torch.no_grad():
            if next_quality is None:
                next_quality = self.forward(next_embedding)

            next_quality, _ = torch.max(next_quality, dim=1, keepdim=True)
            # next_quality = torch.gather(next_quality, 1, action)

        # temporal difference residual
        true_quality = reward + (1 - done) * self.discount_factor * next_quality
        advantage = true_quality - quality

        loss = self.criterion(true_quality, quality)

        return loss, advantage

    def fit(self, embedding: torch.Tensor, true_quality: torch.Tensor):
        warnings.warn("Method QualityHead.fit is deprecated", DeprecationWarning)

        quality = self.forward(embedding)
        loss = self.criterion(quality, true_quality)

        return loss
