from typing import Union
import torch
from torch import nn

__all__ = [
    'QualityNetwork',
]


def get_module_output_dim(model: nn.Module) -> int:
    linear_layers = [l for l in model.modules() if isinstance(l, nn.Linear)]

    if len(linear_layers) == 0:
        raise NotImplemented('get_module_output_dim support only nn.Linear')

    return linear_layers[-1].out_features


class QualityNetwork(nn.Module):
    def __init__(self,
                 embedding_dim: int = 16,
                 output_dim: int = 0,
                 discount_factor: float = 0.95,
                 dueling_dqn: bool = False):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.discount_factor = discount_factor
        self.dueling_dqn = dueling_dqn

        self.features = self.create_feature_extractor()
        features_dim = get_module_output_dim(self.features)
        self.q_net = nn.Linear(features_dim, output_dim, bias=True)

        if dueling_dqn:
            self.dueling_q_net = nn.Linear(features_dim, 1, bias=True)

        self.criterion = nn.MSELoss()

    def create_feature_extractor(self) -> nn.Module:
        features_dim = (self.output_dim + self.embedding_dim) // 2
        return nn.Sequential(
            nn.Linear(self.embedding_dim, features_dim, bias=True),
            nn.Softplus(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        features = self.features(embedding)
        action_quality = self.q_net(features)

        if self.dueling_dqn:
            value = self.dueling_q_net(features)
            action_quality = action_quality + value - torch.mean(action_quality, dim=-1, keepdim=True)

        return action_quality

    def update(self,
               embedding: torch.Tensor,
               action: torch.Tensor,
               reward: Union[float, torch.Tensor],
               next_embedding,
               next_quality=None):

        action = action.long()
        quality = self.forward(embedding)
        quality = torch.gather(quality, 1, action)

        with torch.no_grad():
            if next_quality is None:
                next_quality = self.forward(next_embedding)

            next_quality, _ = torch.max(next_quality, dim=1, keepdim=True)
            # next_quality = torch.gather(next_quality, 1, action)

        # temporal difference residual
        true_quality = reward + self.discount_factor * next_quality
        advantage = true_quality - quality

        loss = self.criterion(true_quality, quality)

        return loss, advantage

    def fit(self, embedding: torch.Tensor, true_quality: torch.Tensor):
        quality = self.forward(embedding)
        loss = self.criterion(quality, true_quality)

        return loss
