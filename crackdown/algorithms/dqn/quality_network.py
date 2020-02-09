from typing import Union
import torch
from torch import nn

__all__ = [
    'QualityNetwork',
]


class QualityNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 16, output_dim: int = 0, discount_factor: float = 0.95):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.discount_factor = discount_factor

        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, 64, bias=True),
            nn.Softplus(),
            nn.Linear(64, output_dim, bias=True)
        )

        self.criterion = nn.MSELoss()

    def forward(self, embedding: torch.tensor):
        quality = self.transform(embedding)

        return quality

    def update(self, embedding, action, reward: Union[float, torch.Tensor], next_embedding, next_quality=None):
        action = action.long()
        quality = self.forward(embedding)
        quality = torch.gather(quality, 1, action)

        if next_quality is None:
            with torch.no_grad():
                next_quality = self.forward(next_embedding)
                next_quality = torch.max(next_quality, dim=1, keepdim=True)[0]

        # temporal difference residual
        true_quality = reward + self.discount_factor * next_quality
        advantage = true_quality - quality

        loss = self.criterion(true_quality, quality)

        return loss, advantage

    def fit(self, embedding, action, true_quality):
        quality = self.forward(embedding, action)
        loss = self.criterion(quality, true_quality)

        return loss

