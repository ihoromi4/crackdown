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

        hidden_dim = (output_dim + embedding_dim) // 2
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )

        self.criterion = nn.MSELoss()

    def forward(self, embedding: torch.tensor):
        quality = self.transform(embedding)

        return quality

    def update(self, embedding, action, reward: Union[float, torch.Tensor], next_embedding, next_quality=None):
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

    def fit(self, embedding, true_quality):
        quality = self.forward(embedding)
        loss = self.criterion(quality, true_quality)

        return loss
