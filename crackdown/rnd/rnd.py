import torch
import torch.nn as nn

__all__ = [
    'RandomNetworkDistillation',
]


class RandomNetworkDistillation(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 32, output_dim: int = 16):
        super().__init__()

        assert isinstance(embedding_dim, int)
        assert isinstance(hidden_dim, int)
        assert isinstance(output_dim, int)

        self.target = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, embedding: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            target = self.target(embedding)

        return target, self.predictor(embedding)

    def update(self, embedding: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        target, predicted = self(embedding)
        loss = self.criterion(target, predicted)
        reward = torch.mean(loss, dim=-1, keepdim=True)
        loss = torch.mean(loss)

        return reward, loss
