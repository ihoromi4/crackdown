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

        self.epsilon = 1e-8
        self.rolling_reward_mean = 0
        self.rolling_reward_std = 0

        self.target = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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

        factor = 0.01
        reward_mean = reward.mean().detach()
        self.rolling_reward_mean = factor * reward_mean + (1 - factor) * self.rolling_reward_mean
        self.rolling_reward_std = factor * torch.abs(reward_mean - self.rolling_reward_mean) + (1 - factor) * self.rolling_reward_std
        scaled_reward = (reward - self.rolling_reward_mean) / (3 * self.rolling_reward_std + self.epsilon)
        clipped_reward = torch.clamp(scaled_reward, -1, 1)

        return clipped_reward, loss
