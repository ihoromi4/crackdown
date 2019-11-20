from typing import Union
import torch
import torch.nn as nn

__all__ = [
    'AdvantageCritic',
]


class AdvantageCritic(nn.Module):
    def __init__(self, embedding_dim: int = 16, action_dim: int = 0, discount_factor: float = 0.95):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        
        self.state_value_net = nn.Sequential(
            nn.Linear(embedding_dim, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True)
        )

        self.action_value_net = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True)
        )
            
        self.criterion = nn.MSELoss()
        
    def forward(self, embedding: torch.Tensor, action: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        embedding_action = torch.cat([embedding, action.float()], dim=-1)

        state_value = self.state_value_net(embedding)
        action_value = self.action_value_net(embedding_action)

        return state_value, action_value
    
    def update(self, embedding, action, reward: Union[float, torch.Tensor], next_embedding, next_action) -> (torch.Tensor, torch.Tensor):
        state_value, action_value = self.forward(embedding, action)

        with torch.no_grad():
            next_state_value, next_action_value = self.forward(next_embedding, next_action)

            true_state_value = reward + self.discount_factor * next_state_value
            true_action_value = reward + self.discount_factor * next_action_value

            advantage = action_value - state_value

        state_value_loss = self.criterion(state_value, true_state_value)
        action_value_loss = self.criterion(action_value, true_action_value)
        loss = state_value_loss + action_value_loss

        return loss, advantage
