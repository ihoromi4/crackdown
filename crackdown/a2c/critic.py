import torch
import torch.nn as nn


class TemporalDifferenceCritic(nn.Module):
    def __init__(self, embedding_dim: int = 16, action_dim: int = 8, discount_factor: float = 0.95):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 1, bias=False)
        )
            
        self.criterion = nn.MSELoss()
        
    def forward(self, embedding, action):
        x = torch.cat([embedding, action], dim=-1)
        quality = self.transform(x)
        
        return quality
    
    def update(self, embedding, action, reward: float, next_embedding, next_action):
        quality = self.forward(embedding, action)
        next_quality = self.forward(next_embedding, next_action.detach())
        
        # temporal difference residual
        true_quality = reward + self.discount_factor * next_quality.detach()
        advantage = true_quality - quality
        
        loss = self.criterion(true_quality, quality)
        
        return loss, true_quality, advantage
    
    def fit(self, embedding, action, true_quality):
        quality = self.forward(embedding, action)
        loss = self.criterion(quality, true_quality)
        
        return loss
