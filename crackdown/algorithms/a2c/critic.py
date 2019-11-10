import torch
import torch.nn as nn


class TemporalDifferenceCritic(nn.Module):
    def __init__(self, embedding_dim: int = 16, action_dim: int = 0, discount_factor: float = 0.95):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True)
        )
            
        self.criterion = nn.MSELoss()
        
    def forward(self, embedding: torch.tensor, action: torch.tensor = None):
        if action is None:
            action = torch.zeros((embedding.shape[0], self.action_dim), dtype=torch.float32)
        else:
            action = action.float()

        x = torch.cat([embedding, action], dim=-1)
        quality = self.transform(x)
        
        return quality
    
    def update(self, embedding, action, reward: float, next_embedding, next_action, next_quality=None):
        quality = self.forward(embedding, action)

        if next_quality is None:
            with torch.no_grad():
                next_quality = self.forward(next_embedding, next_action.detach())
        
        # temporal difference residual
        true_quality = reward + self.discount_factor * next_quality
        advantage = true_quality - quality
        
        loss = self.criterion(true_quality, quality)
        
        return loss, true_quality, advantage
    
    def fit(self, embedding, action, true_quality):
        quality = self.forward(embedding, action)
        loss = self.criterion(quality, true_quality)
        
        return loss
