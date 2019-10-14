import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class Actor(nn.Module):
    def __init__(self, embedding_dim: int = 16, action_dim: int = 8):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, action_dim, bias=True),
            nn.Sigmoid()
        )
            
        learning_rate = 1e-4
        self.criterion = nn.BCELoss(reduction='none')
        
    def forward(self, embedding):
        actions_probs = self.transform(embedding)

        return actions_probs
    
    def sample(self, embedding):
        with torch.no_grad():
            action_probs = self(embedding)
        
        dist = Bernoulli(action_probs)
        action = dist.sample().float()
        
        return action
    
    def simple(self, embedding):
        with torch.no_grad():
            probs = self(embedding)
            
        action = torch.round(probs)
#         action = (action_probs > 0.5).float()
        
        return action
    
    def predict(self, embedding, deterministic: bool = False):
        if deterministic:
            return self.simple(embedding)
        else:
            return self.sample(embedding)
    
    def update(self, embedding, quality, action):
        assert len(quality.shape) == 2
        
        probs = self.forward(embedding)
        
        loss = self.criterion(probs, action)
        loss = torch.mean(loss * quality)
        
        return loss, probs
