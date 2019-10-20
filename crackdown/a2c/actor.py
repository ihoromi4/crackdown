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
            
        self.criterion = nn.BCELoss(reduction='none')
        
    def forward(self, embedding: torch.tensor):
        return self.transform(embedding)

    def sample(self, embedding: torch.tensor):
        with torch.no_grad():
            probabilities = self.forward(embedding)
        
        dist = Bernoulli(probabilities)
        action = dist.sample().float()
        
        return action
    
    def simple(self, embedding: torch.tensor):
        with torch.no_grad():
            probabilities = self.forward(embedding)
            
        action = torch.round(probabilities)
#         action = (probabilities > 0.5).float()
        
        return action

    @staticmethod
    def distribution(probabilities: torch.tensor):
        return Bernoulli(probabilities)
    
    def predict(self, embedding: torch.tensor, deterministic: bool = False):
        if deterministic:
            return self.simple(embedding)
        else:
            return self.sample(embedding)
    
    def update(self, embedding: torch.tensor, quality: torch.tensor, action: torch.tensor):
        assert len(quality.shape) == 2

        probabilities = self.forward(embedding)

        # 1
        # loss = self.criterion(probs, action)
        # loss = torch.mean(loss * quality)

        # 2
        distribution = self.distribution(probabilities)
        loss = torch.mean(-distribution.log_prob(action) * quality)
        
        return loss, probabilities
