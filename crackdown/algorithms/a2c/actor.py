import gym
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from ...core.policy import make_action_head


class Actor(nn.Module):
    def __init__(self, embedding_dim: int, action_space: gym.spaces.Space):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.action_space = action_space

        self.head = make_action_head(action_space)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, self.head.input_shape[-1], bias=True),
        )

        self.criterion = nn.BCELoss(reduction='none')

    @property
    def output_shape(self):
        return self.head.output_shape

    def forward(self, embedding: torch.tensor):
        x = self.net(embedding)
        return self.head(x)

    def sample(self, embedding: torch.tensor, deterministic: bool = False):
        with torch.no_grad():
            x = self.net(embedding)
            action = self.head.sample(x, deterministic)
        
        return action
    
    def distribution(self, embedding: torch.tensor):
        x = self.net(embedding)
        return self.head.distribution(x)
    
    def predict(self, embedding: torch.tensor, deterministic: bool = False):
        return self.sample(embedding, deterministic)
    
    def update(self, embedding: torch.tensor, quality: torch.tensor, action: torch.tensor):
        assert len(quality.shape) == 2

        x = self.net(embedding)
        probabilities = self.head(x)
        distribution = self.head.distribution(x)

        loss = torch.mean(-distribution.log_prob(action) * quality)
        
        return loss, probabilities
