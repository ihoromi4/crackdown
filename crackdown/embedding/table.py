import torch
import torch.nn as nn

__all__ = [
    'TableEmbedding',
]


class TableEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 64,
                 output_dim: int = 16):

        super().__init__()

        assert isinstance(input_dim, int)
        assert isinstance(hidden_dim, int)
        assert isinstance(output_dim, int)

        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    @property
    def shape(self):
        return torch.Size([self.output_dim])
        
    def forward(self, state) -> torch.Tensor:
        return self.net(state)
