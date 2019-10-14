import torch.nn as nn

from ..attention.static_mask import StaticMaskAttention


class ViewEmbedding(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 16, output_size: int = 7):
        super().__init__()
        
        self.output_channels = output_channels
        self.output_size = output_size
        
        self.cnn = nn.Sequential(
            # conv block
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv block
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv block
            nn.Conv2d(32, output_channels, kernel_size=5, stride=1),
            nn.FractionalMaxPool2d((2, 2), (output_size, output_size)),
            nn.Tanh(),
            nn.ReLU()
        )
        
        self.mask = StaticMaskAttention()
        
    @property
    def shape(self):
        return self.output_channels * self.mask.n_masks
        
    def forward(self, state):
        signal = self.cnn(state)
        masked = self.mask(signal)
        
        return masked, signal
