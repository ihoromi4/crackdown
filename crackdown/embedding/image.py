import torch
import torch.nn as nn


class ImageEmbedding(nn.Module):
    def __init__(self,
                 input_channels: int = 3,
                 output_channels: int = 16,
                 feature_map_size: int = 7):

        super().__init__()
        
        self.output_channels = output_channels
        self.feature_map_size = feature_map_size
        
        self.cnn = nn.Sequential(
            # conv block
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv block
            nn.Conv2d(16, output_channels, kernel_size=5, stride=1),
            nn.ReLU(),
        )

        self.fraction = nn.FractionalMaxPool2d((2, 2), (feature_map_size, feature_map_size))
        
    @property
    def output_shape(self):
        return torch.Size([-1, self.output_channels * self.feature_map_size ** 2])
        
    def forward(self, state):
        x = self.cnn(state)
        feature_map = self.fraction(x)
        masked = feature_map.view((feature_map.shape[0], -1))

        return masked, feature_map
