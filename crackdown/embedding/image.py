import torch.nn as nn

from crackdown.attention.static_mask import StaticMaskAttention


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
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv block
            nn.Conv2d(32, output_channels, kernel_size=5, stride=1),
            # nn.FractionalMaxPool2d((2, 2), (feature_map_size, feature_map_size)),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.ReLU()
        )
        
        self.mask = StaticMaskAttention()
        
    @property
    def shape(self):
        return [self.output_channels * self.mask.n_masks]
        
    def forward(self, state):
        feature_map = self.cnn(state)
        masked = self.mask(feature_map)
        
        return masked, feature_map
