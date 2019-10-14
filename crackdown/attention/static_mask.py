import torch
import torch.nn as nn


def get_basis(width, height, device='cpu'):
    w = torch.linspace(-1, 1, width)
    x = torch.stack([w] * height, dim=0).to(device)
    
    h = torch.linspace(1, -1, height)
    y = torch.stack([h] * width, dim=1).to(device)
    
    return x, y


class StaticMaskAttention(nn.Module):
    def __init__(self, reduce: bool = True):
        super().__init__()
        
        self.reduce = reduce
        self.basis = None
        self.mask_functions = [
            lambda x, y: 1.0,  # full
            lambda x, y: (x < 0).float(),  # left
            lambda x, y: (x > 0).float(),  # right
            lambda x, y: (y < 0).float(),  # top
            lambda x, y: (y > 0).float(),  # down
            lambda x, y: ((x > -0.2) * (x < 0.2)).float(),  # vertical channel
            lambda x, y: ((y > -0.2) * (y < 0.2)).float(),  # horizontal channel
            lambda x, y: ((y > -0.2) * (y < 0.2) * (x < 0)).float(),  # horizontal channel, left side
            lambda x, y: ((y > -0.2) * (y < 0.2) * (x > 0)).float(),  # horizontal channel, right side
            lambda x, y: ((y > -0.2) * (y < 0.2) * (x < 0) * (x > -0.3)).float(),  # left view
            lambda x, y: ((y > -0.2) * (y < 0.2) * (x > 0) * (x < 0.3)).float(),  # right view
        ]
       
    @property
    def n_masks(self):
        return len(self.mask_functions)
        
    def get_basis(self, width, height, device='cpu'):
        if self.basis is None:
            self.basis = get_basis(width, height, device)
        elif self.basis[0].shape != (width, height):
            self.basis = get_basis(width, height, device)
        
        return self.basis
    
    def get_masks(self, width, height, device='cpu'):
        x, y = self.get_basis(width, height, device)
        masks = [f(x, y) for f in self.mask_functions]
        
        return masks
            
    def forward(self, x):
        masks = self.get_masks(*x.shape[-2:], x.device)
        masked = [mask * x for mask in masks]
        masked = torch.cat(masked, dim=1)
        
        if self.reduce:
             masked = torch.max(torch.max(masked, dim=-1)[0], dim=-1)[0]
                
        return masked
