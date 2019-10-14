import torch
import torch.nn as nn


class Agent(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def reset(self):
        pass
    
    def predict(self, state, deterministic: bool = False):
        pass
    
    def update(self, state, action, next_state, reward, is_done: bool = False):
        pass
        
    def save(self, filepath: str):
        assert isinstance(filepath, str)
        
        state = {
            'state_dict': self.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath: str):
        assert isinstance(filepath, str)
        
        state = torch.load(filepath)
        self.load_state_dict(state['state_dict'])
