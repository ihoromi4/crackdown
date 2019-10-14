import numpy as np
from collections import deque


class GameReplay:
    def __init__(self, maxlen: int = None):
        self.replay = deque(maxlen=maxlen)

    def reset(self):
        self.replay.clear()

    def put(self, *sample):
        self.replay.append(sample)
    
    def __len__(self):
        return len(self.replay)
    
    def __getitem__(self, key):
        sample = self.replay[key]
        return sample
    
    def batch(self, size):
        size = min(size, len(self))
        keys = np.random.choice(len(self), size, replace=True)
        samples = [self.replay[key] for key in keys]
        batch = tuple(map(np.array, zip(*samples)))
        
        return batch
