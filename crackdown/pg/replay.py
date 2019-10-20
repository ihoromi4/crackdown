import numpy as np
from collections import deque

__all__ = [
    'DiscountReplay',
]


class DiscountReplay:
    def __init__(self, maxlen: int = None, discount_factor: float = 0.95, discount_limit: float = 0.01):
        self.replay = deque(maxlen=maxlen)
        self.discount_factor = discount_factor
        self.steps_limit = int(np.log(discount_limit) / np.log(discount_factor))

    def reset(self):
        self.replay.clear()

    def put(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, done: bool):
        self.add_discounted_reward(reward)

        sample = [state, action, next_state, reward, done]
        self.replay.append(sample)

    def add_discounted_reward(self, reward: float):
        reward_index = 3
        done_index = -1
        discounted_reward = reward

        for i, sample in enumerate(reversed(self.replay)):
            if i >= self.steps_limit or sample[done_index]:
                return

            discounted_reward = self.discount_factor * discounted_reward
            sample[reward_index] += discounted_reward

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
