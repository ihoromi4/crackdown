import numpy as np
import gym
from gym import spaces
from torchvision import datasets

__all__ = [
    'MNISTClassificationBenchmark',
]


class MNISTClassificationBenchmark(gym.Env):
    observation_space = spaces.Box(low=0, high=255, shape=(28, 28, 1), dtype=np.uint8)
    action_space = spaces.Discrete(9)

    def __init__(self):
        self.dataset = datasets.MNIST('./data', download=True)
        self.index = 0
        self.image = None
        self.true_class = 0

        self.next()

    def next(self):
        self.index = np.random.randint(0, len(self.dataset))
        image, self.true_class = self.dataset[self.index]
        self.image = np.array(image)[:, :, np.newaxis]

    def get_observation(self):
        return self.image

    def render(self, mode='human'):
        return

    def reset(self):
        self.next()

        return self.image

    def step(self, action: int):
        action = int(action)
        reward = 2 * int(action == self.true_class) - 1

        self.next()

        obs = self.get_observation()
        done = False
        info = {}

        return obs, reward, done, info
