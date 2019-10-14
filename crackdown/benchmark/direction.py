import numpy as np
import gym
from gym import spaces

__all__ = [
    'DirectionEnvironment',
]


class DirectionEnvironment(gym.Env):
    metadata = {'render.modes': ['human'], 'video.frames_per_second': 30}
    observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
    action_space = spaces.MultiBinary(4)
    reward_range = (-1, 1)

    def __init__(self, size: tuple = (64, 64)):
        assert isinstance(size, (tuple, list))
        assert len(size) == 2

        self.observation_space = spaces.Box(low=0, high=255, shape=tuple(size) + (1,), dtype=np.uint8)

        self.direction = None

    def _get_observation(self):
        shape = self.observation_space.shape
        frame = np.zeros(shape, dtype=np.uint8)

        # add target
        dot_size = 6
        direction = np.random.choice([-1, 0, 1], (2,))
        pos = shape[0] // 2 + shape[0] // 4 * direction
        rect = tuple([slice(*v) for v in zip(pos - dot_size // 2, pos + dot_size // 2)])
        frame[rect] = 255

        self.direction = direction

        return frame

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        return self._get_observation()

    def step(self, action=None):
        scaled = np.array(action).reshape((2, 2)) * np.array([-1, 1])
        cond = np.sum(scaled, axis=-1) == self.direction
        reward = np.all(cond) * 2 - 1

        frame = self._get_observation()
        is_done = False
        info = {}

        return frame, reward, is_done, info

    def close(self):
        pass
