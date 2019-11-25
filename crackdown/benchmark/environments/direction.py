import numpy as np
import gym
from gym import spaces

__all__ = [
    'DirectionEnvironment',
]


class DirectionEnvironment(gym.Env):
    metadata = {'render.modes': ['human'], 'video.frames_per_second': 30}
    observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
    # center, left, right, up, down, + 4 combinations
    action_space = spaces.Discrete(9)
    reward_range = (-1, 1)
    action_map = {
        0: [0, 0],
        1: [1, 0],
        2: [-1, 0],
        3: [0, 1],
        4: [0, -1],
        5: [1, 1],
        6: [1, -1],
        7: [-1, 1],
        8: [-1, -1],
    }

    def __init__(self, size: tuple = (64, 64), point_size: int = 10, position_noise_std: float = 0.0):
        assert isinstance(size, (tuple, list))
        assert len(size) == 2

        self.observation_space = spaces.Box(low=0, high=255, shape=tuple(size) + (1,), dtype=np.uint8)
        self.point_size = point_size
        self.position_noise_std = position_noise_std

        self.direction = np.random.choice([-1, 0, 1], (2,))
        self.shift = np.random.normal(0, self.position_noise_std, (2,))

    def _step(self):
        self.direction = np.random.choice([-1, 0, 1], (2,))
        self.shift = np.random.normal(0, self.position_noise_std, (2,))

    def _get_observation(self):
        shape = self.observation_space.shape
        frame = np.zeros(shape, dtype=np.uint8)

        # add target
        pos = shape[0] // 2 + shape[0] // 4 * (self.direction + self.shift)
        pos = pos.astype(np.int32)
        rect = tuple([slice(*v) for v in zip(pos - self.point_size // 2, pos + self.point_size // 2)])
        frame[rect] = 255

        return frame

    def _get_reward(self, action: int):
        action = self.action_map[action]
        return np.all(np.equal(action, self.direction)) * 2 - 1

    @property
    def optimal_action(self):
        for i, action in enumerate(self.action_map.values()):
            if np.all(np.equal(action, self.direction)):
                return i

    def reset(self):
        return self._get_observation()

    def render(self, mode='human', close=False):
        return self._get_observation()

    def step(self, action: int):
        action = int(action)
        assert 0 <= action < 9

        reward = self._get_reward(action)

        self._step()

        frame = self._get_observation()
        is_done = False
        info = {}

        return frame, reward, is_done, info

    def close(self):
        pass
