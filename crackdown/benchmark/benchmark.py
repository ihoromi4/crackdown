import gym
from ..trainer import Trainer

__all__ = [
    'Benchmark'
]


class Benchmark(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self, env: gym.Env, runs: int = 10, episodes: int = 1):
        runs_rewards = []

        for run_i in range(runs):
            print('Run:', run_i)

            self.agent.reset()

            report = self.train(env, episodes_limit=episodes)

            runs_rewards.append(report['rewards'])

        return runs_rewards
