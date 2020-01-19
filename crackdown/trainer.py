import itertools
import numpy as np
import gym

__all__ = [
    'Trainer'
]


class Trainer:
    def __init__(self, agent, episode_steps_limit: int = 0, render: bool = False):
        self.agent = agent
        self.episode_steps_limit = episode_steps_limit
        self.render = render

    def on_step_end(self, report: dict):
        pass

    def on_episode_end(self, report: dict):
        pass

    def on_train_end(self, report: dict):
        pass

    def train_episode(self, env, steps_limit: int = 0) -> dict:
        assert isinstance(env, gym.Env)
        assert isinstance(steps_limit, int)

        rewards = []
        step_i = 0

        state = env.reset()
        action = self.agent.predict(state)

        for step_i in itertools.count(1):
            next_state, reward, done, info = env.step(action)

            if self.render:
                env.render()

            rewards.append(reward)

            report = self.agent.update(state, action, next_state, reward, done)
            action = self.agent.predict(next_state)

            self.on_step_end(report)

            state = next_state

            if done:
                print("Episode finished after {} timesteps".format(step_i + 1))
                break
            elif (self.episode_steps_limit > 0) and (step_i >= self.episode_steps_limit):
                break
            if (steps_limit > 0) and (step_i >= steps_limit):
                break

        return {
            'steps': step_i,
            'rewards': rewards,
        }

    def train(self, env, steps_limit: int = 0, episodes_limit: int = 0, catch_interruption: bool = False) -> dict:
        assert isinstance(env, gym.Env)
        assert isinstance(steps_limit, int)
        assert isinstance(episodes_limit, int)

        passed_steps = 0
        episode_i = 0
        rewards = []

        try:
            for episode_i in itertools.count(1):
                print('Episode:', episode_i)

                report = self.train_episode(env, steps_limit - passed_steps)

                passed_steps += report['steps']
                rewards.extend(report['rewards'])
                report['episode'] = episode_i

                self.on_episode_end(report)

                if (episodes_limit > 0) and (episode_i >= episodes_limit):
                    break
                elif (steps_limit > 0) and (passed_steps >= steps_limit):
                    break
        except KeyboardInterrupt:
            if catch_interruption:
                print('Interrupted by User.')
            else:
                raise

        report = {
            'episodes': episode_i,
            'steps': passed_steps,
            'rewards': rewards,
        }

        self.on_train_end(report)

        return report
