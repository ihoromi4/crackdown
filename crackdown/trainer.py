import itertools
import numpy as np
import gym
from torch import nn
from torch import optim

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from .memory import TensorBuffer
from .core.report import Report

__all__ = [
    'Trainer'
]


class Trainer:
    def __init__(self,
                 agent: object,
                 replay: object = None,
                 batch_size: int = 8,
                 learning_rate: float = 1e-3,
                 clip_gradient: float = 1.0,
                 episode_steps_limit: int = 0,
                 start_exploration_steps: int = 1000,
                 min_epsilon: float = 0.01,
                 render: bool = False,
                 device: str = None,
                 report: object = None):

        self.agent = agent.to(device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clip_gradient = clip_gradient
        self.episode_steps_limit = episode_steps_limit
        self.start_exploration_steps = start_exploration_steps
        self.min_epsilon = min_epsilon
        self.render = render
        self.device = device
        self.report = report or Report()

        self.replay = replay or TensorBuffer(10000)
        self.replay.batch_template = agent.batch_template

        self.iteration = 0
        self.sum_reward = 0
        self.episode_reward = 0
        self.rolling_reward = 0

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, weight_decay=0)

    def on_step_end(self, report: dict):
        pass

    def on_episode_end(self, report: dict):
        pass

    def on_train_end(self, report: dict):
        pass

    def update_replay(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)

        state = self.agent.observation_transform(state)
        next_state = self.agent.observation_transform(next_state)

        self.replay.put(
            state=state,
            action=action,
            next_state=next_state,
            reward=float(reward),
            done=bool(is_done)
        )

        if len(self.replay) > self.batch_size:
            batch = self.replay.sample(self.batch_size, device=self.device)
            loss = self.agent.update(batch)
            self.optimize(loss)

            self.report.add_scalar('loss', loss.item(), global_step=self.iteration)

    def sample_action(self, state, deterministic: bool = False):
        threshold = (1.0 - self.iteration / self.start_exploration_steps)
        threshold = max(self.min_epsilon, threshold)

        if not deterministic and np.random.random() < threshold:
            return self.agent.ction_head.sample()[0]
        else:
            return self.agent.predict(state, deterministic)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_gradient)
        self.optimizer.step()

    def train_episode(self, env, steps_limit: int = 0) -> dict:
        assert isinstance(env, gym.Env)
        assert isinstance(steps_limit, int)

        rewards = []
        sum_reward = 0
        avg_reward = 0
        step_i = 0
        total_steps = max(0, steps_limit) or None

        state = env.reset()
        action = self.agent.predict(state)

        with tqdm(desc="Step", total=total_steps) as steps_bar:
            for step_i in itertools.count(1):

                next_state, reward, done, info = env.step(action)

                sum_reward += reward
                avg_reward = sum_reward / step_i

                steps_bar.update(1)
                steps_bar.set_postfix(sum_reward=sum_reward, avg_reward=avg_reward)

                if self.render:
                    env.render()

                rewards.append(reward)

                loss = self.update_replay(state, action, next_state, reward, done)
                action = self.agent.predict(next_state)

                # self.on_step_end(report)

                state = next_state

                if done:
                    break
                elif (steps_limit > 0) and (step_i >= steps_limit):
                    break

                self.sum_reward += reward
                self.episode_reward += reward
                self.rolling_reward = (1 - 0.1) * self.rolling_reward + 0.1 * reward

                self.report.add_scalar('reward', reward, global_step=self.iteration)
                self.report.add_scalar('sum_reward', self.sum_reward, global_step=self.iteration)
                self.report.add_scalar('rolling_reward', self.rolling_reward, global_step=self.iteration)

                if done:
                    self.report.add_scalar('episode_reward', self.episode_reward, global_step=self.iteration)
                    self.episode_reward = 0

                self.iteration += 1

        return {
            'steps': step_i,
            'rewards': rewards,
        }

    def train(self, env, steps_limit: int = 0, episodes_limit: int = 0, catch_interruption: bool = False) -> dict:
        assert isinstance(env, gym.Env)
        assert isinstance(steps_limit, int)
        assert isinstance(episodes_limit, int)
        assert isinstance(catch_interruption, bool)

        passed_steps = 0
        episode_i = 0
        rewards = []
        total_episodes = max(0, episodes_limit) or None

        try:
            with tqdm(desc="Episode", total=total_episodes) as progress_bar:
                for episode_i in itertools.count(1):
                    progress_bar.update(1)

                    episode_steps_limit = max(0, self.episode_steps_limit)
                    remains_steps = max(0, steps_limit - passed_steps)
                    episode_steps_limit = min(episode_steps_limit, remains_steps) or max(episode_steps_limit, remains_steps)
                    report = self.train_episode(env, episode_steps_limit)

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
