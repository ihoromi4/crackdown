import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

from ...core.agent import Agent
from ...core.report import Report
from ...memory import TensorBuffer
from ...embedding.image import ImageEmbedding
from crackdown import transforms
from .actor import Actor
from .advantage_critic import AdvantageCritic


__all__ = [
    'ActorCriticAgent',
]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class ActorCriticAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 transform: transforms.Compose = None,
                 replay: object = None,
                 batch_size: int = 8,
                 update_period: int = 1,
                 embedding_capacity: int = 128,
                 critic_learning_rate: float = 1e-3,
                 actor_learning_rate: float = 1e-4,
                 discount_factor: float = 0.95,
                 temperature: float = 1e-2,
                 start_exploration_steps: int = 1000,
                 report: object = None):

        super().__init__()

        observation_space = env.observation_space
        self.action_space = env.action_space

        assert isinstance(env.observation_space, spaces.Box)

        self.iteration = 0
        self.sum_reward = 0
        self.rolling_reward = 0

        self.batch_size = batch_size
        self.update_period = update_period
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.temperature = temperature
        self.start_exploration_steps = start_exploration_steps

        self.observation_transform = transform or transforms.EMPTY
        self.replay = replay or TensorBuffer(1000)
        self.report = report or Report()

        input_channels = observation_space.shape[-1]
        self.embedding = ImageEmbedding(input_channels, embedding_capacity, 16)
        self.actor = Actor(self.embedding.output_shape[-1], self.action_space)
        self.critic = AdvantageCritic(self.embedding.output_shape[-1], self.actor.output_shape[-1], discount_factor)

        self.optimizer = optim.Adam(self.parameters(), lr=self.actor_learning_rate, weight_decay=0)

        self.reset()

    def reset(self):
        self.apply(weight_reset)
        self.replay.reset()
        
    def predict(self, state, deterministic: bool = False):
        exploration_threshold = self.iteration / self.start_exploration_steps
        if np.random.random() > exploration_threshold:
            return self.actor.head.random().squeeze(dim=0).numpy()

        with torch.no_grad():
            state = self.observation_transform(state)
            embedding, feature_map = self.embedding.forward(state.unsqueeze(0))
            action = self.actor.predict(embedding, deterministic)
            action = action.detach().cpu().squeeze(dim=0).numpy()

        self.report.add_images('feature_map', feature_map.unsqueeze(2)[0], global_step=self.iteration)

        return action

    def update(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)

        state = self.observation_transform(state)

        self.replay.put(state, torch.from_numpy(action), float(reward), bool(is_done))

        if len(self.replay) < self.batch_size:
            return {}

        if self.iteration % self.update_period == self.update_period - 1:
            batch = self.replay.batch(self.batch_size)
            report = self.train_batch(batch)
        else:
            report = {}

        self.sum_reward += reward
        self.rolling_reward = (1 - 0.1) * self.rolling_reward + 0.1 * reward
        self.report.add_scalar('reward', reward, global_step=self.iteration)
        self.report.add_scalar('sum_reward', self.sum_reward, global_step=self.iteration)
        self.report.add_scalar('rolling_reward', self.rolling_reward, global_step=self.iteration)

        self.iteration += 1

        return report
        
    def train_batch(self, batch: list) -> dict:
        state, action, next_state, reward = batch

        embedding, signal = self.embedding.forward(state)
        next_embedding, next_signal = self.embedding.forward(next_state)
        
        next_action = self.actor.sample(next_embedding)

        # entropy
        distribution = self.actor.distribution(next_embedding)
        action_entropy = distribution.entropy().mean()
        # entropy_loss = -self.temperature * action_entropy
        entropy_reward = torch.clamp(self.temperature * action_entropy, 0, 1)

        reward = reward + entropy_reward.detach()
        critic_loss, advantage = self.critic.update(embedding, action, reward, next_embedding, next_action)
        actor_loss, action_probs = self.actor.update(embedding, advantage.detach(), action)

        # final loss
        critic_loss_factor = self.critic_learning_rate / self.actor_learning_rate
        loss = critic_loss_factor * critic_loss + actor_loss  # + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        report = {
            'embedding': embedding,
            'next_embedding': next_embedding,
            'signal': signal,
            'next_signal': next_signal,
            # 'quality': quality.mean().item(),
            'advantage': advantage.mean().item(),
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'loss': loss.item(),
        }

        # self.report.add_scalar('quality', quality.mean().item(), global_step=self.iteration)
        self.report.add_scalar('advantage', advantage.mean().item(), global_step=self.iteration)
        self.report.add_scalar('critic_loss', critic_loss.item(), global_step=self.iteration)
        self.report.add_scalar('actor_loss', actor_loss.item(), global_step=self.iteration)
        self.report.add_scalar('action_entropy', action_entropy.item(), global_step=self.iteration)
        self.report.add_scalar('loss', loss.item(), global_step=self.iteration)

        return report
