import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

from ..core.agent import Agent
from ..core.report import Report
from ..embedding.image import ImageEmbedding
from ..embedding import transforms
from ..a2c.actor import Actor
from .replay import DiscountReplay


__all__ = [
    'PolicyGradientAgent',
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


class PolicyGradientAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 transform: transforms.Compose = None,
                 replay: object = None,
                 batch_size: int = 8,
                 embedding_capacity: int = 128,
                 policy_learning_rate: float = 5e-4,
                 discount_factor: float = 0.95,
                 temperature: float = 5e-1,
                 start_exploration_steps: int = 1000,
                 report: object = None):

        super().__init__()

        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.MultiBinary)

        observation_space = env.observation_space
        self.action_space = env.action_space

        self.iteration = 0
        self.sum_reward = 0
        self.rolling_reward = 0

        self.batch_size = batch_size
        self.policy_learning_rate = policy_learning_rate
        self.temperature = temperature
        self.start_exploration_steps = start_exploration_steps

        self.observation_transform = transform or transforms.EMPTY
        self.replay = replay or DiscountReplay(1000, discount_factor)
        self.report = report or Report()

        input_channels = observation_space.shape[-1]
        self.embedding = ImageEmbedding(input_channels, embedding_capacity, 16)
        self.actor = Actor(self.embedding.shape[0], self.action_space.shape[0])

        self.optimizer = optim.Adam(self.parameters(), lr=self.policy_learning_rate)

        self.reset()

    def reset(self):
        self.apply(weight_reset)
        self.replay.reset()
        
    def prepare_state(self, state):
        view = self.observation_transform(state)
        view = view[np.newaxis, :]
        view = torch.from_numpy(view).float().to(self.device)
        
        return view
        
    def predict(self, state, deterministic: bool = False):
        if self.start_exploration_steps > 0:
            self.start_exploration_steps -= 1
            return self.action_space.sample()

        state = self.prepare_state(state)
        
        with torch.no_grad():
            embedding, feature_map = self.embedding.forward(state)
            action = self.actor.predict(embedding, deterministic)
            action = action.detach().cpu().squeeze().numpy()

        self.report.add_images('feature_map', feature_map.unsqueeze(2)[0])

        return action

    def update(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = self.observation_transform(state)
        next_state = self.observation_transform(next_state)
        
        self.replay.put(state, action, next_state, reward, is_done)
        
        batch = self.replay.batch(self.batch_size)
        report = self.train_batch(batch)

        self.sum_reward += reward
        self.rolling_reward = (1 - 0.1) * self.rolling_reward + 0.1 * reward
        self.report.add_scalar('reward', reward, global_step=self.iteration)
        self.report.add_scalar('sum_reward', self.sum_reward, global_step=self.iteration)
        self.report.add_scalar('rolling_reward', self.rolling_reward, global_step=self.iteration)
        self.iteration += 1

        return report
        
    def train_batch(self, batch: list) -> dict:
        if isinstance(batch[0], np.ndarray):
            batch = [torch.from_numpy(v).float().to(self.device) for v in batch]
            
        state, action, next_state, reward, is_done = batch
        reward = reward.view(-1, 1)

        embedding, _ = self.embedding.forward(state)
        actor_loss, action_probs = self.actor.update(embedding, reward, action)
        
        # entropy
        # action_entropy = -(torch.log(action_probs) * action_probs).mean()
        distribution = self.actor.distribution(action_probs)
        action_entropy = distribution.entropy().mean()
        entropy_loss = -self.temperature * action_entropy
        
        # final loss
        loss = actor_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        report = {
            'embedding': embedding,
            'actor_loss': actor_loss.item(),
            'loss': loss.item(),
        }

        self.report.add_scalar('actor_loss', actor_loss.item(), global_step=self.iteration)
        self.report.add_scalar('action_entropy', action_entropy.item(), global_step=self.iteration)
        self.report.add_scalar('loss', loss.item(), global_step=self.iteration)

        return report
