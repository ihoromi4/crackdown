import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

from ..core.agent import Agent
from ..core.report import Report
from ..memory import GameReplay
from ..embedding.image import ImageEmbedding
from ..embedding import transforms
from .actor import Actor
from .critic import TemporalDifferenceCritic


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
                 report: object = None):

        super().__init__()

        observation_space = env.observation_space
        action_space = env.action_space

        assert isinstance(observation_space, spaces.Box)
        assert isinstance(action_space, spaces.MultiBinary)

        self.action_space = action_space
        
        self.critic_learning_rate = 5e-1
        self.actor_learning_rate = 5e-4
        self.temperature = 1e-2

        self.observation_transform = transform or transforms.EMPTY
        self.replay = replay or GameReplay(100)
        self.report = report or Report()

        input_channels = observation_space.shape[-1]
        self.embedding = ImageEmbedding(input_channels, 128, 16)
        self.actor = Actor(self.embedding.shape[0], action_space.shape[0])
        self.critic = TemporalDifferenceCritic(self.embedding.shape[0], action_space.shape[0])

        self.optimizer = optim.Adam(self.parameters(), lr=self.actor_learning_rate)

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
        state = self.prepare_state(state)
        
        with torch.no_grad():
            embedding, feature_map = self.embedding.forward(state)
            action = self.actor.predict(embedding, deterministic)
            action = action.detach().cpu().squeeze().numpy()

        self.report.add_images('feature_map', feature_map.unsqueeze(2)[0])
        self.report.add_scalars('action', {str(i): v for i, v in zip(range(action.shape[0]), action)})

        return action

    def update(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = self.observation_transform(state)
        next_state = self.observation_transform(next_state)
        
        self.replay.put(state, reward, action, next_state)
        
        batch_size = 8
        batch = self.replay.batch(batch_size)
        report = self.train_batch(batch)

        self.report.add_scalar('reward', reward)

        return report
        
    def train_batch(self, batch: list) -> dict:
        if isinstance(batch[0], np.ndarray):
            batch = [torch.from_numpy(v).float().to(self.device) for v in batch]
            
        state, reward, action, next_state = batch
        reward = reward.view(-1, 1)

        embedding, signal = self.embedding.forward(state)
        next_embedding, next_signal = self.embedding.forward(next_state)
        
        next_action = self.actor.sample(next_embedding)
        
        critic_loss, quality, advantage = self.critic.update(
            embedding, action,
            reward,
            next_embedding, next_action)
        
        actor_loss, action_probs = self.actor.update(embedding, advantage.detach(), action)
        
        # entropy
        action_entropy = -(torch.log(action_probs) * action_probs).mean()
        entropy_loss = -self.temperature * action_entropy
        
        # final loss
        loss = self.critic_learning_rate * critic_loss + actor_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        report = {
            'embedding': embedding,
            'next_embedding': next_embedding,
            'signal': signal,
            'next_signal': next_signal,
            'quality': quality.mean().item(),
            'advantage': advantage.mean().item(),
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'loss': loss.item(),
        }

        self.report.add_scalar('quality', quality.mean().item())
        self.report.add_scalar('advantage', advantage.mean().item())
        self.report.add_scalar('critic_loss', critic_loss.item())
        self.report.add_scalar('actor_loss', actor_loss.item())
        self.report.add_scalar('entropy_loss', entropy_loss.item())
        self.report.add_scalar('loss', loss.item())

        return report
