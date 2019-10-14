import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from gym import spaces

from ..core.agent import Agent
from ..core.replay import GameReplay
from .embedding import ViewEmbedding
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


class ToNumpy:
    def __call__(self, img):
        arr = np.array(img)
        
        if len(arr.shape) == 2:
            arr = arr[:, :, np.newaxis]
            
        return arr


class Transpose:
    def __init__(self, order):
        self.order = order
        
    def __call__(self, arr):
        return np.transpose(arr, self.order)


class ActorCriticAgent(Agent):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.MultiBinary, replay=None):
        super().__init__()
        
        assert isinstance(action_space, spaces.MultiBinary)
        
        self.action_space = action_space
        
        self.critic_learning_rate = 1e-1
        self.actor_learning_rate = 5e-4
        self.temperature = 0.001
        
        self.replay = replay or GameReplay(1000)
        
        self.state_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            ToNumpy(),
            Transpose((2, 0, 1))
        ])

        input_channels = observation_space.shape[-1]
        self.embedding = ViewEmbedding(input_channels, 128, 16)
        self.actor = Actor(self.embedding.shape, action_space.shape[0])
        self.critic = TemporalDifferenceCritic(self.embedding.shape, action_space.shape[0])

        self.optimizer = optim.Adam(self.parameters(), lr=self.actor_learning_rate)

    def reset(self):
        self.apply(weight_reset)
        self.replay.reset()
        
    def prepare_state(self, state):
        view = self.state_transform(state)
        view = view[np.newaxis, :]
        view = torch.from_numpy(view).float().to(self.device)
        
        return view
        
    def predict(self, state, deterministic: bool = False):
        state = self.prepare_state(state)
        
        with torch.no_grad():
            embedding, _ = self.embedding.forward(state)
            action = self.actor.predict(embedding, deterministic)
            
        return action.detach().cpu().squeeze().numpy()

    def update(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = self.state_transform(state)
        next_state = self.state_transform(next_state)
        
        self.replay.put(state, reward, action, next_state)
        
        batch_size = 8
        batch = self.replay.batch(batch_size)
        report = self.train_batch(batch)

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
        nn.utils.clip_grad_norm_(self.parameters(), 0.1)
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
        
        return report
