import copy
import numpy as np
import gym
from gym import spaces
import torch
import torch.optim as optim
import torch.nn as nn

from ..core.agent import Agent
from ..core.report import Report
from ..embedding import transforms
from ..memory import GameReplay
from ..embedding.image import ImageEmbedding
from ..a2c.critic import TemporalDifferenceCritic
from ..a2c.actor import Actor

__all__ = [
    'SoftActorCriticAgent',
]


def rsample_bernoulli(probs):
    random = torch.rand(probs.shape)
    z = torch.log(random) - torch.log(1 - random) + torch.log(probs) - torch.log(1 - probs)

    return torch.sigmoid(z)


def polyak_update(target_network, network, factor: float):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(factor*param.data + target_param.data*(1.0 - factor))


class SoftActorCriticAgent(Agent):
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

        self.actor_learning_rate = 5e-4
        self.critic_learning_rate = 1e-4 / self.actor_learning_rate
        self.temperature = 1e-2

        self.observation_transform = transform or transforms.EMPTY
        self.replay = replay or GameReplay(100)
        self.report = report or Report()

        input_channels = observation_space.shape[-1]
        self.embedding = ImageEmbedding(input_channels, 128, 16)
        self.quality_network_1 = TemporalDifferenceCritic(self.embedding.shape[0], action_space.shape[0])
        self.quality_network_2 = TemporalDifferenceCritic(self.embedding.shape[0], action_space.shape[0])
        self.value_network = TemporalDifferenceCritic(self.embedding.shape[0], 0)
        self.target_value_network = copy.deepcopy(self.value_network)
        self.policy = Actor(self.embedding.shape[0], action_space.shape[0])

        self.optimizer = optim.Adam(self.parameters(), lr=self.actor_learning_rate)

        self.reset()

    def prepare_state(self, state):
        view = self.observation_transform(state)
        view = view[np.newaxis, :]
        view = torch.from_numpy(view).float().to(self.device)

        return view

    def reset(self):
        pass

    def predict(self, state, deterministic: bool = False):
        state = self.prepare_state(state)

        with torch.no_grad():
            embedding, _ = self.embedding.forward(state)
            action = self.policy.predict(embedding, deterministic)
            action = action.detach().cpu().squeeze().numpy()

        return action

    def update(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = self.observation_transform(state)
        next_state = self.observation_transform(next_state)

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

        # embedding
        embedding, signal = self.embedding.forward(state)
        next_embedding, next_signal = self.embedding.forward(next_state)

        # resample action
        action_probs = self.policy.forward(embedding)
        action = rsample_bernoulli(action_probs)

        # quality network
        target_next_value = self.target_value_network.forward(next_embedding).detach()
        q1_loss, q1_quality, q1_advantage = self.quality_network_1.update(embedding, action.detach(), reward, None, None, target_next_value)
        q2_loss, q2_quality, q2_advantage = self.quality_network_2.update(embedding, action.detach(), reward, None, None, target_next_value)

        # entropy
        action_entropy = -(torch.log(action_probs) * action_probs).mean(dim=-1, keepdim=True)
        entropy = self.temperature * action_entropy

        # value network
        min_q = torch.min(q1_quality, q2_quality).detach()
        value_loss, quality, advantage = self.value_network.update(embedding, None, min_q + entropy.detach(), None, None, 0)

        # loss
        critic_loss = (value_loss + q1_loss + q2_loss).mean()
        policy_loss = -(self.quality_network_1.forward(embedding.detach(), action) + entropy).mean()
        loss = self.critic_learning_rate * critic_loss + policy_loss

        # update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        # update target value network
        polyak_update(self.target_value_network, self.value_network, 0.1)

        report = {
            # 'embedding': embedding,
            # 'next_embedding': next_embedding,
            # 'signal': signal,
            # 'next_signal': next_signal,
            # 'quality': quality.mean().item(),
            # 'advantage': advantage.mean().item(),
            # 'critic_loss': critic_loss.item(),
            # 'actor_loss': actor_loss.item(),
            # 'loss': loss.item(),
        }

        self.report.add_scalar('quality', quality.mean().item())
        self.report.add_scalar('advantage', advantage.mean().item())
        self.report.add_scalar('critic_loss', critic_loss.item())
        self.report.add_scalar('policy_loss', policy_loss.item())
        self.report.add_scalar('entropy', entropy.mean().item())
        self.report.add_scalar('loss', loss.item())

        return report
