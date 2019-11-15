import copy
import numpy as np
import gym
from gym import spaces
import torch
import torch.optim as optim

from ...core.agent import Agent
from ...core.report import Report
from ...utils import rsample_bernoulli, polyak_update
from crackdown import transforms
from ...memory import GameReplay
from ...embedding.image import ImageEmbedding
from ..a2c.critic import TemporalDifferenceCritic
from ..a2c.actor import Actor

__all__ = [
    'SoftActorCriticAgent',
]


class SoftActorCriticAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 transform: transforms.Compose = None,
                 replay: object = None,
                 batch_size: int = 8,
                 embedding_capacity: int = 128,
                 actor_learning_rate: float = 5e-4,
                 critic_learning_rate: float = 1e-4,
                 discount_factor: float = 0.95,
                 temperature: float = 2e-1,
                 polyak: float = 0.995,
                 start_exploration_steps: int = 1000,
                 report: object = None):

        super().__init__()

        observation_space = env.observation_space
        self.action_space = action_space = env.action_space

        assert isinstance(observation_space, spaces.Box)

        self.iteration = 0
        self.sum_reward = 0
        self.rolling_reward = 0

        self.batch_size = batch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.temperature = temperature
        self.polyak = polyak
        self.start_exploration_steps = start_exploration_steps

        self.observation_transform = transform or transforms.EMPTY
        self.replay = replay or GameReplay(1000)
        self.report = report or Report()

        input_channels = observation_space.shape[-1]
        self.embedding = ImageEmbedding(input_channels, embedding_capacity, 16)
        self.policy = Actor(self.embedding.shape[0], action_space)
        self.quality_network_1 = TemporalDifferenceCritic(self.embedding.shape[0], self.policy.input_shape[-1], discount_factor)
        self.quality_network_2 = TemporalDifferenceCritic(self.embedding.shape[0], self.policy.input_shape[-1], discount_factor)
        self.value_network = TemporalDifferenceCritic(self.embedding.shape[0], 0, discount_factor)
        self.target_value_network = copy.deepcopy(self.value_network)

        self.critic_optimizer = optim.Adam(self.parameters(), lr=self.critic_learning_rate)

        policy_params = list(self.embedding.parameters()) + list(self.policy.parameters())
        self.policy_optimizer = optim.Adam(policy_params, lr=self.actor_learning_rate)

        # self.report.add_hparams({
        #     'batch_size': self.batch_size,
        #     'actor_learning_rate': self.actor_learning_rate,
        #     'critic_learning_rate': self.critic_learning_rate,
        #     'temperature': self.temperature,
        #     'polyak': self.polyak,
        #     'start_exploration_steps': self.start_exploration_steps
        # })

        self.reset()

    def prepare_state(self, state):
        view = self.observation_transform(state)
        view = view[np.newaxis, :]
        view = torch.from_numpy(view).float().to(self.device)

        return view

    def report_observation_embedding(self, n_samples: int = 9999):
        n_samples = min(n_samples, len(self.replay))
        batch = self.replay.batch(n_samples)
        batch = [torch.from_numpy(v).float().to(self.device) for v in batch]
        state, _, _, _ = batch
        embedding, _ = self.embedding.forward(state)

        self.report.add_embedding(embedding, label_img=state, global_step=self.iteration, tag='observation')

        return embedding, state

    def reset(self):
        pass

    def predict(self, state, deterministic: bool = False):
        exploration_threshold = self.iteration / self.start_exploration_steps
        if np.random.random() > exploration_threshold:
            return self.actor.head.random().squeeze(dim=0).numpy()

        state = self.prepare_state(state)

        with torch.no_grad():
            embedding, feature_map = self.embedding.forward(state)
            action = self.policy.predict(embedding, deterministic)
            action = action.detach().cpu().squeeze().numpy()

        self.report.add_images('feature_map', feature_map.unsqueeze(2)[0], global_step=self.iteration)

        return action

    def update(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = self.observation_transform(state)
        next_state = self.observation_transform(next_state)

        self.replay.put(state, reward, action, next_state)

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

        state, reward, action, next_state = batch
        reward = reward.view(-1, 1)

        # embedding
        embedding, signal = self.embedding.forward(state)
        next_embedding, next_signal = self.embedding.forward(next_state)

        # quality network
        target_next_value = self.target_value_network.forward(next_embedding).detach()
        q1_loss, q1_quality, q1_advantage = self.quality_network_1.update(embedding, action, reward, None, None, target_next_value)
        q2_loss, q2_quality, q2_advantage = self.quality_network_2.update(embedding, action, reward, None, None, target_next_value)

        # resample action
        action_probs = self.policy.forward(embedding.detach())
        action = rsample_bernoulli(action_probs)

        # entropy
        distribution = self.policy.distribution(action_probs)
        action_entropy = distribution.entropy().mean(dim=-1, keepdim=True)
        entropy = self.temperature * action_entropy

        # value network
        q1_quality = self.quality_network_1.forward(embedding, action)
        q2_quality = self.quality_network_2.forward(embedding, action)
        min_q = torch.min(q1_quality, q2_quality).detach()
        value_loss, quality, advantage = self.value_network.update(embedding.detach(), None, min_q + entropy.detach(), None, None, 0)

        # loss
        critic_loss = (value_loss + q1_loss + q2_loss).mean()
        policy_quality = (self.quality_network_1.forward(embedding.detach(), action) + entropy).mean()
        policy_loss = -policy_quality
        loss = self.critic_learning_rate * critic_loss + policy_loss

        # update
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update target value network
        polyak_update(self.target_value_network, self.value_network, self.polyak)

        report = {
            'embedding': embedding,
            'next_embedding': next_embedding,
            'signal': signal,
            'next_signal': next_signal,
            'quality': quality.mean().item(),
            'advantage': advantage.mean().item(),
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item(),
            'loss': loss.item(),
        }

        self.report.add_scalar('quality', quality.mean().item(), global_step=self.iteration)
        self.report.add_scalar('advantage', advantage.mean().item(), global_step=self.iteration)
        self.report.add_scalar('q1_loss', q1_loss.mean().item(), global_step=self.iteration)
        self.report.add_scalar('q2_loss', q2_loss.mean().item(), global_step=self.iteration)
        self.report.add_scalar('value_loss', value_loss.mean().item(), global_step=self.iteration)
        self.report.add_scalar('critic_loss', critic_loss.item(), global_step=self.iteration)
        self.report.add_scalar('policy_quality', policy_quality.item(), global_step=self.iteration)
        self.report.add_scalar('action_entropy', action_entropy.mean().item(), global_step=self.iteration)
        self.report.add_scalar('loss', loss.item(), global_step=self.iteration)

        return report
