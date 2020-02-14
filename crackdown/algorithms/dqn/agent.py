import copy
import numpy as np
import torch
from torch import nn
from torch import optim
import gym

from ...core.agent import Agent
from ...core.report import Report
from ... import transforms
from ...memory import TensorBuffer
from ...embedding.image import ImageEmbedding
from .quality_network import QualityNetwork
from ...core.policy import make_action_head
from ...utils import polyak_update

__all__ = [
    'DeepQualityNetworkAgent',
]

BATCH_TEMPLATE = (
    ('state', 0),
    ('action', 0),
    ('next_state', 0),
    ('reward', 0),
    ('done', 0),
)


class DeepQualityNetworkAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 transform: transforms.Compose = None,
                 replay: object = None,
                 batch_size: int = 8,
                 embedding_capacity: int = 128,
                 learning_rate: float = 1e-3,
                 discount_factor: float = 0.95,
                 start_exploration_steps: int = 1000,
                 min_epsilon: float = 0.01,
                 clip_gradient: float = 1.0,
                 target_update_period: int = 10,
                 double_dqn: bool = True,
                 dueling_dqn: bool = False,
                 report: object = None):

        super().__init__()

        self.iteration = 0
        self.sum_reward = 0
        self.episode_reward = 0
        self.rolling_reward = 0
        self.observation_transform = transform or transforms.EMPTY

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_exploration_steps = start_exploration_steps
        self.min_epsilon = min_epsilon
        self.clip_gradient = clip_gradient
        self.target_update_period = target_update_period
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.report = report or Report()

        self.embedding = ImageEmbedding(env.observation_space.shape[-1], embedding_capacity, 5)
        self.action_head = make_action_head(env.action_space)
        self.quality_net = QualityNetwork(self.embedding.output_shape[-1], self.action_head.input_shape[-1], discount_factor)
        if double_dqn:
            self.target_quality_net = copy.deepcopy(self.quality_net)
        self.replay = replay or TensorBuffer(10000, BATCH_TEMPLATE)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.reset()

    def predict(self, state, deterministic: bool = False):
        threshold = (1.0 - self.iteration / self.start_exploration_steps)
        threshold = max(self.min_epsilon, threshold)

        if not deterministic and np.random.random() < threshold:
            return self.action_head.sample()[0]
        else:
            with torch.no_grad():
                state = self.observation_transform(state).to(self.device)
                embedding, feature_map = self.embedding.forward(state.unsqueeze(0))
                quality = self.quality_net(embedding)
                action = self.action_head.sample(quality, True)
                action = action.detach().cpu().squeeze(dim=0).numpy()

                return action

    def update(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)

        state = self.observation_transform(state)
        next_state = self.observation_transform(next_state)

        self.replay.put(
            state=state,
            action=action,
            next_state=next_state,
            reward=float(reward),
            done=bool(is_done)
        )

        if len(self.replay) > self.batch_size:
            batch = self.replay.sample(self.batch_size, device=self.device)
            report = self.train_batch(batch)
        else:
            report = {}

        self.sum_reward += reward
        self.episode_reward += reward
        self.rolling_reward = (1 - 0.1) * self.rolling_reward + 0.1 * reward

        self.report.add_scalar('reward', reward, global_step=self.iteration)
        self.report.add_scalar('sum_reward', self.sum_reward, global_step=self.iteration)
        self.report.add_scalar('rolling_reward', self.rolling_reward, global_step=self.iteration)

        if is_done:
            self.report.add_scalar('episode_reward', self.episode_reward, global_step=self.iteration)
            self.episode_reward = 0

        self.iteration += 1

        return report

    def train_batch(self, batch: list) -> dict:
        state, action, next_state, reward, done = batch

        embedding, signal = self.embedding.forward(state)
        next_embedding, signal = self.embedding.forward(next_state)

        with torch.no_grad():
            if self.double_dqn:
                next_quality = self.target_quality_net(next_embedding)
            else:
                next_quality = None

        loss, advantage = self.quality_net.update(embedding, action, reward, next_embedding, next_quality)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_gradient)
        self.optimizer.step()

        if self.iteration % self.target_update_period == 0:
            polyak_update(self.target_quality_net, self.quality_net, 0.0)

        self.report.add_scalar('advantage', advantage.mean().item(), global_step=self.iteration)
        self.report.add_scalar('loss', loss.item(), global_step=self.iteration)

        return {}
