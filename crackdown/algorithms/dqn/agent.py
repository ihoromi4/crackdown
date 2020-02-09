import copy
import numpy as np
import torch
from torch import nn
from torch import optim
import gym

from ...core.agent import Agent
from ... import transforms
from ...memory import TensorBuffer
from ...embedding.image import ImageEmbedding
from .quality_network import QualityNetwork
from ...core.policy import make_action_head

__all__ = [
    'DeepQualityNetworkAgent',
]


class DeepQualityNetworkAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 transform: transforms.Compose = None,
                 replay: object = None,
                 batch_size: int = 8,
                 embedding_capacity: int = 128,
                 learning_rate: float = 1e-3,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 clip_gradient: float = 1.0):

        super().__init__()

        self.iteration = 0
        self.observation_transform = transform or transforms.EMPTY

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.clip_gradient = clip_gradient

        self.embedding = ImageEmbedding(env.observation_space.shape[-1], embedding_capacity, 16)
        self.action_head = make_action_head(env.action_space)
        self.quality_net = QualityNetwork(self.embedding.output_shape[-1], self.action_head.input_shape[-1], discount_factor)
        # self.target_quality_net = copy.deepcopy(self.quality_net)

        buffer_template = (
            ('state', (env.observation_space.shape[-1],) + env.observation_space.shape[:-1], torch.float32),
            ('action', (self.action_head.output_shape[-1],), torch.float32),
            ('next_state', (env.observation_space.shape[-1],) + env.observation_space.shape[:-1], torch.float32),
            ('reward', (1,), torch.float32),
            ('done', (1,), torch.float32),
        )
        batch_template = (
            ('state', 0),
            ('action', 0),
            ('next_state', 0),
            ('reward', 0),
            ('done', 0),
        )

        self.replay = replay or TensorBuffer(10000, buffer_template, batch_template)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)

        self.reset()

    def predict(self, state, deterministic: bool = False):
        if not deterministic and np.random.random() < self.epsilon:
            return self.action_head.sample()
        else:
            with torch.no_grad():
                state = self.observation_transform(state)
                embedding, feature_map = self.embedding.forward(state.unsqueeze(0))
                quality = self.quality_net(embedding)
                action = self.action_head.sample(quality)
                action = action.detach().cpu().squeeze(dim=0).numpy()

                return action

    def update(self, state, action, next_state, reward, is_done: bool = False) -> dict:
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state)
        reward = np.array(reward)

        state = self.observation_transform(state)
        next_state = self.observation_transform(next_state)

        self.replay.put(state, torch.from_numpy(action), next_state, float(reward), bool(is_done))

        batch = self.replay.batch(self.batch_size)
        report = self.train_batch(batch)

        self.iteration += 1

        return report

    def train_batch(self, batch: list) -> dict:
        state, action, next_state, reward, done = batch

        embedding, signal = self.embedding.forward(state)
        next_embedding, signal = self.embedding.forward(next_state)

        loss, advantage = self.quality_net.update(embedding, action, reward, next_embedding)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_gradient)
        self.optimizer.step()

        return {}
