import copy
from collections import OrderedDict
import torch
import gym

from ...core.agent import Agent
from ...core.report import Report
from ... import transforms
from ...embedding.image import ImageEmbedding
from .quality_network import QualityNetwork
from ...core.policy import make_action_head
from ...utils import polyak_update

__all__ = [
    'DeepQualityNetworkAgent',
]


class DeepQualityNetworkAgent(Agent):
    batch_template = (
        ('state', 0),
        ('action', 0),
        ('next_state', 0),
        ('reward', 0),
        ('done', 0),
    )

    def __init__(self,
                 env: gym.Env,
                 transform: transforms.Compose = None,
                 embedding_capacity: int = 128,
                 discount_factor: float = 0.95,
                 target_update_factor: float = 0.99,
                 double_dqn: bool = True,
                 dueling_dqn: bool = False):

        super().__init__()

        self.observation_transform = transform or transforms.EMPTY

        self.target_update_factor = target_update_factor
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn

        self.embedding = ImageEmbedding(env.observation_space.shape[-1], embedding_capacity, 5)
        self.action_head = make_action_head(env.action_space)
        self.quality_net = QualityNetwork(
            self.embedding.output_shape[-1],
            self.action_head.input_shape[-1],
            discount_factor,
            dueling_dqn
        )

        if double_dqn:
            self.target_quality_net = copy.deepcopy(self.quality_net)

        if dueling_dqn:
            self.value_net = QualityNetwork(self.embedding.output_shape[-1], 1, discount_factor)

        self.reset()

    def predict(self, state, deterministic: bool = False):
        with torch.no_grad():
            state = self.observation_transform(state).to(self.device)
            embedding = self.embedding.forward(state.unsqueeze(0))
            quality = self.quality_net(embedding)
            action = self.action_head.sample(quality, True)
            action = action.detach().cpu().squeeze(dim=0).numpy()

            return action

    def update(self, batch: OrderedDict) -> dict:
        state, action, next_state, reward, done = batch.values()

        embedding = self.embedding.forward(state)
        next_embedding = self.embedding.forward(next_state)

        with torch.no_grad():
            if self.double_dqn:
                next_quality = self.target_quality_net(next_embedding)
            else:
                next_quality = None

        loss, advantage = self.quality_net.update(embedding, action, reward, next_embedding, next_quality)

        polyak_update(self.target_quality_net, self.quality_net, self.target_update_factor)

        return {
            'loss': loss,
            'advantage': advantage,
            'embedding': embedding,
            'next_embedding': next_embedding,
        }
