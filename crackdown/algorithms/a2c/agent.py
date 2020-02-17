from collections import OrderedDict
from typing import Union
import torch
import torch.nn as nn
import gym
from gym import spaces

from ...core.agent import Agent
from ...embedding.image import ImageEmbedding
from crackdown import transforms
from .actor import Actor
from .critic import TemporalDifferenceCritic
from .advantage_critic import AdvantageCritic


__all__ = [
    'ActorCriticAgent',
]

CRITIC_CLASSES = (
    TemporalDifferenceCritic,
    AdvantageCritic,
)


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
                 critic: Union[CRITIC_CLASSES] = TemporalDifferenceCritic,
                 embedding_capacity: int = 128,
                 critic_learning_factor: float = 1.0,
                 discount_factor: float = 0.95,
                 temperature: float = 1e-2):

        super().__init__()

        assert isinstance(env.observation_space, spaces.Box)

        self.critic_learning_factor = critic_learning_factor
        self.temperature = temperature

        self.observation_transform = transform or transforms.EMPTY

        self.embedding = ImageEmbedding(env.observation_space.shape[-1], embedding_capacity, 16)
        self.actor = Actor(self.embedding.output_shape[-1], env.action_space)
        self.critic = critic(self.embedding.output_shape[-1], self.actor.output_shape[-1], discount_factor)

        self.reset()

    def reset(self):
        self.apply(weight_reset)

    def predict(self, state, deterministic: bool = False):
        with torch.no_grad():
            state = self.observation_transform(state).to(self.device)
            embedding = self.embedding.forward(state.unsqueeze(0))
            action = self.actor.predict(embedding, deterministic)
            action = action.detach().cpu().squeeze(dim=0).numpy()

        return action

    def update(self, batch: OrderedDict) -> dict:
        state, action, next_state, reward, done = batch.values()

        embedding = self.embedding.forward(state)
        next_embedding = self.embedding.forward(next_state)
        
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
        loss = self.critic_learning_factor * critic_loss + actor_loss  # + entropy_loss

        return {
            'loss': loss,
            'embedding': embedding,
            'next_embedding': next_embedding,
            'advantage': advantage,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
        }
