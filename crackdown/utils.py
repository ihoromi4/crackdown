import torch

__all__ = [
    'rsample_bernoulli',
    'polyak_update',
]


def rsample_bernoulli(probs):
    random = torch.rand(probs.shape)
    z = torch.log(random) - torch.log(1 - random) + torch.log(probs) - torch.log(1 - probs)

    return torch.sigmoid(z)


def polyak_update(target_network, network, factor: float):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(factor * target_param.data + (1 - factor) * param.data)
