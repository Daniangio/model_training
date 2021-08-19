import torch
from torch import nn, Tensor


class Siren(nn.Module):
    def __init__(self):
        super(Siren, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)
