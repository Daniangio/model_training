import torch
import torch.nn as nn
from common.base_module import BaseModule
from settings import settings


def _init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight, gain=2)
    if type(m) == nn.BatchNorm2d:
        torch.nn.init.uniform_(m.weight, -1, 1)


class Discriminator(BaseModule):
    def __init__(self, input_size: int, in_channels=3, feature_maps=128, **_):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.feature_maps = feature_maps
        last_kernel_size = int(self.input_size / 32)

        self.main = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 256 x 256
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 128 x 128
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 64 x 64
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 32 x 32
            nn.Conv2d(feature_maps * 8, feature_maps * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 16 x 16
            nn.Conv2d(feature_maps * 16, 1, last_kernel_size, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

    def init_weights(self):
        self.apply(_init_weights)

    def get_weights_filename(self):
        return f'triplet_discriminator_sz{self.input_size}_fm{self.feature_maps}' \
               f'_v{settings.triplet_model_version}.pt'
