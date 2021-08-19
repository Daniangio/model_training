import torch
import torch.nn as nn
from common.base_module import BaseModule
from settings import settings


def _init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight, gain=2)
    if type(m) == nn.BatchNorm2d:
        torch.nn.init.uniform_(m.weight, -1, 1)


class Encoder(BaseModule):
    def __init__(self, input_size: tuple = (64, 64, 3), output_size: int = 100, feature_maps=64, **_):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.feature_maps = feature_maps
        last_kernel_size = int(self.input_size[0] / 32)

        self.main = nn.Sequential(
            # input is input_size[2] x input_size[0] x input_size[1]
            nn.Conv2d(input_size[2], feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps) x (input_size[0] / 2) x (input_size[1] / 2)
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps) x (input_size[0] / 4) x (input_size[1] / 4)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps) x (input_size[0] / 8) x (input_size[1] / 8)
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps) x (input_size[0] / 16) x (input_size[1] / 16)
            nn.Conv2d(feature_maps * 8, feature_maps * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps) x (input_size[0] / 32) x (input_size[1] / 32)
            nn.Conv2d(feature_maps * 16, output_size, last_kernel_size, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return 2 * self.main(x)

    def init_weights(self):
        self.apply(_init_weights),

    def set_preprocessing_params(self, params):
        params['mean'] = [0.5]
        params['std'] = [0.5]
        return params

    def get_weights_filename(self):
        return f'dcgan_encoder_isz{self.input_size[0]}_osz{self.output_size}_fm{self.feature_maps}' \
               f'_v{settings.dcgan_model_version}.pt'
