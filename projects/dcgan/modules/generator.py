from math import log2

import torch
import torch.nn as nn
from common.base_module import BaseModule
from settings import settings


def _init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal(m.weight, gain=2)
    if type(m) == nn.BatchNorm2d:
        torch.nn.init.uniform_(m.weight, -1, 1)


class Generator(BaseModule):
    def __init__(self, input_vector_size: int = 100, output_size: tuple = (64, 64, 3), feature_maps: int = 64, **_):
        super(Generator, self).__init__()
        self.input_vector_size = input_vector_size
        self.output_size = output_size
        self.feature_maps = feature_maps
        self.num_conv_layers = int(log2(output_size[0] / 4) - 1)  # -1 perché poi aggiungo il layer finale

        self.conv_list = []
        self.conv_list.append(
            nn.ConvTranspose2d(input_vector_size, feature_maps * 2 ** self.num_conv_layers, 4, 1, 0, bias=False))
        self.conv_list.append(nn.BatchNorm2d(feature_maps * 2 ** self.num_conv_layers))
        self.conv_list.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        # state size is (feature_maps * 2 ** num_conv_layers) x 4 x 4

        for i in range(1, self.num_conv_layers + 1):
            self.conv_list.append(nn.ConvTranspose2d(feature_maps * 2 ** (self.num_conv_layers - i + 1),
                                                     feature_maps * 2 ** (self.num_conv_layers - i),
                                                     4, 2, 1, bias=False))
            self.conv_list.append(nn.BatchNorm2d(feature_maps * 2 ** (self.num_conv_layers - i)))
            self.conv_list.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            # state size is (feature_maps * 2^(num_conv_layers - i)) x (4 x 2^i) x (4 x 2^i)

        # state size is (feature_maps) x (output_size[0] / 2) x (output_size[0] / 2)
        self.conv_list.append(nn.ConvTranspose2d(feature_maps, output_size[2], 4, 2, 1, bias=False))
        self.conv_list.append(nn.Tanh())
        self.conv_layer = nn.Sequential(*self.conv_list)

    def forward(self, x):
        return self.conv_layer(x)

    def init_weights(self):
        self.apply(_init_weights)

    def set_preprocessing_params(self, params):
        params['mean'] = [0.5]
        params['std'] = [0.5]
        return params

    def get_weights_filename(self):
        return f'dcgan_generator_isz{self.input_vector_size}_osz{self.output_size[0]}_fm{self.feature_maps}' \
               f'_v{settings.dcgan_model_version}.pt'
