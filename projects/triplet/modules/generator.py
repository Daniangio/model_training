from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.base_module import BaseModule
from common.siren import Siren
from settings import settings


def _init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight, gain=2)
    if type(m) == nn.BatchNorm2d:
        torch.nn.init.uniform_(m.weight, -1, 1)


class Generator(BaseModule):
    ''' Old version with exponentially growing features_map
        def __init__(self, input_size: int, in_channels=4, features_map=16, **_):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.out_channels = 3
        self.features_map = features_map

        self.conv1 = nn.Conv2d(in_channels, features_map, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(features_map)
        self.conv2 = nn.Conv2d(features_map, features_map * 2, 4, 2, 1, bias=False)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        number_features = int(features_map / 2 * (input_size / 4) * (input_size / 4))
        number_convtr_layers = int(log2(input_size)) - 1
        self.convtr_list = []
        self.convtr_list.append(nn.ConvTranspose2d(number_features, self.out_channels * 2**number_convtr_layers, 4, 2, 1, bias=False))
        for i in range(1, number_convtr_layers + 1):
            self.convtr_list.append(nn.BatchNorm2d(self.out_channels * 2 ** (number_convtr_layers - i + 1)))
            self.convtr_list.append(Siren())
            self.convtr_list.append(nn.ConvTranspose2d(self.out_channels * 2**(number_convtr_layers - i + 1), self.out_channels * 2**(number_convtr_layers - i), 4, 2, 1, bias=False))
        self.convtr_layer = nn.Sequential(*self.convtr_list)
    '''

    def __init__(self, input_size: int, in_channels=4, features_map=64, **_):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.out_channels = 3
        self.features_map = features_map

        self.conv1 = nn.Conv2d(in_channels, features_map, 4, 2, 1, bias=False)
        self.bn2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(features_map, features_map, 4, 2, 1, bias=False)
        self.mp4 = nn.BatchNorm2d(features_map)

        number_features = int(features_map / 4 * (input_size / 4) * (input_size / 4))
        number_convtr_layers = int(log2(input_size)) - 1
        self.convtr_list = []
        self.convtr_list.append(nn.ConvTranspose2d(number_features, features_map, 4, 2, 1, bias=False))
        self.convtr_list.append(nn.BatchNorm2d(features_map))
        self.convtr_list.append(Siren())
        for i in range(1, number_convtr_layers):
            self.convtr_list.append(nn.ConvTranspose2d(features_map, features_map, 4, 2, 1, bias=False))
            self.convtr_list.append(nn.BatchNorm2d(features_map))
            self.convtr_list.append(Siren())
        self.convtr_list.append(nn.ConvTranspose2d(features_map, self.out_channels, 4, 2, 1, bias=False))
        self.convtr_layer = nn.Sequential(*self.convtr_list)

    def forward(self, x):
        # x size is (in_channels) x 512 x 512
        x = F.leaky_relu(self.bn2(self.conv1(x)), negative_slope=0.2, inplace=True)
        # x size is (features_map) x 256 x 256
        x = F.leaky_relu(self.mp4(self.conv3(x)), negative_slope=0.2, inplace=True)
        # x size is (features_map * 2) x 128 x 128
        x = x.view(x.size()[0], -1, 1, 1)
        # for i in range(len(self.convtr_list)):
        #     x = F.relu(self.bn_list[i](self.convtr_list[i](x)), inplace=True)
        x = self.convtr_layer(x)
        x = torch.sigmoid(x)
        return x

    def init_weights(self):
        self.apply(_init_weights)

    def get_weights_filename(self):
        return f'triplet_generator_sz{self.input_size}_fm{self.features_map}' \
               f'_v{settings.triplet_model_version}.pt'
