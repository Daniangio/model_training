from math import log2

import torch
import torch.nn as nn
from common.base_module import BaseModule
from common.siren import Siren
from settings import settings


def _init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight, gain=1)
    if type(m) == nn.BatchNorm2d:
        torch.nn.init.uniform_(m.weight, -1, 1)


class Completer(BaseModule):
    def __init__(self, input_size: int = 256, output_size: int = 128, features_map: int = 32, hidden_layers: int = 3,
                 **_):
        super(Completer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.feature_maps = features_map
        self.hidden_layers = hidden_layers
        self.num_halving_size_layers = int(log2(int(input_size / output_size)))

        self.conv_list = []
        self.conv_list.append(
            nn.Conv2d(1, features_map, kernel_size=int(output_size / 2 + 1), stride=1, padding=int(output_size / 4),
                      bias=True))
        self.conv_list.append(nn.BatchNorm2d(features_map))
        self.conv_list.append(Siren())
        new_features_map = features_map

        for i in range(1, self.hidden_layers + 1):
            self.conv_list.append(
                nn.Conv2d(new_features_map, new_features_map, kernel_size=int(output_size / 4 + 1), stride=1,
                          padding=int(output_size / 8), bias=True))
            self.conv_list.append(nn.BatchNorm2d(new_features_map))
            self.conv_list.append(Siren())
            new_features_map = new_features_map

        for i in range(1, self.num_halving_size_layers + 1):
            self.conv_list.append(nn.Conv2d(new_features_map, features_map, 4, 2, 1, bias=True))
            self.conv_list.append(nn.BatchNorm2d(features_map))
            self.conv_list.append(Siren())
            new_features_map = features_map
        self.conv_layer = nn.Sequential(*self.conv_list)

        self.lateral_conv_list = []
        self.lateral_conv_list.append(nn.Conv2d(1, features_map, kernel_size=3, stride=1, padding=1, bias=True))
        self.lateral_conv_list.append(nn.BatchNorm2d(features_map))
        self.lateral_conv_list.append(Siren())
        new_features_map = features_map

        for i in range(1, self.hidden_layers + 1):
            self.lateral_conv_list.append(nn.Conv2d(new_features_map, new_features_map, kernel_size=3, stride=1,
                                                    padding=1, bias=True))
            self.lateral_conv_list.append(nn.BatchNorm2d(new_features_map))
            self.lateral_conv_list.append(Siren())
            new_features_map = new_features_map

        for i in range(1, self.num_halving_size_layers + 1):
            self.lateral_conv_list.append(nn.Conv2d(new_features_map, features_map, 4, 2, 1, bias=True))
            self.lateral_conv_list.append(Siren())
            new_features_map = features_map
        self.lateral_conv_list = nn.Sequential(*self.lateral_conv_list)

        self.final_layer = []
        self.final_layer.append(nn.Conv2d(2 * new_features_map, 1, 3, 1, 1, bias=True))
        self.final_layer.append(nn.Sigmoid())
        self.final_layer = nn.Sequential(*self.final_layer)

    def forward(self, x):
        x1 = self.conv_layer(x)
        x2 = self.lateral_conv_list(x)
        x = torch.cat((x1, x2), dim=1)
        return self.final_layer(x)

    def init_weights(self):
        self.apply(_init_weights)

    def set_preprocessing_params(self, params):
        return params

    def get_weights_filename(self):
        return f'coordmap_completer_isz{self.input_size}_osz{self.output_size}_fm{self.feature_maps}' \
               f'_hl{self.hidden_layers}_v{settings.coordmap_model_version}.pt'
