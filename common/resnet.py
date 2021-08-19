import functools
from collections import OrderedDict

import torch

from common.siren import Siren

from torch import nn


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2


class ConvTranspose2dAuto(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2


conv3x3 = functools.partial(Conv2dAuto, kernel_size=3, bias=False)
convT4x4 = functools.partial(ConvTranspose2dAuto, kernel_size=4, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetInverseResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, upsampling=1, conv=convT4x4, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.upsampling, self.conv = expansion, upsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': ConvTranspose2dAuto(self.in_channels, self.expanded_channels, kernel_size=1,
                                            stride=self.upsampling, bias=False, output_padding=self.upsampling - 1),
                'bn': nn.BatchNorm2d(self.expanded_channels)
            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


def activation_func(activation):
    return nn.ModuleDict({
        'relu': nn.ReLU(inplace=True),
        'siren': Siren(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'selu': nn.SELU(inplace=True),
        'none': nn.Identity()
    })[activation]


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation='siren', *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation='siren', *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetInverseBottleNeckBlock(ResNetInverseResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation='siren', *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.upsampling,
                    output_padding=self.upsampling - 1),
            activation_func(activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block: ResNetResidualBlock = ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetInverseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetInverseBottleNeckBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform upsampling directly by convolutional layers that have a stride of 2.'
        upsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, upsampling=upsampling),
            *[block(out_channels * block.expansion,
                    out_channels, upsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=(64, 128, 256, 512), depths=(3, 4, 23, 3),
                 activation='siren', block=ResNetBottleNeckBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        x = self.avg(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, in_channels, blocks_sizes=(16, 32, 64, 2), depths=(2, 3, 2, 2),
                 activation='siren', block=ResNetInverseBottleNeckBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        reshaped_in_channels = int(in_channels / (16 * 16))
        self.gate = nn.Sequential(
            nn.ConvTranspose2d(reshaped_in_channels, self.blocks_sizes[0], kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetInverseLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation,
                               block=block, *args, **kwargs),
            *[ResNetInverseLayer(in_channels * block.expansion,
                                 out_channels, n=n, activation=activation,
                                 block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
        self.out_gate = nn.ConvTranspose2d(self.blocks[-1].blocks[-1].expanded_channels, 3,
                                           kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1, 16, 16)
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_gate(x)
        x = torch.sigmoid(x)
        return x
