from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def make_divisible(value: float, divisor: int, min_value: int | None = None) -> int:
    min_value = divisor if min_value is None else min_value
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def _conv_bn(
    ch_in: int,
    ch_out: int,
    kernel_size: int,
    *,
    stride: int = 1,
    groups: int = 1,
    activation: type[nn.Module] | None = nn.ReLU6,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(ch_out),
    ]
    if activation is not None:
        layers.append(activation(inplace=True))
    return nn.Sequential(*layers)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            _conv_bn(ch_in, ch_in, 3, stride=stride, groups=ch_in, activation=nn.ReLU),
            _conv_bn(ch_in, ch_out, 1, activation=nn.ReLU),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MobileNetV1(nn.Module):
    def __init__(self, ch_in: int = 3, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn(ch_in, 32, 3, stride=2),
            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


class InvertedBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, expand_ratio: float, stride: int):
        super().__init__()
        hidden_dim = int(ch_in * expand_ratio)
        self.use_res_connect = stride == 1 and ch_in == ch_out

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(_conv_bn(ch_in, hidden_dim, 1))
        layers.extend(
            [
                _conv_bn(hidden_dim, hidden_dim, 3, stride=stride, groups=hidden_dim),
                _conv_bn(hidden_dim, ch_out, 1, activation=None),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.use_res_connect:
            return x + out
        return out


class MobileNetV2(nn.Module):
    def __init__(self, ch_in: int = 3, num_classes: int = 1000):
        super().__init__()
        configs = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.stem = _conv_bn(ch_in, 32, 3, stride=2)
        layers: list[nn.Module] = []
        input_channel = 32
        for expand_ratio, channels, repeats, stride in configs:
            for repeat in range(repeats):
                current_stride = stride if repeat == 0 else 1
                layers.append(InvertedBlock(input_channel, channels, expand_ratio, current_stride))
                input_channel = channels
        self.blocks = nn.Sequential(*layers)
        self.last_conv = _conv_bn(input_channel, 1280, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).flatten(1)
        return self.classifier(x)


class HSwish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_channels = make_divisible(in_channels // reduction_ratio, 8)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layers(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        use_se: bool,
        activation: type[nn.Module],
    ):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(_conv_bn(in_channels, hidden_dim, 1, activation=activation))

        layers.append(
            _conv_bn(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride=stride,
                groups=hidden_dim,
                activation=activation,
            )
        )
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))
        layers.append(_conv_bn(hidden_dim, out_channels, 1, activation=None))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.use_res_connect:
            return x + out
        return out


class MobileNetV3(nn.Module):
    def __init__(
        self,
        mode: str = "small",
        num_classes: int = 1000,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mode == "large":
            configs = [
                (3, 1, 16, False, nn.ReLU, 1),
                (3, 4, 24, False, nn.ReLU, 2),
                (3, 3, 24, False, nn.ReLU, 1),
                (5, 3, 40, True, nn.ReLU, 2),
                (5, 3, 40, True, nn.ReLU, 1),
                (5, 3, 40, True, nn.ReLU, 1),
                (3, 6, 80, False, HSwish, 2),
                (3, 2.5, 80, False, HSwish, 1),
                (3, 2.3, 80, False, HSwish, 1),
                (3, 2.3, 80, False, HSwish, 1),
                (3, 6, 112, True, HSwish, 1),
                (3, 6, 112, True, HSwish, 1),
                (5, 6, 160, True, HSwish, 2),
                (5, 6, 160, True, HSwish, 1),
                (5, 6, 160, True, HSwish, 1),
            ]
            last_conv_out = 960
            classifier_hidden = 1280
        else:
            configs = [
                (3, 1, 16, True, nn.ReLU, 2),
                (3, 4.5, 24, False, nn.ReLU, 2),
                (3, 3.67, 24, False, nn.ReLU, 1),
                (5, 4, 40, True, HSwish, 2),
                (5, 6, 40, True, HSwish, 1),
                (5, 6, 40, True, HSwish, 1),
                (5, 3, 48, True, HSwish, 1),
                (5, 3, 48, True, HSwish, 1),
                (5, 6, 96, True, HSwish, 2),
                (5, 6, 96, True, HSwish, 1),
                (5, 6, 96, True, HSwish, 1),
            ]
            last_conv_out = 576
            classifier_hidden = 1024

        input_channel = make_divisible(16 * width_mult, 8)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            HSwish(inplace=True),
        )

        blocks: list[nn.Module] = []
        for kernel_size, expand_ratio, out_channels, use_se, activation, stride in configs:
            block_out = make_divisible(out_channels * width_mult, 8)
            blocks.append(
                InvertedResidualBlock(
                    in_channels=input_channel,
                    out_channels=block_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    use_se=use_se,
                    activation=activation,
                )
            )
            input_channel = block_out
        self.blocks = nn.ModuleList(blocks)

        last_conv_out = make_divisible(last_conv_out * width_mult, 8)
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, last_conv_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_conv_out),
            HSwish(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_out, classifier_hidden),
            HSwish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).flatten(1)
        return self.classifier(x)
