from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def _dataset_image_size(dataset: str | None) -> int:
    if dataset is None:
        return 84
    dataset_name = dataset.lower()
    if dataset_name == "cifar":
        return 64
    return 84

class CosineDistLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.utils.parametrizations.weight_norm(self.linear, name="weight", dim=0)
        self.scale_factor = 2.0 if out_dim <= 200 else 10.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        scores = self.scale_factor * self.linear(x)
        return scores.view(*original_shape, -1)


def _init_layer(layer: nn.Module) -> None:
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pool: bool = True, padding: int = 1):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.trunk = nn.Sequential(*layers)
        self.trunk.apply(_init_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class ConvNet(nn.Module):
    def __init__(
        self,
        depth: int,
        flatten: bool = True,
        image_size: int = 84,
        input_channels: int = 3,
        pool_layers: set[int] | None = None,
        no_padding_layers: set[int] | None = None,
    ):
        super().__init__()
        pool_layers = set(range(min(4, depth))) if pool_layers is None else pool_layers
        no_padding_layers = set() if no_padding_layers is None else no_padding_layers

        layers: list[nn.Module] = []
        for layer_index in range(depth):
            in_dim = input_channels if layer_index == 0 else 64
            padding = 0 if layer_index in no_padding_layers else 1
            layers.append(ConvBlock(in_dim, 64, pool=layer_index in pool_layers, padding=padding))
        if flatten:
            layers.append(nn.Flatten())

        self.trunk = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            feature = self.trunk(dummy)
        self.final_feat_dim = int(feature.shape[1]) if flatten else list(feature.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def _conv(in_planes: int, out_planes: int, kernel_size: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = _conv(inplanes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, layers: list[int], flatten: bool = False, image_size: int = 84):
        super().__init__()
        self.inplanes = 64
        self.flatten = flatten

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feature = self._forward_features(dummy)
            self.final_feat_dim = int(feature.flatten(1).shape[1]) if flatten else list(feature.shape[1:])

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        if self.flatten:
            return x.flatten(1)
        return x


def Conv4(dataset: str = "miniImagenet", flatten: bool = True) -> ConvNet:
    return ConvNet(depth=4, flatten=flatten, image_size=_dataset_image_size(dataset))


def Conv6(dataset: str = "miniImagenet", flatten: bool = True) -> ConvNet:
    return ConvNet(depth=6, flatten=flatten, image_size=_dataset_image_size(dataset))


def Conv4NP(dataset: str = "miniImagenet", flatten: bool = True) -> ConvNet:
    return ConvNet(
        depth=4,
        flatten=flatten,
        image_size=_dataset_image_size(dataset),
        pool_layers={0, 1},
        no_padding_layers={0, 1},
    )


def Conv6NP(dataset: str = "miniImagenet", flatten: bool = True) -> ConvNet:
    return ConvNet(
        depth=6,
        flatten=flatten,
        image_size=_dataset_image_size(dataset),
        pool_layers={0, 1},
        no_padding_layers={0, 1},
    )


def ResNet12(_feti: bool = False, dataset: str = "miniImagenet", flatten: bool = True) -> ResNetBackbone:
    return ResNetBackbone([2, 1, 1, 1], flatten=flatten, image_size=_dataset_image_size(dataset))


def ResNet18(_feti: bool = False, dataset: str = "miniImagenet", flatten: bool = True) -> ResNetBackbone:
    return ResNetBackbone([2, 2, 2, 2], flatten=flatten, image_size=_dataset_image_size(dataset))


def ResNet34(_feti: bool = False, dataset: str = "miniImagenet", flatten: bool = True) -> ResNetBackbone:
    return ResNetBackbone([3, 4, 6, 3], flatten=flatten, image_size=_dataset_image_size(dataset))


BACKBONE_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "Conv4": Conv4,
    "Conv4NP": Conv4NP,
    "Conv6": Conv6,
    "Conv6NP": Conv6NP,
    "ResNet12": ResNet12,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
}
