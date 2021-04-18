import math
from typing import Type, Union

import torch
from torch import nn
from torch.nn import functional

__all__ = ["wideresnet16", "wideresnet28", "wideresnet40"]


class BasicBlock(nn.Module):
    def __init__(
        self, in_planes: int, out_planes: int, stride: int, droprate: float = 0.0
    ) -> None:
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = droprate
        self.equal_io = in_planes == out_planes
        self.shortcut = (
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            if not self.equal_io
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.relu1(self.bn1(x))
        if not self.equal_io:  # for shortcut
            x = self.shortcut(o)  # type: ignore

        o = self.relu2(self.bn2(self.conv1(o)))
        if self.droprate > 0:
            o = functional.dropout(o, p=self.droprate, training=self.training)
        o = self.conv2(o)

        return x + o


class NetworkBlock(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        in_planes: int,
        out_planes: int,
        n: Union[int, float],
        stride: int,
        droprate: float = 0.0,
    ) -> None:
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, n, stride, droprate)

    def _make_layer(
        self,
        block: Type[nn.Module],
        in_planes: int,
        out_planes: int,
        n: Union[int, float],
        stride: int,
        droprate: float,
    ) -> nn.Module:
        layers = []
        for i in range(int(n)):
            layers.append(
                block(  # type: ignore
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    droprate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self, num_classes: int, depth: int, width: int = 1, droprate: float = 0.0
    ) -> None:
        super(WideResNet, self).__init__()
        nc = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock

        self.depth = depth
        self.width = width

        self.conv1 = nn.Conv2d(3, nc[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(block, nc[0], nc[1], n, 1, droprate)
        self.block2 = NetworkBlock(block, nc[1], nc[2], n, 2, droprate)
        self.block3 = NetworkBlock(block, nc[2], nc[3], n, 2, droprate)
        self.bn = nn.BatchNorm2d(nc[3])
        self.relu = nn.ReLU(True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nc[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(-1, 64 * self.width)
        return self.fc(x)


def wideresnet16(
    num_classes: int = 1000, widening_factor: int = 8, droprate: float = 0.3
) -> nn.Module:
    """factory class of Wide ResNet-16.
    Parameters
    ----------
    num_classes : int
        Number of output class.
    widening_factor : int
        This number decide width of model which is represented by k in original paper.
    droprate : float
        Probability of dropout.
    """
    return WideResNet(num_classes, 16, widening_factor, droprate)


def wideresnet28(
    num_classes: int = 1000, widening_factor: int = 10, droprate: float = 0.3
) -> nn.Module:
    """factory class of Wide ResNet-28.
    Parameters
    ----------
    num_classes : int
        Number of output class.
    widening_factor : int
        This number decide width of model which is represented by k in original paper.
    droprate : float
        Probability of dropout.
    """
    return WideResNet(num_classes, 28, widening_factor, droprate)


def wideresnet40(
    num_classes: int = 1000, widening_factor: int = 2, droprate: float = 0.3
) -> nn.Module:
    """factory class of Wide ResNet-40.
    Parameters
    ----------
    num_classes : int
        Number of output class.
    widening_factor : int
        This number decide width of model which is represented by k in original paper.
    droprate : float
        Probability of dropout.
    """
    return WideResNet(num_classes, 40, widening_factor, droprate)
