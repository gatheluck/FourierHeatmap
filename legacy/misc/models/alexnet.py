"""
AlexNet
  This implementation is AlexNet model.
  AlexNet V2 is basically same as pytorch (https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)
  
  For details, please refer to the original paper (
    V1: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    V2: https://arxiv.org/pdf/1404.5997.pdf
  )
"""

import torch
import torchvision
from torch import nn


__all__ = [
    'AlexNet_v1',
    'AlexNet_v2',
    'alexnet_v1',
    'alexnet_v2'
]


def convblock(in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
    """
    Returns convolution block
    """
    if use_bn:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
    else:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(True)
        ]


class AlexNet_v1(nn.Module):
    """AlexNet model (version 1)
    Original paper is "ImageNet Classification with Deep Convolutional Neural Networks" (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
    In this implementation, local response normalization layer is replced by batch normalization layer.
    Args:
        num_classes (int): number of classes
        use_bn (bool): if True, use batch normalization layer
    """
    def __init__(self, num_classes=1000, use_bn=True):
        super(AlexNet_v1, self).__init__()

        self.features = nn.Sequential(
            *convblock(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, use_bn=use_bn),
            nn.MaxPool2d(kernel_size=3, stride=2),
            *convblock(in_channels=96, out_channels=256, kernel_size=5, padding=2, use_bn=use_bn),
            nn.MaxPool2d(kernel_size=3, stride=2),
            *convblock(in_channels=256, out_channels=384, kernel_size=3, padding=1, use_bn=use_bn),
            *convblock(in_channels=384, out_channels=384, kernel_size=3, padding=1, use_bn=use_bn),
            *convblock(in_channels=384, out_channels=256, kernel_size=3, padding=1, use_bn=use_bn)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNet_v2(nn.Module):
    """AlexNet model (version 2)
    Original paper is "One Wierd Trick for Parallelizing Convolutional Neural Networks" (https://arxiv.org/pdf/1404.5997.pdf).
    But model specification is based on Alex Krizhevsky's own cuda-convnet2 implementation (https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-imagenet-1gpu.cfg)
    This implementation has use_bn flag, compared to PyTorch implementation (https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py).
    Args:
        num_classes (int): number of classes
        use_bn (bool): if True, use batch normalization layer
    """
    def __init__(self, num_classes=1000, use_bn=True):
        super(AlexNet_v2, self).__init__()

        self.features = nn.Sequential(
            *convblock(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2, use_bn=use_bn),
            nn.MaxPool2d(kernel_size=3, stride=2),
            *convblock(in_channels=64, out_channels=192, kernel_size=5, padding=2, use_bn=use_bn),
            nn.MaxPool2d(kernel_size=3, stride=2),
            *convblock(in_channels=192, out_channels=384, kernel_size=3, padding=1, use_bn=use_bn),
            *convblock(in_channels=384, out_channels=256, kernel_size=3, padding=1, use_bn=use_bn),
            *convblock(in_channels=256, out_channels=256, kernel_size=3, padding=1, use_bn=use_bn),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_v1(num_classes=1000, use_bn=True):
    """AlexNet model (version 1)
    pre-trained model is not available.
    Args:
        num_classes (int): number of classes
        use_bn (bool): If True, returns a model with batch normalization layer
    """

    return AlexNet_v1(num_classes, use_bn)


def alexnet_v2(num_classes=1000, use_bn=True):
    """AlexNet model (version 1)
    pre-trained model is not available.
    Args:
        num_classes (int): number of classes
        use_bn (bool): If True, returns a model with batch normalization layer
    """

    return AlexNet_v2(num_classes, use_bn)