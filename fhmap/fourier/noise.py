import random
from typing import Final, cast

import torch


class AddFourierNoise:
    """
    Add Fourier noise to RGB channels respectively.
    This class is able to use as same as the functions in torchvision.transforms.

    Attributes:
        basis (torch.Tensor): scaled 2D Fourier basis. In the original paper, it is reperesented by 'v*U_{i,j}'.

    """

    def __init__(self, basis: torch.Tensor):
        assert len(basis.size()) == 2
        assert basis.size(0) == basis.size(1)
        self.basis: Final[torch.Tensor] = basis

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        c, h, w = x.shape[-3:]
        assert c == 3
        assert h >= 1 or w >= 1

        fourier_noise = self.basis.unsqueeze(0).repeat(c, 1, 1)  # (c, h, w)

        # Sign of noise is chosen uniformly at random from {-1, 1} per channel.
        # In the original paper,this factor is prepresented by 'r'.
        fourier_noise[0, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[1, :, :] *= random.randrange(-1, 2, 2)
        fourier_noise[2, :, :] *= random.randrange(-1, 2, 2)

        return cast(torch.Tensor, torch.clamp(x + fourier_noise, min=0.0, max=1.0))
