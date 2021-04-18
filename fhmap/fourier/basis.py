from typing import cast

import torch
import torch.fft as fft


def spectrum_to_basis(
    spectrum: torch.Tensor, l2_normalize: bool = True
) -> torch.Tensor:
    """
    Convert spectrum matrix to Fourier basis by 2D FFT.
    Shape of returned basis is (B, H, W).

    Note:
        - Currently, only supported the case H==W. If H!=W, returned basis might be wrong.
        - In order to apply 2D FFT, dim argument of torch.fft.irfftn should be =(-2,-1).

    Args:
        spectrum (torch.Tensor): Batched 2D spectrum matrix. Its shape should be (B, H, W//2+1).
                                 Here, (H, W) represent the size of 2D Fourier basis you want to get.
        l2_normalize (bool): If True, basis is l2 normalized.

    Returns:
        torch.Tensor: Batched 2D Fourier basis.

    """
    _, H, _ = spectrum.size()  # currently, only consider the case H==W
    basis = fft.irfftn(spectrum, s=(H, H), dim=(-2, -1))

    if l2_normalize:
        return cast(torch.Tensor, basis / basis.norm(dim=(-2, -1))[:, None, None])
    else:
        return cast(torch.Tensor, basis)


def get_basis_spectrum(
    height: int, width: int, low_center: bool = False
) -> torch.Tensor:
    """
    Get all specrum matrics of 2D Fourier basis.
    Size of return is (H*W, H, W).

    Args:
        height (int): Height of spectrum.
        width (int): Width of spectrum.
        low_center (bool): If True, returned low frequency centered spectrum.

    Returns
        torch.Tensor: Batched spectrum.

    """
    x = torch.arange(height * width)
    spectrum = torch.nn.functional.one_hot(x).view(-1, height, width).float()
    if low_center:
        B = height * width
        return torch.cat([spectrum[B // 2 :], spectrum[: B // 2]])
    else:
        return spectrum
