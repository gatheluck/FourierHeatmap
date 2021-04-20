from typing import Iterator, cast

import torch
import torch.fft as fft


def get_spectrum(
    height: int,
    width: int,
    height_ignore_edge_size: int = 0,
    width_ignore_edge_size: int = 0,
    low_center: bool = False,
) -> Iterator[torch.Tensor]:
    """Return generator of specrum matrics of 2D Fourier basis.

    Note:
        - height_ignore_edge_size and width_ignore_edge_size are used for getting subset of spectrum.
          e.g.) In the original paper, Fourier Heat Map was created for a 63x63 low frequency region for ImageNet.
        - We generate spectrum one by one to avoid waste of memory.
          e.g.) We need to generate more than 25,000 basis for ImageNet.

    Args:
        height (int): Height of spectrum.
        width (int): Width of spectrum.
        height_ignore_edge_size (int, optional): Size of the edge to ignore about height.
        width_ignore_edge_size (int, optional): Size of the edge to ignore about width.
        low_center (bool, optional): If True, returned low frequency centered spectrum.

    Yields:
        torch.Tensor: Generator of spectrum size of (H, W).

    """
    indices = torch.arange(height * width)
    if low_center:
        B = height * width
        indices = torch.cat([indices[B // 2 :], indices[: B // 2]])

    # drop ignoring edges
    indices = indices.view(height, width)
    if height_ignore_edge_size:
        indices = indices[height_ignore_edge_size:-height_ignore_edge_size, :]
    if width_ignore_edge_size:
        indices = indices[:, :-width_ignore_edge_size]
    indices = indices.flatten()

    for idx in indices:
        yield torch.nn.functional.one_hot(idx, num_classes=B).view(
            height, width
        ).float()


def spectrum_to_basis(
    spectrum: torch.Tensor, l2_normalize: bool = True
) -> torch.Tensor:
    """Convert spectrum matrix to Fourier basis by 2D FFT. Shape of returned basis is (H, W).

    Note:
        - Currently, only supported the case H==W. If H!=W, returned basis might be wrong.
        - In order to apply 2D FFT, dim argument of torch.fft.irfftn should be =(-2,-1).

    Args:
        spectrum (torch.Tensor): 2D spectrum matrix. Its shape should be (H, W//2+1).
                                 Here, (H, W) represent the size of 2D Fourier basis we want to get.
        l2_normalize (bool): If True, basis is l2 normalized.

    Returns:
        torch.Tensor: 2D Fourier basis.

    """
    assert len(spectrum.size()) == 2
    H = spectrum.size(-2)  # currently, only consider the case H==W
    basis = fft.irfftn(spectrum, s=(H, H), dim=(-2, -1))

    if l2_normalize:
        return cast(torch.Tensor, basis / basis.norm(dim=(-2, -1))[None, None])
    else:
        return cast(torch.Tensor, basis)


if __name__ == "__main__":
    import pathlib
    from typing import Final

    import torchvision

    height: Final = 32
    width: Final = 17
    height_ignore_edge_size: Final = 0
    width_ignore_edge_size: Final = 0
    image_size: Final = height
    padding: Final = 2

    savedir: Final = pathlib.Path("outputs")
    savedir.mkdir(exist_ok=True)

    spectrum = torch.stack(
        [
            x
            for x in get_spectrum(
                height,
                width,
                height_ignore_edge_size,
                width_ignore_edge_size,
                low_center=True,
            )
        ]
    )
    basis = torch.stack(
        [spectrum_to_basis(s, l2_normalize=True) * 10.0 for s in spectrum]
    )

    basis_rightside = torchvision.utils.make_grid(
        basis[:, None, :, :], nrow=width - width_ignore_edge_size, padding=padding
    )[:, (image_size + padding) :, : -(image_size + padding)]
    basis_leftside = torch.flip(basis_rightside, (-2, -1))
    all_basis = torch.cat(
        [
            basis_leftside[:, :, : -(image_size + padding)],
            basis_rightside[:, :, padding:],
        ],
        dim=-1,
    )
    torchvision.utils.save_image(
        all_basis, savedir / "basis.png", nrow=width - width_ignore_edge_size
    )
