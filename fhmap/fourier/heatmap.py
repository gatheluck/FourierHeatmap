import pathlib
from collections import OrderedDict
from typing import Final, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

import fhmap
import fhmap.fourier as fourier
from fhmap.factory.dataset import BaseDataModule


def create_fourier_heatmap_from_error_matrix(error_matrix: torch.Tensor) -> torch.Tensor:
    """Create Fourier Heat Map from error matrix (about quadrant 1 and 4).

    Note:
        Fourier Heat Map is symmetric about the origin.
        So by performing an inversion operation about the origin, Fourier Heat Map is created from error matrix.

    Args:
        error_matrix (torch.Tensor): The size of error matrix should be (H, H/2+1). Here, H is height of image.
                                     This error matrix shoud be about quadrant 1 and 4.

    Returns:
        torch.Tensor (torch.Tensor): Fourier Heat Map created from error matrix.

    """
    assert len(error_matrix.size()) == 2
    assert error_matrix.size(0) == 2 * (error_matrix.size(1) - 1)

    fhmap_rightside = error_matrix[1:, :-1]
    fhmap_leftside = torch.flip(fhmap_rightside, (0, 1))
    return torch.cat([fhmap_leftside[:, :-1], fhmap_rightside], dim=1)


def save_fourier_heatmap(
    fhmap: torch.Tensor, savedir: pathlib.Path, suffix: str = ""
) -> None:
    """Save Fourier Heat Map as a png image.

    Args:
        fhmap (torch.Tensor): Fourier Heat Map.
        savedir (pathlib.Path): Path to the directory where the results will be saved.
        suffix (str, optional): Suffix which is attached to result file of Fourier Heat Map.

    """
    torch.save(fhmap, savedir / ("fhmap_data" + suffix + ".pth"))  # save raw data.
    sns.heatmap(
        fhmap.numpy(),
        vmin=0.0,
        vmax=1.0,
        cmap="jet",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.savefig(savedir / ("fhmap" + suffix + ".png"))
    plt.close("all")  # This is needed for continuous figure generation.


def eval_fourier_heatmap(
    height: int,
    arch: nn.Module,
    datamodule: BaseDataModule,
    num_samples: int,
    eps: float,
    device: torch.device,
    topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]:
    """Evaluate Fourier Heat Map about given architecture and dataset.

    Args:
        height (int):
        arch (nn.Module): An architecture to be evaluated.
        datamodule (BaseDataModule):
        num_samples (int): The number of samples used from dataset. If -1, use all samples.
        eps (float): The L2 norm size of Fourier basis.
        device (torch.device): A device used for calculation.
        topk (Tuple[int, ...], optional): Tuple of int which you want to know error.

    Returns:
        List[torch.Tensor]: List of Fourier Heat Map.

    """
    assert height % 2 == 0, "currently we only support even input size."
    width: Final[int] = height // 2 + 1

    error_matrix_dict = {k: torch.zeros(height * width).float() for k in topk}
    spectrum = fourier.get_basis_spectrum(height, width, low_center=True)

    with tqdm(
        torch.chunk(
            fourier.spectrum_to_basis(spectrum, l2_normalize=True),
            spectrum.size(0),
        ),
        ncols=80,
    ) as pbar:
        for i, basis in enumerate(pbar):  # Size of basis is [1, height, width]
            basis = basis.squeeze(0) * eps
            datamodule.setup("test", basis=basis)
            loader = datamodule.test_dataloader(num_samples)

            for k, mean_err in zip(topk, fhmap.eval_mean_errors(arch, loader, device, topk)):
                error_matrix_dict[k][i] = mean_err

            # show result to pbar
            results = OrderedDict()
            for k, v in error_matrix_dict.items():
                results[f"err{k}"] = v[i]
            pbar.set_postfix(results)
            pbar.update()

    return [create_fourier_heatmap_from_error_matrix(error_matrix_dict[k].view(height, width)) for k in topk]
