import copy
import pathlib
from collections import OrderedDict
from typing import Final, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

import fhmap
import fhmap.fourier as fourier
from fhmap.fourier.noise import AddFourierNoise


def create_fourier_heatmap_from_error_matrix(
    error_matrix: torch.Tensor,
) -> torch.Tensor:
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


def insert_fourier_noise(transforms: List, basis: torch.Tensor) -> None:
    """Insert Fourier noise transform to given a list of transform by inplace operation.

    Note:
        If Normalize transform is included in the given list, Fourier noise transform is added to just before Normalize transform.
        If not, Fourier noise transform is added at the end of the list.

    Args:
        transforms (List): A list of transform.
        basis (torch.Tensor): 2D Fourier basis.

    """

    def _get_index_of_normalize(transforms: List) -> Optional[int]:
        for index, transform in enumerate(transforms):
            if isinstance(transform, torchvision.transforms.Normalize):
                return index

        return None

    normalize_index: Final = _get_index_of_normalize(transforms)
    insert_index: Final = normalize_index if normalize_index else -1

    transforms.insert(insert_index, AddFourierNoise(basis))


def eval_fourier_heatmap(
    input_size: int,
    ignore_edge_size: int,
    eps: float,
    arch: nn.Module,
    dataset: torchvision.datasets.VisionDataset,
    batch_size: int,
    device: torch.device,
    topk: Tuple[int, ...] = (1,),
    savedir: Optional[pathlib.Path] = None,
) -> List[torch.Tensor]:
    """Evaluate Fourier Heat Map about given architecture and dataset.

    Args:
        input_size (int): A size of input image.
        ignore_edge_size (int): A size of the edge to ignore.
        eps (float): L2 norm size of Fourier basis.
        arch (nn.Module): An architecture to be evaluated.
        dataset (torchvision.datasets.VisionDataset): A dataset used for evaluation.
        batch_size (int): A size of batch.
        device (torch.device): A device used for calculation.
        topk (Tuple[int, ...], optional): Tuple of int which you want to know error.
        savedir (pathlib.Path, optional): Path to the directory where the results will be saved.

    Returns:
        List[torch.Tensor]: List of Fourier Heat Map.

    """
    if input_size % 2 != 0:
        raise ValueError("currently we only support even input size.")

    height: Final[int] = input_size
    width: Final[int] = height // 2 + 1
    fhmap_height: Final[int] = height - 2 * ignore_edge_size
    fhmap_width: Final[int] = width - ignore_edge_size

    if not isinstance(dataset.transform, torchvision.transforms.Compose):
        raise ValueError(
            f"type of dataset.transform should be torchvision.transforms.Compose, not {type(dataset.transform)}"
        )
    original_transforms: Final = dataset.transform.transforms

    error_matrix_dict = {
        k: torch.zeros(fhmap_height * fhmap_width).float() for k in topk
    }

    spectrums = fourier.get_spectrum(height, width, ignore_edge_size, ignore_edge_size)
    with tqdm(
        spectrums, ncols=160, total=fhmap_height * fhmap_width
    ) as pbar:  # without total progress par might not be shown.
        for i, spectrum in enumerate(pbar):  # Size of basis is [height, width]
            basis = fourier.spectrum_to_basis(spectrum, l2_normalize=True) * eps

            # insert fourier noise by inpalce operation
            noised_transforms = copy.deepcopy(original_transforms)
            insert_fourier_noise(noised_transforms, basis)

            # overwrite torchvision.transforms.Compose.transforms
            dataset.transform.transforms = noised_transforms

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

            for k, mean_err in zip(
                topk, fhmap.eval_mean_errors(arch, loader, device, topk)
            ):
                error_matrix_dict[k][i] = mean_err

            # show result to pbar
            results = OrderedDict()
            for k, v in error_matrix_dict.items():
                results[f"err{k}"] = v[i].item()
            pbar.set_postfix(results)
            pbar.update()

    fourier_heatmaps: Final[List[torch.Tensor]] = [
        create_fourier_heatmap_from_error_matrix(
            error_matrix_dict[k].view(fhmap_height, fhmap_width)
        )
        for k in topk
    ]

    if savedir:
        for k, fourier_heatmap in zip(topk, fourier_heatmaps):
            save_fourier_heatmap(fourier_heatmap / 100.0, savedir, f"_top{k}")

    return fourier_heatmaps
