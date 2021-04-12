import logging
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Final, Tuple, cast

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import fhmap
import fhmap.fourier as fourier
import fhmap.schema as schema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def eval_error(
    arch: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return top1 and top5 mean errors.

    Args:
        arch (nn.Module): The architecture to be evaluated.
        loader (DataLoader): The data loader
        device (torch.device): The device used for calculation.

    """
    arch = arch.to(device)
    err1_list, err5_list = list(), list()

    for x, t in loader:
        x, t = x.to(device), t.to(device)

        output = arch(x)
        err1, err5 = fhmap.calc_errors(output, t, topk=(1, 5))
        err1_list.append(err1.item())
        err5_list.append(err5.item())

    mean_err1 = sum(err1_list) / len(err1_list)
    mean_err5 = sum(err5_list) / len(err5_list)
    return mean_err1, mean_err5


def save_fhmap(
    error_matrix: torch.Tensor, savedir: pathlib.Path, suffix: str = ""
) -> None:
    """Save Fourier Heat Map as a png image.

    Args:
        error_matrix (torch.Tensor): The matrix of errors. The size of error matrix should be (H, H/2+1). Here, H is height of image.
        savedir (pathlib.Path): Path to the directory where the results will be saved.
        suffix (str): Suffix which is attached to result file of Fourier Heat Map.

    """
    assert len(error_matrix.size()) == 2
    assert error_matrix.size(0) == 2 * (error_matrix.size(1) - 1)

    # drop the edge for fliping
    error_matrix = error_matrix[1:, :-1]

    # fill left side of error_matrix
    error_matrix = torch.cat(
        [torch.flip(error_matrix, (0, 1))[:, :-1], error_matrix], dim=1
    )

    torch.save(error_matrix, savedir / ("fhmap_data" + suffix + ".pth"))
    sns.heatmap(
        error_matrix.numpy(),
        vmin=0.0,
        vmax=1.0,
        cmap="jet",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.savefig(savedir / ("fhmap" + suffix + ".png"))
    plt.close("all")  # this is needed for continuous figure generation.


@dataclass
class EvalFhmapConfig:
    """

    Attributes:
        arch (schema.ArchConfig): The config of architecture.
        env (schema.EnvConfig): The config of computational environment.
        dataset (schema.DatasetConfig): The config of dataset.
        batch_size (int): The size of batch.
        num_samples (int): The  number of samples used from dataset. If -1, use all samples.
        eps (float): The L2 norm size of Fourier basis.
        weightpath (str): The path to pytorch model weight.

    """
    # grouped configs
    arch: schema.ArchConfig = schema.Resnet56Config  # type: ignore
    env: schema.EnvConfig = schema.DefaultEnvConfig  # type: ignore
    dataset: schema.DatasetConfig = schema.Cifar10Config  # type: ignore
    # ungrouped configs
    batch_size: int = 512
    num_samples: int = -1
    eps: float = 4.0
    weightpath: str = MISSING


cs = ConfigStore.instance()
cs.store(name="eval_fhmap", node=EvalFhmapConfig)
# arch
cs.store(group="arch", name="resnet50", node=schema.Resnet50Config)
cs.store(group="arch", name="resnet56", node=schema.Resnet56Config)
cs.store(group="arch", name="wideresnet40", node=schema.Wideresnet40Config)
# dataset
cs.store(group="dataset", name="cifar10", node=schema.Cifar10Config)
cs.store(group="dataset", name="imagenet100", node=schema.Imagenet100Config)
cs.store(group="dataset", name="imagenet", node=schema.ImagenetConfig)
# env
cs.store(group="env", name="default", node=schema.DefaultEnvConfig)


@hydra.main(config_path="../config", config_name="eval_fhmap")
def eval_fhmap(cfg: EvalFhmapConfig) -> None:
    """Evalutate Fourier heat map. The result is saved under outpus/eval_fhmap.

    Note:
        Currently, we only supports the input of even-sized images.

    Args:
        cfg (EvalFhmapConfig): The config of Fourier heat map evaluation.

    """
    # Make config read only.
    # without this, config values might be changed accidentally.
    OmegaConf.set_readonly(cfg, True)  # type: ignore
    logger.info(OmegaConf.to_yaml(cfg))

    # Set constants.
    # device: The device which is used in culculation.
    # cwd: The original current working directory. hydra automatically changes it.
    # weightpath: The path of target trained weight.
    device: Final = "cuda" if cfg.env.gpus > 0 else "cpu"
    cwd: Final[pathlib.Path] = pathlib.Path(hydra.utils.get_original_cwd())
    weightpath: Final[pathlib.Path] = pathlib.Path(cfg.weightpath)
    savedir: Final[pathlib.Path] = pathlib.Path(cfg.env.savedir)

    # Setup datamodule
    root: Final[pathlib.Path] = cwd / "data"
    datamodule = instantiate(cfg.dataset, cfg.batch_size, cfg.env.num_workers, root)
    datamodule.prepare_data()

    # Setup model
    arch = instantiate(cfg.arch, num_classes=datamodule.num_classes)
    arch.load_state_dict(torch.load(weightpath))
    arch = arch.to(device)
    arch.eval()

    # height and width of error_matrix
    # Note: Since Fourier Heat Map has origin symmetry, it is sufficient to calculate only half the region about width.
    height: Final[int] = datamodule.input_size
    width: Final[int] = height // 2 + 1
    assert height % 2 == 0, "currently we only support even input size."

    error_matrix_top1 = torch.zeros(height * width).float()
    error_matrix_top5 = torch.zeros(height * width).float()

    spectrum = fourier.get_basis_spectrum(height, width, low_center=True)

    with tqdm(
        torch.chunk(
            fourier.spectrum_to_basis(spectrum, l2_normalize=True),
            spectrum.size(0),
        ),
        ncols=80,
    ) as pbar:
        for i, basis in enumerate(pbar):  # Size of basis is [1, height, width]
            basis = basis.squeeze(0) * cfg.eps
            datamodule.setup("test", basis=basis)
            loader = datamodule.test_dataloader(cfg.num_samples)

            mean_err1, mean_err5 = eval_error(arch, loader, cast(torch.device, device))
            error_matrix_top1[i], error_matrix_top5[i] = mean_err1, mean_err5

            results = OrderedDict()
            results["err1"] = mean_err1
            results["err5"] = mean_err5
            pbar.set_postfix(results)
            pbar.update()

    error_matrix_top1 = error_matrix_top1.view(height, width)
    error_matrix_top5 = error_matrix_top5.view(height, width)

    save_fhmap(error_matrix_top1 / 100.0, savedir, "_top1")
    save_fhmap(error_matrix_top5 / 100.0, savedir, "_top5")


if __name__ == "__main__":
    eval_fhmap()
