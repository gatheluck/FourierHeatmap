import logging
import pathlib
from dataclasses import dataclass
from typing import Final, Tuple, cast

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

import fhmap.fourier.heatmap as heatmap
import fhmap.schema as schema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        topk (Tuple): Tuple of int which you want to know error.
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
    topk: Tuple = (1, 5)
    weightpath: str = MISSING


cs = ConfigStore.instance()
cs.store(name="eval_fhmap", node=EvalFhmapConfig)
# arch
cs.store(group="arch", name="resnet50", node=schema.Resnet50Config)
cs.store(group="arch", name="resnet56", node=schema.Resnet56Config)
cs.store(group="arch", name="wideresnet40", node=schema.Wideresnet40Config)
# NOTE: If you want to add your custom architecture, please add YourCustomArchConfig as a node here.
# dataset
cs.store(group="dataset", name="cifar10", node=schema.Cifar10Config)
cs.store(group="dataset", name="imagenet100", node=schema.Imagenet100Config)
cs.store(group="dataset", name="imagenet", node=schema.ImagenetConfig)
# NOTE: If you want to add your custom dataset, please add YourCustomDatasetConfig as a node here.
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

    # Setup datamodule
    root: Final[pathlib.Path] = cwd / "data"
    datamodule = instantiate(cfg.dataset, cfg.batch_size, cfg.env.num_workers, root)
    datamodule.prepare_data()

    # Setup model
    arch = instantiate(cfg.arch, num_classes=datamodule.num_classes)
    arch.load_state_dict(torch.load(weightpath))
    arch = arch.to(device)
    arch.eval()

    fhmaps = heatmap.eval_fourier_heatmap(
        datamodule.input_size,
        arch,
        datamodule,
        cfg.num_samples,
        cfg.eps,
        cast(torch.device, device),  # needed for passing mypy check.
        cfg.topk,
    )

    for k, fourier_heatmap in zip(cfg.topk, fhmaps):
        heatmap.save_fourier_heatmap(
            fourier_heatmap / 100.0, pathlib.Path("."), f"_top{k}"
        )


if __name__ == "__main__":
    eval_fhmap()
