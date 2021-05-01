import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Final, Tuple

import hydra
import torch
from tqdm import tqdm
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

import fhmap
import fhmap.schema as schema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class EvalErrorConfig:
    """

    Attributes:
        arch (schema.ArchConfig): The config of architecture.
        env (schema.EnvConfig): The config of computational environment.
        dataset (schema.DatasetConfig): The config of dataset.
        batch_size (int): A size of batch.
        num_samples (int): A number of samples used from dataset. If -1, use all samples.
        topk (Tuple): Tuple of int which you want to know error.
        weightpath (str): A path to pytorch model weight.

    """

    # grouped configs
    arch: schema.ArchConfig = schema.Resnet56Config  # type: ignore
    env: schema.EnvConfig = schema.DefaultEnvConfig  # type: ignore
    dataset: schema.DatasetConfig = schema.Cifar10Config  # type: ignore
    # ungrouped configs
    batch_size: int = 512
    num_samples: int = -1
    topk: Tuple = (1, 5)
    weightpath: str = MISSING


cs = ConfigStore.instance()
cs.store(name="eval_error", node=EvalErrorConfig)
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


@hydra.main(config_path="../config", config_name="eval_error")
def eval_error(cfg: EvalErrorConfig) -> None:
    """Evalutate Fourier heat map. The result is saved under outpus/eval_error.

    Note:
        Currently, we only supports the input of even-sized images.

    Args:
        cfg (EvalErrorConfig): The config of error evaluation.

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
    logger.info("datamodule setup: done.")

    # Setup model
    arch = instantiate(cfg.arch, num_classes=datamodule.num_classes)
    arch.load_state_dict(torch.load(weightpath))
    arch = arch.to(device)
    arch.eval()
    logger.info("architecture setup: done.")

    datamodule.setup("test")
    loader = datamodule.test_dataloader(cfg.num_samples)

    error_dict: Dict[float] = dict()
    with tqdm(
        zip(cfg.topk, fhmap.eval_mean_errors(arch, loader, device, cfg.topk)),
        ncols=80,
    ) as pbar:
        for k, mean_err in pbar:
            error_dict[k] = mean_err

            # show result to pbar
            logger.info(f"err{k} [%]: {mean_err:0.3}")
            pbar.update()


if __name__ == "__main__":
    import os

    logger.info(os.system("nvidia-smi"))
    eval_error()
