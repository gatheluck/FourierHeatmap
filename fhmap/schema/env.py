from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class EnvConfig:
    gpus: int = MISSING
    num_nodes: int = MISSING
    num_workers: int = MISSING
    savedir: str = MISSING


@dataclass
class DefaultEnvConfig(EnvConfig):
    gpus: int = 1
    num_nodes: int = 1
    num_workers: int = 8
    savedir: str = "outputs"
