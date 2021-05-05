from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class EnvConfig:
    device: str = MISSING
    num_nodes: int = MISSING
    num_workers: int = MISSING
    savedir: str = MISSING


@dataclass
class DefaultEnvConfig(EnvConfig):
    device: str = "cuda:0"
    num_nodes: int = 1
    num_workers: int = 8
    savedir: str = "outputs"
