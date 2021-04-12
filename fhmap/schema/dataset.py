from dataclasses import dataclass

from omegaconf import MISSING


@dataclass()
class DatasetConfig:
    _target_: str = MISSING


@dataclass()
class Cifar10Config(DatasetConfig):
    _target_: str = "fhmap.factory.dataset.Cifar10DataModule"


@dataclass()
class Imagenet100Config(DatasetConfig):
    _target_: str = "fhmap.factory.dataset.Imagenet100DataModule"


@dataclass()
class ImagenetConfig(DatasetConfig):
    _target_: str = "fhmap.factory.dataset.ImagenetDataModule"
