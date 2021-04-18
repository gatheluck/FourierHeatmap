import pathlib
import random
from typing import Final

import numpy as np
import pytest
import torch

LOGDIR: Final = pathlib.Path("outputs/tests")
CIFAR10_ROOT: Final = pathlib.Path("data/cifar10")
PRETRAONED_WEIGHT = pathlib.Path("tests/weight/cifar10_resnet50.pth")
PRETRAONED_WEIGHT_ADV = pathlib.Path("tests/weight/cifar10_resnet50_adv.pth")
CIFAR10_MEAN: Final = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_STD: Final = [0.24703223, 0.24348513, 0.26158784]
BATCH_SIZE = 16

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


@pytest.fixture
def logdir() -> pathlib.Path:
    return LOGDIR
