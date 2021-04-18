import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calc_errors(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]:
    """Calculate top-k errors over output from architecture (model).

    Args:
        output (torch.Tensor): Output tensor from architecture (model).
        target (torch.Tensor): Training target tensor.
        topk (Tuple[int, ...]): Tuple of int which you want to know error.

    Returns:
        List[torch.Tensor]: list of errors.

    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(
            maxk, dim=1
        )  # return the k larget elements. top-k index: size (b, k).
        pred = pred.t()  # (k, b)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        errors = list()
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            wrong_k = batch_size - correct_k
            errors.append(wrong_k.mul_(100.0 / batch_size))

        return errors


def eval_mean_errors(
    arch: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: Tuple[int, ...] = (1,)
) -> List[float]:
    """Evaluate top-k mean errors of the architecture over given dataloader.

    Args:
        arch (nn.Module): An architecture to be evaluated.
        loader (DataLoader): A dataloader.
        device (torch.device): A device used for calculation.
        topk (Tuple[int, ...]): Tuple of int which you want to know error.

    Returns:
        List[float]: list of mean errors.

    """
    arch = arch.to(device)
    err_dict: Dict[int, List[float]] = {k: list() for k in topk}

    for x, t in loader:
        x, t = x.to(device), t.to(device)

        output = arch(x)
        for k, err in zip(topk, calc_errors(output, t, topk=topk)):
            err_dict[k].append(err.item())

    return [sum(err_dict[k]) / len(err_dict[k]) for k in err_dict]
