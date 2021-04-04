import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calc_errors(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]:
    """Calculate top-k errors.

    Args:
        output (torch.Tensor): Output tensor from model.
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
