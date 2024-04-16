import math
from typing import Literal

import axon as ax

Reduction = Literal["none", "mean", "sum"]


def _reduce(loss: ax.Tensor, reduction: Reduction = "none") -> ax.Tensor:
    if reduction == "mean":
        return ax.mean(loss).squeeze()
    elif reduction == "sum":
        return ax.reduce_sum(loss).squeeze()
    elif reduction == "none":
        return loss.squeeze()
    else:
        raise ValueError("Invalid reduction. Must be 'none', 'mean', or 'sum'.")


def mse_loss(
    predictions: ax.Tensor, targets: ax.Tensor, reduction: Reduction = "mean"
) -> ax.Tensor:
    """
    Computes the mean squared error loss.

    Args:
        predictions (array): The predicted values.
        targets (array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

    Returns:
        array: The computed mean squared error loss.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Predictions shape {predictions.shape} does not match "
            f"targets shape {targets.shape}."
        )

    loss = ax.power(predictions - targets, 2.0)
    return _reduce(loss, reduction)
