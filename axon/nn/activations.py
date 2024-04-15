from typing import *

import axon as ax


def sigmoid(x: ax.Tensor) -> ax.Tensor:
    return ax.sigmoid(x)


def silu(x: ax.Tensor) -> ax.Tensor:
    return x * ax.sigmoid(x)