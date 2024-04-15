from typing import *

import axon as ax


def sigmoid(x: ax.Tensor) -> ax.Tensor:
    return ax.sigmoid(x)


def silu(x: ax.Tensor) -> ax.Tensor:
    return x * ax.sigmoid(x)


def relu(x: ax.Tensor) -> ax.Tensor:
    return ax.maximum(x, 0.0)


def leaky_relu(x: ax.Tensor, alpha: float = 0.01) -> ax.Tensor:
    return ax.maximum(x, alpha * x)


def softmax(x: ax.Tensor) -> ax.Tensor:
    return ax.softmax(x)
