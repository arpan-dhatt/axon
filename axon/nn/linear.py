from typing import *
import math

import axon as ax
from axon.nn.base import Module


class Linear(Module):
    def __init__(self, input_size: int, output_size: int, bias=True):
        super().__init__()
        scale = math.sqrt(1.0 / input_size)
        self.weights = ax.random.uniform(-scale, scale, shape=(input_size, output_size), dtype=ax.Float32)
        if bias:
            self.bias = True
            self.biases = ax.random.uniform(-scale, scale, shape=(output_size,), dtype=ax.Float32)

    def _extra_repr(self) -> str:
        return f"input_dims={self.weights.shape[1]}, output_dims={self.weights.shape[0]}, bias={'biases' in self}"

    def __call__(self, x: ax.Tensor) -> ax.Tensor:
        if self.bias:
            return x @ self.weights + self.biases
        else:
            return x @ self.weights
