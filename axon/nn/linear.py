from typing import *

import axon as ax
from axon.nn.base import Module


class Linear(Module):
    def __init__(self, input_size: int, output_size: int, bias=True):
        super().__init__()
        self.weights = ax.fill(0.1, (input_size, output_size), ax.Float32)
        if bias:
            self.bias = True
            self.biases = ax.fill(0.0, (output_size,), ax.Float32)

    def _extra_repr(self) -> str:
        return f"input_dims={self.weights.shape[1]}, output_dims={self.weights.shape[0]}, bias={'biases' in self}"

    def __call__(self, x: ax.Tensor) -> ax.Tensor:
        if self.bias:
            return x @ self.weights + self.biases
        else:
            return x @ self.weights
