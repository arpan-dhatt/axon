from typing import *

import axon as ax

from axon.dtype import DType


class Tensor:
    shape: Tuple[int, ...]
    dtype: DType
    data: Any

    prim: Optional['Primitive']  # type: ignore

    def __init__(self, shape: Tuple[int, ...], dtype: DType, data=None, prim=None):
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.prim = prim

    @staticmethod
    def scalar(value: Union[int, float, bool], dtype: DType = None):
        if dtype is None:
            if isinstance(value, int):
                dtype = ax.Int32
            elif isinstance(value, float):
                dtype = ax.Float32
            elif isinstance(value, bool):
                dtype = ax.Bool
            else:
                raise ValueError("Scalar can only be implicitly initialized value int, float, or bool")
        return Tensor((), dtype, data=value)

    def __add__(self, other):
        return ax.add(self, other)

    def __sub__(self, other):
        return ax.subtract(self, other)

    def __mul__(self, other):
        return ax.multiply(self, other)

    def __truediv__(self, other):
        return ax.divide(self, other)

    def __neg__(self):
        return ax.negate(self)

    def __matmul__(self, other):
        return ax.matmul(self, other)

    def cast(self, dtype: DType) -> 'Tensor':
        return ax.cast(self, dtype)

    def reduce_sum(self, axes: Union[int, Tuple[int, ...], None] = None) -> 'Tensor':
        return ax.reduce_sum(self, axes)

    def product(self, axes: Union[int, Tuple[int, ...], None] = None) -> 'Tensor':
        return ax.product(self, axes)

    def mean(self, axes: Union[int, Tuple[int, ...], None] = None) -> 'Tensor':
        return ax.mean(self, axes)

    def reduce_max(self, axes: Union[int, Tuple[int, ...], None] = None) -> 'Tensor':
        return ax.reduce_max(self, axes)

    def reduce_min(self, axes: Union[int, Tuple[int, ...], None] = None) -> 'Tensor':
        return ax.reduce_min(self, axes)

    def maximum(self, other: 'Tensor') -> 'Tensor':
        return ax.maximum(self, other)

    def minimum(self, other: 'Tensor') -> 'Tensor':
        return ax.minimum(self, other)

    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        return ax.reshape(self, shape)

    def permute_dims(self, dims: Tuple[int, ...]) -> 'Tensor':
        return ax.permute_dims(self, dims)

    def mT(self):
        return ax.matrix_transpose(self)
