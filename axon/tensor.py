from typing import *

import axon as ax

from axon.dtype import DType


class Tensor:
    shape: Tuple[int, ...]
    dtype: DType
    data: Any

    prim: Optional['ax.Primitive']  # type: ignore

    tracer: bool = False

    def __init__(self, shape: Tuple[int, ...], dtype: DType, data=None, prim=None, tracer=False):
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.prim = prim
        self.tracer = tracer

    @staticmethod
    def scalar(value: Union[int, float, bool], dtype: DType = None) -> 'Tensor':
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

    @staticmethod
    def zeros_like(arg: 'Tensor') -> 'Tensor':
        return ax.Tensor(arg.shape, arg.dtype)

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

    def __gt__(self, other):
        return ax.greater(self, other)

    def __lt__(self, other):
        return ax.lesser(self, other)

    def __eq__(self, other):
        return ax.equal(self, other)

    def __ge__(self, other):
        return ax.greater_or_equal(self, other)

    def __le__(self, other):
        return ax.lesser_or_equal(self, other)

    def __and__(self, other):
        return ax.logical_and(self, other)

    def __or__(self, other):
        return ax.logical_or(self, other)

    def __invert__(self):
        return ax.logical_not(self)

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

    def mT(self) -> 'Tensor':
        return ax.matrix_transpose(self)

    def expand_dims(self, axis: int = 0) -> 'Tensor':
        return ax.expand_dims(self, axis)

    def flatten(self):
        return self.reshape((-1,))

    def squeeze(self):
        return ax.squeeze(self)

    def set_trace(self) -> 'Tensor':
        self.tracer = True
        return self

    def unset_trace(self) -> 'Tensor':
        self.tracer = False
        return self

    def __getitem__(self, indices):
        return ax.array_slice(self, indices)

    def __hash__(self):
        return super().__hash__()
