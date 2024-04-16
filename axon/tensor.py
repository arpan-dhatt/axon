from typing import *

import axon as ax

from axon.dtype import DType


class Tensor:
    shape: Tuple[int, ...]
    dtype: DType
    data: Any

    prim: Optional['ax.Primitive']
    # denotes which of N outputs a primitive has since it can have multiple
    siblings: List['Tensor']
    sibling_ix: int

    tracer: bool = False

    def __init__(self, shape: Tuple[int, ...], dtype: DType, data=None, prim: Optional['ax.Primitive'] = None,
                 siblings=None, sibling_ix: int = -1, tracer: bool = False):
        if siblings is None:
            siblings = []
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.prim = prim
        self.siblings = siblings
        self.sibling_ix = sibling_ix
        self.tracer = tracer

    def __add__(self, rhs):
        return ax.add(self, rhs)

    def __radd__(self, lhs):
        return ax.add(lhs, self)

    def __sub__(self, rhs):
        return ax.subtract(self, rhs)

    def __rsub__(self, lhs):
        return ax.subtract(lhs, self)

    def __mul__(self, rhs):
        return ax.multiply(self, rhs)

    def __rmul__(self, lhs):
        return ax.multiply(lhs, self)

    def __truediv__(self, rhs):
        return ax.divide(self, rhs)

    def __rtruediv__(self, lhs):
        return ax.divide(lhs, self)

    def __neg__(self):
        return ax.negate(self)

    def __matmul__(self, rhs):
        return ax.matmul(self, rhs)

    def __rmatmul__(self, lhs):
        return ax.matmul(lhs, self)

    def __gt__(self, rhs):
        return ax.greater(self, rhs)

    def __lt__(self, rhs):
        return ax.lesser(self, rhs)

    def __eq__(self, rhs):
        return ax.equal(self, rhs)

    def __ge__(self, rhs):
        return ax.greater_or_equal(self, rhs)

    def __le__(self, rhs):
        return ax.lesser_or_equal(self, rhs)

    def __and__(self, rhs):
        return ax.logical_and(self, rhs)

    def __rand__(self, lhs):
        return ax.logical_and(lhs, self)

    def __or__(self, rhs):
        return ax.logical_or(self, rhs)

    def __ror__(self, lhs):
        return ax.logical_or(lhs, self)

    def __invert__(self):
        return ax.logical_not(self)

    def __pow__(self, rhs):
        return ax.power(self, rhs)

    def __rpow__(self, lhs):
        return ax.power(lhs, self)

    def sqrt(self) -> 'Tensor':
        return ax.sqrt(self)

    def cast(self, dtype: DType) -> 'Tensor':
        return ax.cast(self, dtype)

    def reduce_sum(self, axes: Union[int, Sequence[int], None] = None) -> 'Tensor':
        return ax.reduce_sum(self, axes)

    def product(self, axes: Union[int, Sequence[int], None] = None) -> 'Tensor':
        return ax.product(self, axes)

    def mean(self, axes: Union[int, Sequence[int], None] = None) -> 'Tensor':
        return ax.mean(self, axes)

    def reduce_max(self, axes: Union[int, Sequence[int], None] = None) -> 'Tensor':
        return ax.reduce_max(self, axes)

    def reduce_min(self, axes: Union[int, Sequence[int], None] = None) -> 'Tensor':
        return ax.reduce_min(self, axes)

    def maximum(self, rhs: 'Tensor') -> 'Tensor':
        return ax.maximum(self, rhs)

    def minimum(self, rhs: 'Tensor') -> 'Tensor':
        return ax.minimum(self, rhs)

    def reshape(self, shape: Sequence[int]) -> 'Tensor':
        return ax.reshape(self, shape)

    def permute_dims(self, dims: Sequence[int]) -> 'Tensor':
        return ax.permute_dims(self, dims)

    def mT(self) -> 'Tensor':
        return ax.matrix_transpose(self)

    def expand_dims(self, axis: int = 0) -> 'Tensor':
        return ax.expand_dims(self, axis)

    def flatten(self):
        return self.reshape((-1,))

    def squeeze(self):
        return ax.squeeze(self)

    def sin(self) -> 'Tensor':
        return ax.sin(self)

    def arcsin(self) -> 'Tensor':
        return ax.arcsin(self)

    def sinh(self) -> 'Tensor':
        return ax.sinh(self)

    def arcsinh(self) -> 'Tensor':
        return ax.arcsinh(self)

    def cos(self) -> 'Tensor':
        return ax.cos(self)

    def arccos(self) -> 'Tensor':
        return ax.arccos(self)

    def cosh(self) -> 'Tensor':
        return ax.cosh(self)

    def arccosh(self) -> 'Tensor':
        return ax.arccosh(self)

    def tan(self) -> 'Tensor':
        return ax.tan(self)

    def arctan(self) -> 'Tensor':
        return ax.arctan(self)

    def tanh(self) -> 'Tensor':
        return ax.tanh(self)

    def arctanh(self) -> 'Tensor':
        return ax.arctanh(self)

    def log(self) -> 'Tensor':
        return ax.log(self)

    def mask(self, rhs: 'Tensor') -> 'Tensor':
        return ax.mask(self, rhs)

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

    def __repr__(self):
        return f"axon.Tensor({self.shape}, {self.dtype}, {self.data})"

    def __str__(self):
        self.eval()
        return str(self.data)

    def eval(self, backend: Optional['ax.Backend'] = None) -> Any:
        ax.eval(self, backend)
