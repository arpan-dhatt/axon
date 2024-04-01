from typing import *

import axon as ax


class Primitive:
    args: Tuple[ax.Tensor, ...]

    def __init__(self, args: Tuple[ax.Tensor, ...]):
        self.args = args

    def backward(self, adjoint: ax.Tensor, argnums: Tuple[int, ...]) -> Tuple[ax.Tensor, ...]:
        """
        Performs an adjoint trace through the primitive
        :param adjoint: adjoint of output tensor
        :param argnums: which args of the primitive require their adjoint
        """
        raise NotImplementedError(f"{type(self).__name__}.backward(...) not implemented")

    def __str__(self) -> str:
        return type(self).__name__


class UnaryPrimitive(Primitive):
    def __init__(self, arg: ax.Tensor):
        super().__init__((arg,))


class Cast(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, dtype: ax.DType):
        super().__init__(arg)
        self.dtype = dtype

    def __str__(self) -> str:
        return f"Cast<{self.dtype}>"


class Reshape(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, shape: Tuple[int, ...]):
        super().__init__(arg)
        self.shape = shape

    def __str__(self):
        return f"Reshape<{self.shape}>"


class Broadcast(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, shape: Tuple[int, ...]):
        super().__init__(arg)
        self.shape = shape

    def __str__(self):
        return f"Broadcast<{self.shape}>"


class PermuteDims(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, dims: Tuple[int, ...]):
        super().__init__(arg)
        self.dims = dims

    def __str__(self):
        return f"PermuteDims<{self.dims}>"


class Negate(UnaryPrimitive):
    pass


class ReductionPrimitive(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, axes: Tuple[int, ...]):
        super().__init__(arg)
        self.axes = axes

    def __str__(self) -> str:
        return f"{type(self).__name__}<{self.axes}>"


class Sum(ReductionPrimitive):
    pass


class Product(ReductionPrimitive):
    pass


class Max(ReductionPrimitive):
    pass


class Min(ReductionPrimitive):
    pass


class BinaryPrimitive(Primitive):
    def __init__(self, lhs: ax.Tensor, rhs: ax.Tensor):
        super().__init__((lhs, rhs))


class Add(BinaryPrimitive):
    pass


class Subtract(BinaryPrimitive):
    pass


class Multiply(BinaryPrimitive):
    pass


class Divide(BinaryPrimitive):
    pass


class Maximum(BinaryPrimitive):
    pass


class Minimum(BinaryPrimitive):
    pass


class MatMul(BinaryPrimitive):
    pass


class Concatenate(Primitive):
    def __init__(self, args: Tuple[ax.Tensor, ...], axis: int):
        super().__init__(args)
        self.axis = axis

    def __str__(self):
        return f"Concatenate<{self.axis}>"


class Slice(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, indices: Tuple[Union[slice, int], ...]):
        super().__init__(arg)
        self.indices = indices

    def __str__(self):
        def format_slice(s, length):
            start, stop, step = s.indices(length)
            start = start if start != 0 else ""
            stop = stop if stop != length else ""
            step = step if step != 1 else ""
            return f"{start}:{stop}:{step}"

        formatted_indices = []
        for i, index in enumerate(self.indices):
            if isinstance(index, slice):
                formatted_indices.append(format_slice(index, self.args[0].shape[i]))
            else:
                formatted_indices.append(str(index))

        return f"Slice<{', '.join(formatted_indices)}>"
