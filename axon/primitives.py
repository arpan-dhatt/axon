from typing import *

import axon as ax


class Primitive:
    args: Tuple[ax.Tensor, ...]

    def __init__(self, args: Tuple[ax.Tensor, ...]):
        self.args = args

    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        """
        Performs an adjoint trace through the primitive
        :param adjoint: adjoint of output tensor
        :param argnums: which args of the primitive require their adjoint (by default None means all args)
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

    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (ax.reshape(adjoint, self.args[0].shape),)

    def __str__(self):
        return f"Reshape<{self.shape}>"


class Broadcast(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, shape: Tuple[int, ...]):
        super().__init__(arg)
        self.shape = shape

    def __str__(self):
        return f"Broadcast<{self.shape}>"


class StopGradient(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor):
        super().__init__(arg)

    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (ax.zeros_like(self.args[0]),)


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
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return adjoint, adjoint


class Subtract(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return adjoint, -adjoint


class Multiply(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        lhs, rhs = self.args
        lhs_adjoint = adjoint * rhs
        rhs_adjoint = adjoint * lhs
        return lhs_adjoint, rhs_adjoint


class Divide(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        lhs, rhs = self.args
        lhs_adjoint = adjoint / rhs
        rhs_adjoint = -adjoint * lhs / (rhs * rhs)
        return lhs_adjoint, rhs_adjoint


class MatMul(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        lhs, rhs = self.args
        lhs_adjoint = ax.matmul(adjoint, rhs.mT())
        rhs_adjoint = ax.matmul(lhs.mT(), adjoint)
        return lhs_adjoint, rhs_adjoint


class Maximum(BinaryPrimitive):
    pass


class Minimum(BinaryPrimitive):
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
