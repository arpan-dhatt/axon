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

    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint.cast(self.args[0].dtype),)

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
    def __init__(self, arg: ax.Tensor, shape: Tuple[int, ...], semantics=ax.utils.BroadcastSemantics.Elementwise):
        super().__init__(arg)
        self.shape = shape
        self.semantics = semantics

    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        # deal with expanded dims first
        sum_axes = list(range(0, len(adjoint.shape) - len(self.args[0].shape)))
        # now add any dims in adjoint_shape that were 1 in original shape but expanded
        for i in range(-len(self.args[0].shape), 0):
            if self.args[0].shape[i] == 1 and adjoint.shape[i] > 1:
                sum_axes.append(len(adjoint.shape) - i)
            if i >= -2 and self.semantics == ax.utils.BroadcastSemantics.MatMul:
                # ignore last two dimensions on matmul
                break

        return (ax.reshape(ax.reduce_sum(adjoint, tuple(sum_axes)), self.args[0].shape),)

    def __str__(self):
        sem_str = 'E' if self.semantics == ax.utils.BroadcastSemantics.Elementwise else 'M'
        return f"Broadcast<{self.shape}, {sem_str}>"


class StopGradient(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor):
        super().__init__(arg)

    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (ax.zeros_like(self.args[0]),)


class PermuteDims(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, dims: Tuple[int, ...]):
        super().__init__(arg)
        self.dims = dims

    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        reverse_dims = [0] * len(self.dims)
        for i, dim in enumerate(self.dims):
            reverse_dims[dim] = i
        return (adjoint.permute_dims(tuple(reverse_dims)),)

    def __str__(self):
        return f"PermuteDims<{self.dims}>"


class Negate(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (-adjoint,)


class ReductionPrimitive(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, axes: Tuple[int, ...]):
        super().__init__(arg)
        self.axes = axes

    def __str__(self) -> str:
        return f"{type(self).__name__}<{self.axes}>"


class Sum(ReductionPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (ax.broadcast(self.args[0], adjoint)[1],)

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
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        lhs, rhs = self.args
        lhs_mask = ax.greater_or_equal(lhs, rhs)
        rhs_mask = ax.logical_not(lhs_mask)
        lhs_adjoint = ax.mask(adjoint, lhs_mask)
        rhs_adjoint = ax.mask(adjoint, rhs_mask)
        return lhs_adjoint, rhs_adjoint


class Minimum(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        lhs, rhs = self.args
        lhs_mask = ax.lesser_or_equal(lhs, rhs)
        rhs_mask = ax.logical_not(lhs_mask)
        lhs_adjoint = ax.mask(adjoint, lhs_mask)
        rhs_adjoint = ax.mask(adjoint, rhs_mask)
        return lhs_adjoint, rhs_adjoint


class Greater(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class Lesser(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class Equal(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class GreaterOrEqual(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class LesserOrEqual(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class LogicalNot(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class LogicalAnd(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class LogicalOr(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


def _slice_on_axis(axis, dims, offset, length) -> Tuple[slice | int, ...]:
    out = [slice(None, None, None) for _ in range(dims)]
    out[axis] = offset if length == 1 else slice(offset, offset + length, 1)
    return tuple(out)


class Concatenate(Primitive):
    def __init__(self, args: Tuple[ax.Tensor, ...], axis: int):
        super().__init__(args)
        self.axis = axis

    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        out = []
        offset = 0

        for arg in self.args:
            slice_len = arg.shape[self.axis]
            indices = _slice_on_axis(self.axis, len(adjoint.shape), offset, slice_len)
            out.append(ax.array_slice(adjoint, indices))
            offset += slice_len

        return tuple(out)

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


class Sin(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint * self.args[0].cos(),)


class ArcSin(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint / (1 - self.args[0] ** 2).sqrt(),)


class Sinh(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint * self.args[0].cosh(),)


class ArcSinh(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint / (1 + self.args[0] ** 2).sqrt(),)


class Cos(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (-adjoint * self.args[0].sin(),)


class ArcCos(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (-adjoint / (1 - self.args[0] ** 2).sqrt(),)


class Cosh(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint * self.args[0].sinh(),)


class ArcCosh(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint / (self.args[0] ** 2 - 1).sqrt(),)


class Tan(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint / self.args[0].cos() ** 2,)


class ArcTan(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint / (1 + self.args[0] ** 2),)


class Tanh(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint * (1 - self.args[0].tanh() ** 2),)


class ArcTanh(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint / (1 - self.args[0] ** 2),)


class Log(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint / self.args[0],)


class Power(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        lhs, rhs = self.args
        lhs_adjoint = adjoint * rhs * lhs ** (rhs - 1)
        rhs_adjoint = adjoint * lhs ** rhs * ax.log(lhs)
        return lhs_adjoint, rhs_adjoint


class Sqrt(UnaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        return (adjoint / (2 * self.args[0].sqrt()),)


class Mask(BinaryPrimitive):
    def backward(self, adjoint: ax.Tensor, argnums: Optional[Tuple[int, ...]] = None) -> Tuple[ax.Tensor, ...]:
        lhs, rhs = self.args
        lhs_adjoint = ax.mask(adjoint, rhs)
        rhs_adjoint = ax.zeros_like(rhs)
        return lhs_adjoint, rhs_adjoint
