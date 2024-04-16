from typing import *

import math
import axon as ax


class Primitive:
    args: Tuple[ax.Tensor, ...]

    def __init__(self, args: Tuple[ax.Tensor, ...]):
        self.args = args

    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        """
        Performs an adjoint trace through the primitive
        :param adjoints: adjoints of output tensors
        :param outputs: outputs of primitive
        :param argnums: which args of the primitive require their adjoint (by default None means all args)
        """
        raise NotImplementedError(f"{type(self).__name__}.backward(...) not implemented")

    def __str__(self) -> str:
        return type(self).__name__


class LoadPrimitive(Primitive):
    def __init__(self):
        super().__init__(tuple())

    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        raise RuntimeError("Load Primitives should never have backwards called on them")


class UnaryPrimitive(Primitive):
    def __init__(self, arg: ax.Tensor):
        super().__init__((arg,))


class Cast(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, dtype: ax.DType):
        super().__init__(arg)
        self.dtype = dtype

    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0].cast(self.args[0].dtype),)

    def __str__(self) -> str:
        return f"Cast<{self.dtype}>"


class Reshape(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, shape: Tuple[int, ...]):
        super().__init__(arg)
        self.shape = shape

    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (ax.reshape(adjoints[0], self.args[0].shape),)

    def __str__(self):
        return f"Reshape<{self.shape}>"


class Broadcast(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, shape: Tuple[int, ...], semantics=ax.utils.BroadcastSemantics.Elementwise):
        super().__init__(arg)
        self.shape = shape
        self.semantics = semantics

    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        # deal with expanded dims first
        sum_axes = list(range(0, len(adjoints[0].shape) - len(self.args[0].shape)))
        # now add any dims in adjoint_shape that were 1 in original shape but expanded
        for i in range(-len(self.args[0].shape), 0):
            if self.args[0].shape[i] == 1 and adjoints[0].shape[i] > 1:
                sum_axes.append(len(adjoints[0].shape) - i)
            if i >= -2 and self.semantics == ax.utils.BroadcastSemantics.MatMul:
                # ignore last two dimensions on matmul
                break

        return (ax.reshape(ax.reduce_sum(adjoints[0], tuple(sum_axes)), self.args[0].shape),)

    def __str__(self):
        sem_str = 'E' if self.semantics == ax.utils.BroadcastSemantics.Elementwise else 'M'
        return f"Broadcast<{self.shape}, {sem_str}>"


class StopGradient(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (None,)


class CustomGradient(Primitive):
    def __init__(self, args: Tuple[ax.Tensor, ...],
                 grad_fn=Callable[[List[ax.Tensor], Optional[Tuple[int, ...]]], Tuple[Optional[ax.Tensor], ...]],
                 **kwargs):
        super().__init__(args)
        self.grad_fn = grad_fn
        self.kwargs = kwargs

    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        grads = self.grad_fn(adjoints, argnums, **self.kwargs)
        for arg, grad in zip(self.args, grads):
            assert grad is None or (arg.shape == grad.shape and arg.dtype == grad.dtype), \
                "Custom gradient function must output grads with same shape and dtype as original args"
        return tuple(grads)


class PermuteDims(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, dims: Tuple[int, ...]):
        super().__init__(arg)
        self.dims = dims

    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        reverse_dims = [0] * len(self.dims)
        for i, dim in enumerate(self.dims):
            reverse_dims[dim] = i
        return (adjoints[0].permute_dims(tuple(reverse_dims)),)

    def __str__(self):
        return f"PermuteDims<{self.dims}>"


class Negate(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (-adjoints[0],)


class ReductionPrimitive(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, axes: Tuple[int, ...]):
        super().__init__(arg)
        self.axes = axes

    def __str__(self) -> str:
        return f"{type(self).__name__}<{self.axes}>"


class Sum(ReductionPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (ax.broadcast(adjoints[0], self.args[0].shape),)

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
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return adjoints[0], adjoints[0]


class Subtract(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return adjoints[0], -adjoints[0]


class Multiply(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        lhs, rhs = self.args
        lhs_adjoint = adjoints[0] * rhs
        rhs_adjoint = adjoints[0] * lhs
        return lhs_adjoint, rhs_adjoint


class Divide(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        lhs, rhs = self.args
        lhs_adjoint = adjoints[0] / rhs
        rhs_adjoint = -adjoints[0] * lhs / (rhs * rhs)
        return lhs_adjoint, rhs_adjoint


class MatMul(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        lhs, rhs = self.args
        lhs_adjoint = ax.matmul(adjoints[0], rhs.mT())
        rhs_adjoint = ax.matmul(lhs.mT(), adjoints[0])
        return lhs_adjoint, rhs_adjoint


class Maximum(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        lhs, rhs = self.args
        lhs_mask = ax.greater_or_equal(lhs, rhs)
        rhs_mask = ax.logical_not(lhs_mask)
        lhs_adjoint = ax.mask(adjoints[0], lhs_mask)
        rhs_adjoint = ax.mask(adjoints[0], rhs_mask)
        return lhs_adjoint, rhs_adjoint


class Minimum(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        lhs, rhs = self.args
        lhs_mask = ax.lesser_or_equal(lhs, rhs)
        rhs_mask = ax.logical_not(lhs_mask)
        lhs_adjoint = ax.mask(adjoints[0], lhs_mask)
        rhs_adjoint = ax.mask(adjoints[0], rhs_mask)
        return lhs_adjoint, rhs_adjoint


class Greater(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class Lesser(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class Equal(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class GreaterOrEqual(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class LesserOrEqual(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class LogicalNot(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class LogicalAnd(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


class LogicalOr(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return tuple(map(lambda a: ax.zeros_like(a), self.args))


def _slice_on_axis(axis, dims, offset, length) -> Tuple[slice | int, ...]:
    out = [slice(None, None, None) for _ in range(dims)]
    out[axis] = offset if length == 1 else slice(offset, offset + length, 1)
    return tuple(out)


class Concatenate(Primitive):
    def __init__(self, args: Tuple[ax.Tensor, ...], axis: int):
        super().__init__(args)
        self.axis = axis

    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        offset = 0

        indices = []
        # skip last one since it is implied by split index
        for arg in self.args[:-1]:
            offset += arg.shape[self.axis]
            indices.append(offset)

        return ax.split(adjoints[0], indices, self.axis)

    def __str__(self):
        return f"Concatenate<{self.axis}>"


class Split(UnaryPrimitive):
    def __init__(self, arg: ax.Tensor, indices_or_sections: Union[int, Tuple[int, ...]], axis: int):
        super().__init__(arg)
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def backward(self, adjoints: Sequence[ax.Tensor], outputs: [ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (ax.concat(adjoints, self.axis),)

    def __str__(self):
        return f"Split<{self.indices_or_sections}, {self.axis}>"


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
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] * self.args[0].cos(),)


class ArcSin(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] / (1 - self.args[0] ** 2).sqrt(),)


class Sinh(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] * self.args[0].cosh(),)


class ArcSinh(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] / (1 + self.args[0] ** 2).sqrt(),)


class Cos(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (-adjoints[0] * self.args[0].sin(),)


class ArcCos(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (-adjoints[0] / (1 - self.args[0] ** 2).sqrt(),)


class Cosh(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] * self.args[0].sinh(),)


class ArcCosh(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] / (self.args[0] ** 2 - 1).sqrt(),)


class Tan(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] / self.args[0].cos() ** 2,)


class ArcTan(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] / (1 + self.args[0] ** 2),)


class Tanh(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] * (1 - self.args[0].tanh() ** 2),)


class ArcTanh(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] / (1 - self.args[0] ** 2),)


class Log(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] / self.args[0],)


class Power(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        lhs, rhs = self.args
        lhs_adjoint = adjoints[0] * rhs * lhs ** (rhs - 1)
        rhs_adjoint = adjoints[0] * lhs ** rhs * ax.log(lhs)
        return lhs_adjoint, rhs_adjoint


class Sqrt(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] / (2 * self.args[0].sqrt()),)


class Sigmoid(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (outputs[0] * (1 - outputs[0]),)


class Softmax(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return ((outputs[0] * (ax.reduce_sum(outputs[0] * adjoints[0], -1))),)


class Exp(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        return (adjoints[0] * outputs[0],)


_M_2_SQRTPI = 2.0 / math.sqrt(math.pi)


class Erf(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        scale = ax.scalar(1.0 / _M_2_SQRTPI, outputs[0].dtype) * adjoints[0]
        return scale * ax.exp(-ax.power(self.args[0], ax.scalar(2.0, outputs[0].dtype)))


class ErfInv(UnaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        scale = ax.scalar(1.0 / _M_2_SQRTPI, outputs[0].dtype) * adjoints[0]
        return scale * ax.exp(ax.power(ax.erfinv(self.args[0]), ax.scalar(2.0, outputs[0].dtype)))


class Mask(BinaryPrimitive):
    def backward(self, adjoints: List[ax.Tensor], outputs: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) \
            -> Tuple[Optional[ax.Tensor], ...]:
        lhs, rhs = self.args
        lhs_adjoint = ax.mask(adjoints[0], rhs)
        rhs_adjoint = ax.zeros_like(rhs)
        return lhs_adjoint, rhs_adjoint


class RandomBits(LoadPrimitive):
    pass
