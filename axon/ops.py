from typing import *
import functools

import axon as ax
import axon.primitives as prims
import axon.utils as utils

from axon.dtype import DType


def broadcast(lhs: ax.Tensor, rhs: ax.Tensor, semantics=utils.BroadcastSemantics.Elementwise) \
        -> Tuple[ax.Tensor, ax.Tensor]:
    shape = utils.broadcast_shapes(lhs.shape, rhs.shape, semantics)
    if semantics == utils.BroadcastSemantics.Elementwise:
        lhs_out = lhs if lhs.shape == shape else ax.Tensor(shape, lhs.dtype,
                                                           prim=prims.Broadcast(lhs, shape),
                                                           tracer=lhs.tracer)
        rhs_out = rhs if rhs.shape == shape else ax.Tensor(shape, rhs.dtype,
                                                           prim=prims.Broadcast(rhs, shape),
                                                           tracer=rhs.tracer)
    else:  # BroadcastSemantics.MatMul
        lhs_shape, rhs_shape = shape[:-2] + lhs.shape[-2:], shape[:-2] + rhs.shape[-2:]
        lhs_replacement = ax.Tensor(lhs_shape, lhs.dtype,
                                    prim=prims.Broadcast(lhs, lhs_shape,
                                                         semantics=utils.BroadcastSemantics.MatMul),
                                    tracer=lhs.tracer)
        lhs_out = lhs if lhs.shape == lhs_shape else lhs_replacement
        rhs_replacement = ax.Tensor(rhs_shape, rhs.dtype,
                                    prim=prims.Broadcast(rhs, rhs_shape,
                                                         semantics=utils.BroadcastSemantics.MatMul),
                                    tracer=rhs.tracer)
        rhs_out = rhs if rhs.shape == rhs_shape else rhs_replacement
    return lhs_out, rhs_out


def cast(arg: ax.Tensor, dtype: DType) -> ax.Tensor:
    return ax.Tensor(arg.shape, dtype, prim=prims.Cast(arg, dtype), tracer=arg.tracer)


def stop_gradient(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.StopGradient(arg), tracer=arg.tracer)


def reshape(arg: ax.Tensor, shape: Union[int, Tuple[int, ...]]) -> ax.Tensor:
    if isinstance(shape, int):
        shape = (shape,)

    # handle a single free dimension (-1)
    free_dim = None
    prod = 1
    for dim, length in enumerate(shape):
        if length == -1:
            if free_dim is None:
                free_dim = dim
            else:
                raise ValueError("Reshape can only receive at most one free-form (-1) dimension")
        else:
            prod *= length

    # fill in free dimension with concrete length if possible
    if free_dim is not None:
        assert utils.shaped_size(arg.shape) % prod == 0, "Size across free dim must be a factor of the original size"
        shape = list(shape)
        shape[free_dim] = utils.shaped_size(arg.shape) // prod
        shape = tuple(shape)

    assert utils.shaped_size(arg.shape) == utils.shaped_size(shape), f"Can't reshape {arg.shape} to {shape}"
    return ax.Tensor(shape, arg.dtype, prim=prims.Reshape(arg, shape), tracer=arg.tracer)


def permute_dims(arg: ax.Tensor, dims: Tuple[int, ...]) -> ax.Tensor:
    new_shape = []
    for dim in dims:
        assert 0 <= dim < len(arg.shape), f"dim {dim} in permute_dims out of bounds"
        new_shape.append(arg.shape[dim])
    new_shape = tuple(new_shape)
    return ax.Tensor(new_shape, arg.dtype, prim=prims.PermuteDims(arg, dims), tracer=arg.tracer)


def matrix_transpose(arg: ax.Tensor) -> ax.Tensor:
    assert len(arg.shape) >= 2, "Cannot matrix transpose array with less than 2 dimensions"
    permutation = list(range(len(arg.shape)))
    permutation[-1], permutation[-2] = permutation[-2], permutation[-1]
    return permute_dims(arg, tuple(permutation))


def add(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.add requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Add(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def subtract(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.subtract requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Subtract(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def multiply(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.multiply requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Multiply(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def divide(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.divide requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Divide(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def negate(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Negate(arg), tracer=arg.tracer)


def reduce_sum(arg: ax.Tensor, axes: Union[int, Tuple[int, ...], None] = None) -> ax.Tensor:
    axes = utils.reformat_reduce_axes(arg.shape, axes)
    ndim = len(arg.shape)
    for axis in axes:
        assert 0 <= axis < ndim, f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
    new_shape = tuple(1 if i in axes else dim for i, dim in enumerate(arg.shape))
    return ax.Tensor(new_shape, arg.dtype, prim=prims.Sum(arg, axes), tracer=arg.tracer)


def product(arg: ax.Tensor, axes: Union[int, Tuple[int, ...], None] = None) -> ax.Tensor:
    axes = utils.reformat_reduce_axes(arg.shape, axes)
    ndim = len(arg.shape)
    for axis in axes:
        assert 0 <= axis < ndim, f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
    new_shape = tuple(1 if i in axes else dim for i, dim in enumerate(arg.shape))
    return ax.Tensor(new_shape, arg.dtype, prim=prims.Product(arg, axes), tracer=arg.tracer)


def mean(arg: ax.Tensor, axes: Union[int, Tuple[int, ...], None] = None) -> ax.Tensor:
    axes = utils.reformat_reduce_axes(arg.shape, axes)
    summed = reduce_sum(arg, axes)
    return summed / ax.Tensor.scalar(
        utils.shaped_size([l for dim, l in enumerate(arg.shape) if (dim in axes)]), arg.dtype)


def reduce_max(arg: ax.Tensor, axes: Union[int, Tuple[int, ...], None] = None) -> ax.Tensor:
    axes = utils.reformat_reduce_axes(arg.shape, axes)
    ndim = len(arg.shape)
    for axis in axes:
        assert 0 <= axis < ndim, f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
    new_shape = tuple(1 if i in axes else dim for i, dim in enumerate(arg.shape))
    return ax.Tensor(new_shape, arg.dtype, prim=prims.Max(arg, axes), tracer=arg.tracer)


def reduce_min(arg: ax.Tensor, axes: Union[int, Tuple[int, ...], None]) -> ax.Tensor:
    axes = utils.reformat_reduce_axes(arg.shape, axes)
    ndim = len(arg.shape)
    for axis in axes:
        assert 0 <= axis < ndim, f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
    new_shape = tuple(1 if i in axes else dim for i, dim in enumerate(arg.shape))
    return ax.Tensor(new_shape, arg.dtype, prim=prims.Min(arg, axes), tracer=arg.tracer)


def maximum(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.maximum requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Maximum(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def minimum(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.minimum requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Minimum(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def greater(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.greater requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, ax.Bool, prim=prims.Greater(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def lesser(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.lesser requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, ax.Bool, prim=prims.Lesser(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def equal(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.equal requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, ax.Bool, prim=prims.Equal(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def greater_or_equal(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.greater_or_equal requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, ax.Bool, prim=prims.GreaterOrEqual(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def lesser_or_equal(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.lesser_or_equal requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, ax.Bool, prim=prims.LesserOrEqual(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def logical_and(lhs: Union[ax.Tensor, bool], rhs: Union[ax.Tensor, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == ax.Bool and rhs.dtype == ax.Bool, (
        f"axon.logical_and requires both elements to be of type ax.Bool "
        f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, ax.Bool, prim=prims.LogicalAnd(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def logical_or(lhs: Union[ax.Tensor, bool], rhs: Union[ax.Tensor, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == ax.Bool and rhs.dtype == ax.Bool, (
        f"axon.logical_or requires both elements to be of type ax.Bool "
        f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, ax.Bool, prim=prims.LogicalOr(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def logical_not(arg: ax.Tensor) -> ax.Tensor:
    assert arg.dtype == ax.Bool, (f"axon.logical_not requires the argument to be of type ax.Bool "
                                  f"({arg.dtype} != ax.Bool)")
    return ax.Tensor(arg.shape, ax.Bool, prim=prims.LogicalNot(arg), tracer=arg.tracer)


def matmul(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.MatMul)
    assert lhs.dtype == rhs.dtype, (f"axon.matmul requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(utils.broadcast_shapes(lhs.shape, rhs.shape, semantics=utils.BroadcastSemantics.MatMul),
                     lhs.dtype, prim=prims.MatMul(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def concat(args: Tuple[ax.Tensor, ...], axis: int) -> ax.Tensor:
    assert len(args) > 0
    # all shapes must be same except on concatenation axis
    assert all(
        map(lambda t: len(t.shape) == len(args[0].shape), args[1:])), "concat requires number of dims to be the same"
    # all dtypes of args must be the same
    assert all(map(lambda t: t.dtype == args[0].dtype, args[1:])), "concat requires all dtypes be the same"

    output_shape = []
    for i in range(len(args[0].shape)):
        output_shape.append(args[0].shape[i])
        for arg in args[1:]:
            # add length along concat axis
            if i == axis:
                output_shape[-1] += arg.shape[i]
            else:
                assert arg.shape[i] == args[0].shape[i], "length along all axes except concat axis must be the same"
    return ax.Tensor(tuple(output_shape), dtype=args[0].dtype, prim=prims.Concatenate(args, axis),
                     tracer=any(map(lambda t: t.tracer, args)))


def array_slice(arg: ax.Tensor, indices: Union[int, slice, Tuple[Union[int, slice], ...]]) -> ax.Tensor:
    # convert everything to tuples
    if isinstance(indices, int):
        indices = (indices,)
    elif isinstance(indices, slice):
        indices = (indices,)

    assert len(indices) <= len(arg.shape)
    normalized_indices = []
    output_shape = []
    for i, length in enumerate(arg.shape):
        if i < len(indices):
            index = indices[i]
            if isinstance(index, slice):
                start = index.start if index.start is not None else 0
                if start < 0:
                    start = length + start
                stop = index.stop if index.stop is not None else length
                if stop < 0:
                    stop = length + stop
                step = index.step if index.step is not None else 1
                assert 0 <= start < length, "start index out of range"
                assert 0 < stop <= length, "stop index out of range"
                assert stop > start, "Slice stop index must be greater than start"
                # funky calculation basically just offset x in f(x)
                output_shape.append(((stop - start) + (step - 1)) // step)
                normalized_indices.append(
                    slice(start, stop, step))
            if isinstance(index, int):
                index = length + index if index < 0 else index
                assert 0 <= index < length, "index out of range"
                output_shape.append(1)
                normalized_indices.append(index)
        else:
            output_shape.append(length)

    return ax.Tensor(tuple(output_shape), arg.dtype, prim=prims.Slice(arg, tuple(normalized_indices)),
                     tracer=arg.tracer)


def expand_dims(arg: ax.Tensor, axis: int = 0) -> ax.Tensor:
    shape = list(arg.shape)
    shape.insert(axis, 1)
    return reshape(arg, tuple(shape))


def stack(args: Tuple[ax.Tensor, ...], axis: int = 0) -> ax.Tensor:
    expanded = map(lambda t: expand_dims(t, axis=axis), args)
    return concat(tuple(expanded), axis=axis)


def flatten(arg: ax.Tensor) -> ax.Tensor:
    return reshape(arg, (-1,))


def squeeze(arg: ax.Tensor) -> ax.Tensor:
    new_shape = tuple([dim for dim in arg.shape if dim != 1])
    return reshape(arg, new_shape)


def mask(lhs: ax.Tensor, rhs: Union[ax.Tensor, bool]) -> ax.Tensor:
    rhs = wrap_scalar(rhs, ax.Bool)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert rhs.dtype == ax.Bool, f"The second argument of axon.mask must be of type ax.Bool, but got {rhs.dtype}"
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Mask(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def power(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) -> ax.Tensor:
    lhs, rhs = wrap_scalars_helper(lhs, rhs)
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.power requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Power(lhs, rhs), tracer=any([lhs.tracer, rhs.tracer]))


def sqrt(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Sqrt(arg), tracer=arg.tracer)


def sin(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Sin(arg), tracer=arg.tracer)


def arcsin(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.ArcSin(arg), tracer=arg.tracer)


def sinh(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Sinh(arg), tracer=arg.tracer)


def arcsinh(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.ArcSinh(arg), tracer=arg.tracer)


def cos(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Cos(arg), tracer=arg.tracer)


def arccos(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.ArcCos(arg), tracer=arg.tracer)


def cosh(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Cosh(arg), tracer=arg.tracer)


def arccosh(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.ArcCosh(arg), tracer=arg.tracer)


def tan(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Tan(arg), tracer=arg.tracer)


def arctan(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.ArcTan(arg), tracer=arg.tracer)


def tanh(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Tanh(arg), tracer=arg.tracer)


def arctanh(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.ArcTanh(arg), tracer=arg.tracer)


def log(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Log(arg), tracer=arg.tracer)


def wrap_scalars_helper(lhs: Union[ax.Tensor, int, float, bool], rhs: Union[ax.Tensor, int, float, bool]) \
        -> Tuple[ax.Tensor, ax.Tensor]:
    if isinstance(lhs, ax.Tensor) and isinstance(rhs, ax.Tensor):
        return lhs, rhs
    elif isinstance(lhs, ax.Tensor):
        return lhs, wrap_scalar(rhs, lhs.dtype)
    elif isinstance(rhs, ax.Tensor):
        return wrap_scalar(lhs, rhs.dtype), rhs
    else:
        raise TypeError(f"At least one operand must be an ax.Tensor")


def wrap_scalar(value: Union[ax.Tensor, int, float, bool], intended_dtype: ax.DType) -> ax.Tensor:
    if isinstance(value, (int, float, bool)):
        if isinstance(value, int):
            if intended_dtype in [ax.UInt8, ax.UInt16, ax.UInt32, ax.UInt64, ax.Int8, ax.Int16, ax.Int32, ax.Int64]:
                return ax.scalar(value, dtype=intended_dtype)
            elif intended_dtype in [ax.BFloat16, ax.Float16, ax.Float32, ax.Float64]:
                return ax.scalar(float(value), dtype=intended_dtype)
            elif intended_dtype == ax.Bool:
                return ax.scalar(bool(value), dtype=ax.Bool)
        elif isinstance(value, float):
            if intended_dtype in [ax.Float32, ax.Float64]:
                return ax.scalar(value, dtype=intended_dtype)
        elif isinstance(value, bool):
            if intended_dtype == ax.Bool:
                return ax.scalar(value, dtype=ax.Bool)
        raise TypeError(f"Incompatible data types: {type(value).__name__} and {intended_dtype}")
    elif isinstance(value, ax.Tensor):
        return value
    else:
        raise TypeError(f"Unsupported type for scalar wrapping: {type(value).__name__}")


def print_graph(tree):
    name_counter = [0]
    # keep names in original tree for easier debugging
    tree_names = dict(map(lambda t: (t[1], t[0]), ax.utils.tree_flatten(tree)))
    visited = {}

    def traverse(cursor: ax.Tensor):
        if cursor in visited: return

        # post-order traversal
        if cursor.prim is not None:
            for arg in cursor.prim.args:
                traverse(arg)

        # print cursor using tree or intermediate name
        if cursor in tree_names:
            name = f"@{tree_names[cursor]}"
        else:
            name = f"%{name_counter[0]}"
            name_counter[0] += 1
        visited[cursor] = name
        print(f"{name}:<{cursor.shape}, {cursor.dtype.name}>{'*' if cursor.tracer else ''} = ", end="")
        if cursor.prim is not None:
            print(f"{str(cursor.prim)}(", end="")
            print(", ".join(map(lambda t: f"{visited[t]}", cursor.prim.args)), end="")
            print(")")
        else:
            print(cursor.data)

    for key, tensor in ax.utils.tree_flatten(tree):
        traverse(tensor)
