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
        lhs_out = lhs if lhs.shape == shape else ax.Tensor(shape, lhs.dtype, prim=prims.Broadcast(lhs, shape))
        rhs_out = rhs if rhs.shape == shape else ax.Tensor(shape, rhs.dtype, prim=prims.Broadcast(rhs, shape))
    else:  # BroadcastSemantics.MatMul
        lhs_shape, rhs_shape = shape[:-2] + lhs.shape[-2:], shape[:-2] + rhs.shape[-2:]
        lhs_out = lhs if lhs.shape == lhs_shape else ax.Tensor(lhs_shape, lhs.dtype,
                                                               prim=prims.Broadcast(lhs, lhs_shape))
        rhs_out = rhs if rhs.shape == rhs_shape else ax.Tensor(rhs_shape, rhs.dtype,
                                                               prim=prims.Broadcast(rhs, rhs_shape))
    return lhs_out, rhs_out


def cast(arg: ax.Tensor, dtype: DType) -> ax.Tensor:
    return ax.Tensor(arg.shape, dtype, prim=prims.Cast(arg, dtype))


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
    return ax.Tensor(shape, arg.dtype, prim=prims.Reshape(arg, shape))


def permute_dims(arg: ax.Tensor, dims: Tuple[int, ...]) -> ax.Tensor:
    new_shape = []
    for dim in dims:
        assert 0 <= dim < len(arg.shape), f"dim {dim} in permute_dims out of bounds"
        new_shape.append(arg.shape[dim])
    new_shape = tuple(new_shape)
    return ax.Tensor(new_shape, arg.dtype, prim=prims.PermuteDims(arg, dims))


def matrix_transpose(arg: ax.Tensor) -> ax.Tensor:
    assert len(arg.shape) >= 2, "Cannot matrix transpose array with less than 2 dimensions"
    permutation = list(range(len(arg.shape)))
    permutation[-1], permutation[-2] = permutation[-2], permutation[-1]
    return permute_dims(arg, tuple(permutation))


def add(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.add requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Add(lhs, rhs))


def subtract(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.subtract requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Subtract(lhs, rhs))


def multiply(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, \
        f"axon.multiply requires both elements have the same dtype ({lhs.dtype} != {rhs.dtype})"
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Multiply(lhs, rhs))


def divide(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.divide requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Divide(lhs, rhs))


def negate(arg: ax.Tensor) -> ax.Tensor:
    return ax.Tensor(arg.shape, arg.dtype, prim=prims.Negate(arg))


def reduce_sum(arg: ax.Tensor, axes: Union[int, Tuple[int, ...], None] = None) -> ax.Tensor:
    axes = utils.reformat_reduce_axes(arg.shape, axes)
    ndim = len(arg.shape)
    for axis in axes:
        assert 0 <= axis < ndim, f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
    new_shape = tuple(1 if i in axes else dim for i, dim in enumerate(arg.shape))
    return ax.Tensor(new_shape, arg.dtype, prim=prims.Sum(arg, axes))


def product(arg: ax.Tensor, axes: Union[int, Tuple[int, ...], None] = None) -> ax.Tensor:
    axes = utils.reformat_reduce_axes(arg.shape, axes)
    ndim = len(arg.shape)
    for axis in axes:
        assert 0 <= axis < ndim, f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
    new_shape = tuple(1 if i in axes else dim for i, dim in enumerate(arg.shape))
    return ax.Tensor(new_shape, arg.dtype, prim=prims.Product(arg, axes))


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
    return ax.Tensor(new_shape, arg.dtype, prim=prims.Max(arg, axes))


def reduce_min(arg: ax.Tensor, axes: Union[int, Tuple[int, ...], None]) -> ax.Tensor:
    axes = utils.reformat_reduce_axes(arg.shape, axes)
    ndim = len(arg.shape)
    for axis in axes:
        assert 0 <= axis < ndim, f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
    new_shape = tuple(1 if i in axes else dim for i, dim in enumerate(arg.shape))
    return ax.Tensor(new_shape, arg.dtype, prim=prims.Min(arg, axes))


def maximum(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.maximum requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Maximum(lhs, rhs))


def minimum(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.Elementwise)
    assert lhs.dtype == rhs.dtype, (f"axon.minimum requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(lhs.shape, lhs.dtype, prim=prims.Minimum(lhs, rhs))


def matmul(lhs: ax.Tensor, rhs: ax.Tensor) -> ax.Tensor:
    lhs, rhs = broadcast(lhs, rhs, semantics=utils.BroadcastSemantics.MatMul)
    assert lhs.dtype == rhs.dtype, (f"axon.matmul requires both elements have the same dtype "
                                    f"({lhs.dtype} != {rhs.dtype})")
    return ax.Tensor(utils.broadcast_shapes(lhs.shape, rhs.shape, semantics=utils.BroadcastSemantics.MatMul),
                     lhs.dtype, prim=prims.MatMul(lhs, rhs))


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
    return ax.Tensor(tuple(output_shape), dtype=args[0].dtype, prim=prims.Concatenate(args, axis))


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

    return ax.Tensor(tuple(output_shape), arg.dtype, prim=prims.Slice(arg, tuple(normalized_indices)))


def expand_dims(arg: ax.Tensor, axis: int = 0) -> ax.Tensor:
    shape = list(arg.shape)
    shape.insert(axis, 1)
    return reshape(arg, tuple(shape))


def stack(args: Tuple[ax.Tensor, ...], axis: int = 0) -> ax.Tensor:
    expanded = map(lambda t: expand_dims(t, axis=axis), args)
    return concat(tuple(expanded), axis=axis)


def flatten(arg: ax.Tensor) -> ax.Tensor:
    return reshape(arg, (-1,))


def print_graph(tensors: List[ax.Tensor]):
    name_counter = [0]
    visited = {}

    def traverse(cursor: ax.Tensor):
        if cursor in visited: return

        # post-order traversal
        if cursor.prim is not None:
            for arg in cursor.prim.args:
                traverse(arg)

        # print cursor
        name = name_counter[0]
        name_counter[0] += 1
        visited[cursor] = name
        print(f"%{name}:<{cursor.shape}, {cursor.dtype.name}> = ", end="")
        if cursor.prim is not None:
            print(f"{str(cursor.prim)}(", end="")
            print(", ".join(map(lambda t: f"%{visited[t]}", cursor.prim.args)), end="")
            print(")")
        else:
            print(cursor.data)

    for tensor in tensors:
        traverse(tensor)
