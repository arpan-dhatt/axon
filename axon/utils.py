from typing import *
from enum import Enum
import functools

from axon.tree_utils import tree_map, tree_flatten, tree_unflatten


class BroadcastSemantics(Enum):
    Elementwise = 0
    MatMul = 1


def broadcast_shapes(lhs: Tuple[int, ...], rhs: Tuple[int, ...], semantics=BroadcastSemantics.Elementwise) -> Tuple[
    int, ...]:
    if semantics == BroadcastSemantics.Elementwise:
        return elementwise_broadcast(lhs, rhs)
    elif semantics == BroadcastSemantics.MatMul:
        return matmul_broadcast(lhs, rhs)
    else:
        raise ValueError(f"Unsupported broadcast semantics: {semantics}")


def elementwise_broadcast(lhs: Tuple[int, ...], rhs: Tuple[int, ...]) -> Tuple[int, ...]:
    len_lhs, len_rhs = len(lhs), len(rhs)
    max_len = max(len_lhs, len_rhs)
    result_shape = []

    for i in range(1, max_len + 1):
        lhs_dim = lhs[-i] if i <= len_lhs else 1
        rhs_dim = rhs[-i] if i <= len_rhs else 1

        if lhs_dim == rhs_dim or lhs_dim == 1 or rhs_dim == 1:
            result_shape.append(max(lhs_dim, rhs_dim))
        else:
            raise ValueError(f"Cannot broadcast shapes {lhs} and {rhs}")

    return tuple(reversed(result_shape))


def matmul_broadcast(lhs: Tuple[int, ...], rhs: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(lhs) < 2 or len(rhs) < 2:
        raise ValueError(f"MatMul broadcast requires arrays with at least 2 dimensions, got shapes {lhs} and {rhs}")

    if lhs[-1] != rhs[-2]:
        raise ValueError(
            f"MatMul broadcast requires the last dimension of the first array to match the second-to-last dimension "
            f"of the second array, got shapes {lhs} and {rhs}")

    # get product matrix dimensions
    matrix_shape = (lhs[-2], rhs[-1])

    # determine broadcasting along stack dimensions
    # add additional dim for elementwise-broadcasting to work
    if len(lhs) < 3:
        lhs = (1,)
    if len(rhs) < 3:
        rhs = (1,)
    try:
        stacked_dims = elementwise_broadcast(lhs[:-2], rhs[:-2])
    except ValueError as err:
        raise ValueError(f"Failed to broadcast stacked matmul: {err}")

    return stacked_dims + matrix_shape


def normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Invalid axis {axis} for tensor with {ndim} dimensions")
    return axis


def reformat_reduce_axes(shape: Tuple[int, ...], axes: Union[int, Tuple[int, ...], None]) -> Tuple[int, ...]:
    if axes is None:
        axes = tuple(range(len(shape)))
    if isinstance(axes, int):
        axes = (axes,)
    return axes


def shaped_size(shape: Sequence[int]) -> int:
    return functools.reduce(lambda acc, e: acc * e, shape, 1)