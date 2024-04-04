from typing import *

import axon as ax
from numpy_backend import NumpyBackend


def fn(params):
    a, b = params
    return (a * b).reduce_sum().squeeze()


def ones_backward(input: ax.Tensor):
    def one_grad(adjoints: List[ax.Tensor], argnums: Optional[Tuple[int, ...]] = None) -> Tuple[
        Optional[ax.Tensor], ...]:
        return (ax.fill_like(input, 24.0),)

    return ax.custom_gradient((input,), lambda t: (ax.negate(input),), one_grad)


def fn2(params):
    a, b = params
    c = ax.scalar(1.0, ax.Float16)
    return ones_backward(ax.concat((a, b), 0))[0].reduce_sum().squeeze()


if __name__ == "__main__":
    x = ax.fill(2, (100,), dtype=ax.Float16)
    loss, grads = ax.value_and_grad(fn)((x, x))
    ax.print_graph({"loss": loss, "grads": grads})
    ax.eval({"loss": loss, "grads": grads}, NumpyBackend())
    print(loss.data)

    loss, grads = ax.value_and_grad(fn2)((x, x))
    ax.print_graph({"loss": loss, "grads": grads})
    ax.eval({"loss": loss, "grads": grads}, NumpyBackend())
    ax.utils.tree_map(lambda t: print(t.data), {"loss": loss, "grads": grads})
