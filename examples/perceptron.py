from typing import *

import axon as ax
import numpy as np
from numpy_backend import NumpyBackend


class MLP:
    def __init__(self, layers: List[int]):
        self.layers = []
        for i, layer in enumerate(layers):
            if i == len(layers) - 1: break
            w, b = (ax.fill(1, (layer, layers[i + 1]), ax.Float32),
                    ax.fill(0, (layers[i + 1],), ax.Float32))
            self.layers.append((w, b))

    def __call__(self, x: ax.Tensor):
        for w, b in self.layers:
            x = ((x @ w) + b).maximum(0)
        return x


def loss_fn(params, x):
    for w, b in params:
        x = ((x @ w) + b).maximum(0)
    return ax.stop_gradient(x).mean().squeeze()


if __name__ == "__main__":
    bknd = NumpyBackend()

    import time
    net = MLP([32] * 2)
    x = ax.fill(1, (128, 32), dtype=ax.Float32)

    tick = time.time()
    out = net(x)
    print("fwddef", time.time() - tick)
    # ax.print_graph(out)
    tick = time.time()
    ax.eval(out, bknd)
    print("fwdrun", time.time() - tick)
    print(out.data)

    tick = time.time()
    loss, grads = ax.value_and_grad(loss_fn)(net.layers, x)
    print("vgrun", time.time() - tick)
    ax.print_graph({"loss": loss, "grads": grads})
    ax.eval(grads, bknd)
    print(grads[-1][-1].data)