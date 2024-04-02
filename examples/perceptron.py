from typing import *

import axon as ax


class MLP:
    def __init__(self, layers: List[int]):
        self.layers = []
        for i, layer in enumerate(layers):
            if i == len(layers) - 1: break
            w, b = (ax.Tensor((layer, layers[i + 1]), ax.Float16),
                    ax.Tensor((layers[i + 1],), ax.Float16))
            self.layers.append((w, b))

    def __call__(self, x: ax.Tensor):
        zero = ax.scalar(0, dtype=ax.Float16)
        for w, b in self.layers:
            x = ((x @ w) + b).maximum(zero)
        return x


def loss_fn(params, x):
    zero = ax.scalar(0, dtype=ax.Float16)
    for w, b in params:
        x = ((x @ w) + b)
    return x.mean().squeeze()


if __name__ == "__main__":
    net = MLP([64, 32, 16, 10])
    x = ax.Tensor((128, 64), dtype=ax.Float16)
    out = net(x)

    grads = ax.grad(loss_fn)(net.layers, x)
    ax.print_graph({"grads": grads})