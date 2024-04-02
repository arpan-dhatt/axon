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
        for w, b in self.layers:
            x = ((x @ w) + b).maximum(0)
        return x


def loss_fn(params, x):
    for w, b in params:
        x = ((x @ w) + b).maximum(0)
    return x.mean().squeeze()


if __name__ == "__main__":
    net = MLP([64, 32, 16, 10])
    x = ax.Tensor((128, 64), dtype=ax.Float16)
    out = net(x)

    loss, grads = ax.value_and_grad(loss_fn)(net.layers, x)
    ax.print_graph({"loss": loss, "grads": grads})