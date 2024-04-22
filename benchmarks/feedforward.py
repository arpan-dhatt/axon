import mlx.core as mx
from mlx import nn
import numpy as np
import time
from benchmarker import benchmarker

class MLP(nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mx.maximum(l(x), 0.0)
        return self.layers[-1](x)




if __name__ == "__main__":
    batch_sizes = [max(e, 1) for e in range(0, 1024 + 1, 128)]
    layer_sizes = [e for e in range(2048, 2**14 + 1, 2048)]
    print(batch_sizes, len(batch_sizes))
    print(layer_sizes, len(layer_sizes))

    TRIALS = 20
    INIT_WARMUP = 100
    WARMUP = 10
    NUM_LAYERS = 4


    def make_model(batch_size, layer_size):
        model = MLP(NUM_LAYERS, layer_size, layer_size, layer_size)
        model.set_dtype(mx.float16)
        mx.eval(model.parameters())
        return model

    def make_input(batch_size, layer_size):
        input = mx.random.normal((batch_size, layer_size), dtype=mx.float16)
        mx.eval(input)
        return input

    benchmarker(("Batch Size", batch_sizes), ("Layer Size", layer_sizes),
                make_model, make_input, "ff.mlx.gpu.npz",
                trace=[(1, 16384), (128, 16384)])