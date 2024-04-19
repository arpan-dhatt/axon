import mlx.core as mx
from mlx import nn
import numpy as np
import time

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
    batch_sizes = [max(e, 1) for e in range(0, 129, 8)]
    layer_sizes = [e for e in range(1024, 2**15 + 1, 1024)]
    print(batch_sizes)
    print(layer_sizes)

    TRIALS = 100
    WARMUP = 20
    NUM_LAYERS = 8
    latency_mat = np.zeros((len(batch_sizes), len(layer_sizes)), dtype=np.float32)
    for i, bs in enumerate(batch_sizes):
        for j, ls in enumerate(layer_sizes):
            # initialize model
            model = MLP(NUM_LAYERS, ls, ls, ls)
            model.set_dtype(mx.float16)
            mx.eval(model.parameters())

            agg_latency = 0.0
            for t in range(TRIALS):
                input = mx.random.normal((bs, ls), dtype=mx.float16)
                mx.eval(input)

                pred = model(input)

                tic = time.process_time()
                mx.eval(pred)
                if t >= WARMUP:
                    agg_latency += time.process_time() - tic

            latency_mat[i, j] = agg_latency / (TRIALS - WARMUP)
            print(bs, ls, latency_mat[i, j])