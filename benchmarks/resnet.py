import mlx.core as mx
from benchmarker import benchmarker
from mlx import nn
import numpy as np
import time

from resnet_model import resnet56

if __name__ == "__main__":
    batch_sizes = [max(e, 1) for e in range(0, 129, 16)]
    image_sizes = [e for e in range(32, 128 + 1, 8)]
    print(batch_sizes, len(batch_sizes))
    print(image_sizes, len(image_sizes))

    TRIALS = 20
    INIT_WARMUP = 100
    WARMUP = 10

    def make_model(batch_size, layer_size):
        model = resnet56()
        model.set_dtype(mx.float16)
        mx.eval(model.parameters())
        return model

    def make_input(batch_size, image_size):
        input = mx.random.normal((batch_size, image_size, image_size, 3), dtype=mx.float16)
        mx.eval(input)
        return input

    benchmarker(
        ("Batch Size", batch_sizes), ("Image Size", image_sizes),
        make_model, make_input,
        "resnet.mlx.gpu.npz",
        trace=[(1, 64), (64, 64)]
    )