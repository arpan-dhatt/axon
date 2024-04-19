import mlx.core as mx
from mlx import nn
import numpy as np
import time

from resnet_model import resnet56

if __name__ == "__main__":
    batch_sizes = [max(e, 1) for e in range(0, 129, 16)]
    image_sizes = [e for e in range(32, 128, 8)]
    print(batch_sizes, len(batch_sizes))
    print(image_sizes, len(image_sizes))

    TRIALS = 100
    WARMUP = 20
    NUM_LAYERS = 8
    latency_mat = np.zeros((len(batch_sizes), len(image_sizes)), dtype=np.float32)
    for i, bs in enumerate(batch_sizes):
        for j, iwh in enumerate(image_sizes):
            # initialize model
            model = resnet56()
            model.set_dtype(mx.float16)
            mx.eval(model.parameters())

            agg_latency = 0.0
            for t in range(TRIALS):
                input = mx.random.normal((bs, iwh, iwh, 3), dtype=mx.float16)
                mx.eval(input)

                pred = model(input)

                tic = time.process_time()
                mx.eval(pred)
                if t >= WARMUP:
                    agg_latency += time.process_time() - tic

            latency_mat[i, j] = agg_latency / (TRIALS - WARMUP)
            print(bs, iwh, latency_mat[i, j])