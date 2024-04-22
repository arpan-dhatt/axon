import mlx.core as mx
from mlx import nn
import numpy as np
import time
from benchmarker import benchmarker

from mistral import make_mistral

if __name__ == "__main__":
    mistralmodel = make_mistral()
    mx.eval(mistralmodel.parameters())

    batch_sizes = [max(e, 1) for e in range(1, 8 + 1, 1)]
    prompt_sizes = [e for e in range(128, 2**10 + 1, 128)]
    print(batch_sizes, len(batch_sizes))
    print(prompt_sizes, len(prompt_sizes))

    def make_model(batch_size, prompt_size):
        return mistralmodel

    def make_input(batch_size, prompt_size):
        input = mx.random.uniform(0.0, 32768, (batch_size, prompt_size)).astype(mx.int32)
        print(input.shape)
        mx.eval(input)
        return input

    benchmarker(("Batch Size", batch_sizes), ("Prompt Length", prompt_sizes),
                make_model, make_input, "llm.mlx.gpu.npz",
                trails=7, init_warmup=10, warmup=2)