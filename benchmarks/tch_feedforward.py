import torch
import torch.nn as nn
from benchmarker import benchmarker

class MLP(nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int, dev="mps"
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim, dtype=torch.float16).to(dev)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def forward(self, x):
        for l in self.layers[:-1]:
            x = torch.relu(l(x))
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

    mps = torch.device("mps")

    def make_model(batch_size, layer_size):
        model = MLP(NUM_LAYERS, layer_size, layer_size, layer_size).to(mps)
        return model

    def make_input(batch_size, layer_size):
        input = torch.normal(0.0, 1.0, size=(batch_size, layer_size), dtype=torch.float16).to(mps)
        return input

    benchmarker(("Batch Size", batch_sizes), ("Layer Size", layer_sizes),
                make_model, make_input, "ff.torch.gpu.npz", sem="tch")