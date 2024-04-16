import axon as ax
from numpy_backend import NumpyBackend


class MLP(ax.nn.Module):
    def __init__(self, layers: list, lact: callable, fact: callable):
        super().__init__()
        self.lact = lact
        self.fact = fact

        self.layers = []
        self.norms = []
        for i in range(1, len(layers)):
            self.layers.append(ax.nn.Linear(layers[i-1], layers[i], bias=True))
            self.norms.append(ax.nn.LayerNorm(layers[i]))

    def __call__(self, x):
        for i, (lin, norm) in enumerate(zip(self.layers, self.norms)):
            if i == len(self.layers) - 1:
                return self.fact(norm(lin(x)))
            else:
                x = self.lact(norm(lin(x)))


def loss_fn(module, x: ax.Tensor, y: ax.Tensor):
    return (y - module(x)).mean().squeeze()


if __name__ == "__main__":
    with NumpyBackend():
        mod = MLP([128, 64, 64, 32, 10], ax.nn.relu, ax.nn.softmax)
        inp = ax.fill(1.0, (128, 128), dtype=ax.Float32)
        out = ax.fill(0.5, (128, 10), dtype=ax.Float32)
        pred = mod(inp)
        ax.print_graph(pred)
        print(pred)
        loss, grads = ax.nn.value_and_grad(mod, loss_fn)(mod, inp, out)
        ax.print_graph({"loss": loss, "grads": grads})
        print(mod)