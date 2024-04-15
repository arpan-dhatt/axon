import axon as ax
from numpy_backend import NumpyBackend

if __name__ == "__main__":
    bknd = NumpyBackend()

    lin = ax.nn.Linear(100, 10, bias=True)
    inp = ax.fill(1.0, (128, 100), dtype=ax.Float32)

    out = lin(inp)
    ax.print_graph({"out": out})
    ax.print_graph(lin)
    print(lin)