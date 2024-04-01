import axon as ax

if __name__ == "__main__":
    A = ax.Tensor((10, 10, 10), ax.Float16)
    B = ax.Tensor((128, 10), ax.Float16)
    out = B @ A + B
    transposed = out.permute_dims((0, 2, 1))
    ax.print_graph([out.mean(axes=0), transposed.mT().mean((1,)).reshape(-1)])
