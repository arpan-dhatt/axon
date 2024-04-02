import axon as ax


def fn(params):
    a, b = params
    return (a * b).reduce_sum().squeeze()


def fn2(params):
    a, b = params
    c = ax.scalar(1.0, ax.Float16)
    return ax.concat((a,b), 0).reduce_sum().squeeze()


if __name__ == "__main__":
    x = ax.Tensor((100,), dtype=ax.Float16)
    # loss, grads = ax.value_and_grad(fn)((x, x))
    # ax.print_graph({"loss": loss, "grads": grads})

    print()
    loss, grads = ax.value_and_grad(fn2)((x, x))
    ax.print_graph({"loss": loss, "grads": grads})
