import axon as ax


def linear_fn(params, x):
    w, b = params
    return x @ w + b


def loss_fn(params, x, y):
    y_hat = ax.stop_gradient(linear_fn(params, x))
    return ((y - y_hat) * (y - y_hat)).mean().squeeze()


def fold_sum(w):
    stuff = ax.split(w, 8, 0)
    summed = stuff[0]
    for i in range(1, 4):
        summed = summed + stuff[i]
    return ax.stop_gradient(summed.reduce_sum().squeeze())


if __name__ == "__main__":
    w = ax.Tensor((64, 10), ax.Float16)
    b = ax.Tensor((10,), ax.Float16)

    x = ax.Tensor((128, 64), ax.Float16)
    y = ax.Tensor((128, 10), ax.Float16)

    # loss, grads = ax.value_and_grad(fold_sum)(w)
    # ax.print_graph({"w": w, "loss": loss, "grads": grads})

    loss, grads = ax.value_and_grad(loss_fn)((w, b), x, y)
    print(grads)
    ax.print_graph({"w": w, "b": b, "loss": loss, "grads": grads, "x": x, "gt": y})
