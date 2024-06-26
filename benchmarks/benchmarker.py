from typing import *
import mlx.core as mx
from mlx import nn
import numpy as np
import time

def benchmarker(
        x: Tuple[str, list],
        y: Tuple[str, list],
        make_model: callable,
        make_input: callable,
        fname: str,
        trails = 20,
        init_warmup = 100,
        warmup = 10,
        sem = "mx",
        trace=None
):
    if trace is None:
        trace = []
    print(x[0], len(x[1]))
    print(y[0], len(y[1]))

    TRIALS = trails
    INIT_WARMUP = init_warmup
    WARMUP = warmup

    model = make_model(x[1][0], y[1][0])
    print("starting warmup")

    if sem == "mx":
        input = make_input(x[1][0], y[1][0])
        mx.eval(input)
        for w in range(INIT_WARMUP):
            mx.eval(model(input))
    else:
        input = make_input(x[1][0], y[1][0])
        for w in range(INIT_WARMUP):
            model(input)

    print("finished warmup")

    latency_mat = np.zeros((len(x[1]), len(y[1]), TRIALS - WARMUP), dtype=np.float64)
    for i, xs in enumerate(x[1]):
        for j, ys in enumerate(y[1]):
            # skip any xz/ys if we're tracing
            if len(trace) > 0 and (xs, ys) not in trace:
                continue

            # initialize model
            model = make_model(xs, ys)

            agg_latency = 0.0
            for t in range(TRIALS):
                input = make_input(xs, ys)

                bb = 0.0
                trace_name = None
                if t == TRIALS - 1 and (xs, ys) in trace:
                    print("getting trace")
                    # get a trace file of this
                    trace_name = f"{x[0][0]}{xs}_{y[0][0]}{ys}_" + fname.replace(".npz", ".gputrace")
                    assert mx.metal.start_capture(trace_name)

                if sem == "mx":
                    pred = model(input)

                    tic = time.time()
                    mx.eval(pred)
                    toc = time.time() - tic
                else:
                    tic = time.time()
                    pred = model(input)
                    bb += pred.flatten()[0].item()
                    toc = time.time() - tic
                if t >= WARMUP:
                    latency_mat[i, j, t - WARMUP] = toc

                if trace_name is not None:
                    mx.metal.stop_capture()


            print(xs, ys, np.median(latency_mat[i, j]), latency_mat[i, j].std())

    np.savez(fname, x=x[1], y=y[1], z=latency_mat)
