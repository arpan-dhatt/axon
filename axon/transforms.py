from typing import *
from collections import defaultdict

import axon as ax


def value_and_grad(fn: Callable, argnum: int = 0) -> callable:
    def value_and_grad_fn(*args, **kwargs):
        # set tracers on target arg
        traced_arg = ax.utils.tree_map(lambda t: t.set_trace(), args[argnum])
        traced_args = [arg if i != argnum else traced_arg for i, arg in enumerate(args)]

        # run traced args through fn to get all outputs
        outputs = fn(*traced_args, **kwargs)

        # extract scalar output and reroute other outputs
        output: ax.Tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        assert len(output.shape) == 0, ("first output of fn in value_and_grad "
                                        "must be a scalar (shape () not {output.shape})")
        if not output.tracer:
            # output didn't appear in trace, so grads must be zeros
            raise NotImplementedError("TODO filling grads with zeros_like")

        # perform backward pass along primal trace to track adjoint dependencies
        # stores the adjoint's primal
        adjoint_dep_mapping: Dict[ax.Tensor, List[ax.Tensor]] = defaultdict(list)

        def traverse_adjoint(primal_cursor: ax.Tensor):
            if primal_cursor.prim is not None:
                # prim args depend on cursor for adjoint trace (reverse of primal trace)
                for arg in filter(lambda t: t.tracer, primal_cursor.prim.args):
                    adjoint_dep_mapping[arg].append(primal_cursor)
                    traverse_adjoint(arg)

        traverse_adjoint(output)

        # recurse from primals using deps, caching completed {primal: adjoints}
        adjoint_mapping: Dict[ax.Tensor, ax.Tensor] = {output: ax.scalar(1, output.dtype)}
        # unset trace here since acc_adjoint won't recalculate this adjoint we've made already
        output.unset_trace()
        # use incomplete adjoints {primal: {adjoint_dep: adjoint}}
        incomplete_adjoints: Dict[ax.Tensor, Dict[ax.Tensor, ax.Tensor]] = defaultdict(dict)
        # prevents running backward multiple times
        backward_has_run: Dict[ax.Tensor, bool] = defaultdict(bool)
        grads: List[Tuple[str, ax.Tensor]] = []

        def acc_adjoint(primal_cursor: ax.Tensor) -> ax.Tensor:
            """
            Returns the accumulated adjoint, either recursively
            calculated from adjoint dependencies or retrieved
            from adjoint_mapping
            :param primal_cursor: respective primal of adjoint to calculate
            """
            if primal_cursor in adjoint_mapping:
                # complete adjoint, just return it
                return adjoint_mapping[primal_cursor]
            else:
                # unset trace for all primals along the way
                primal_cursor.unset_trace()
                # go through all adjoint trace dependencies and run backward
                for adjoint_dep in adjoint_dep_mapping[primal_cursor]:
                    backward(adjoint_dep)

                # incomplete cache should be filled now
                adjoint_addends = list(incomplete_adjoints[primal_cursor].values())
                adjoint = adjoint_addends[0]
                for addend in adjoint_addends[1:]:
                    adjoint = adjoint + addend
                adjoint_mapping[primal_cursor] = adjoint
                return adjoint

        def backward(primal_cursor: ax.Tensor):
            """
            Runs backward on a tensor's primitive and fills the incomplete adjoints cache
            :param primal_cursor: tensor (with primitive) to run backward
            """
            if backward_has_run[primal_cursor]:
                return

            # get (probably incomplete) adjoints of arguments
            adjoint = acc_adjoint(primal_cursor)
            args_adjoints = primal_cursor.prim.backward(adjoint)

            for arg, arg_adjoint in zip(primal_cursor.prim.args, args_adjoints):
                incomplete_adjoints[arg][primal_cursor] = arg_adjoint

        for key, primal in ax.utils.tree_flatten(traced_arg):
            grads.append((key, acc_adjoint(primal)))

        return outputs, ax.utils.tree_unflatten(grads)

    return value_and_grad_fn


def grad(fn: Callable, argnum: int = 0) -> callable:
    def grad_fn(*args, **kwargs):
        return value_and_grad(fn, argnum)(*args, **kwargs)[1]

    return grad_fn
