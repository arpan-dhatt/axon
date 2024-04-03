from collections import defaultdict
from typing import *

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
        adjoint_dep_mapping: Dict[ax.Tensor, List[ax.Tensor]] = defaultdict(list)
        # use visited set to ensure deps are double-mapped if there's a fork/join in DAG
        primal_traversal_visited = set()

        def traverse_adjoint(primal_cursor: ax.Tensor):
            if primal_cursor.prim is not None and primal_cursor not in primal_traversal_visited:
                primal_traversal_visited.add(primal_cursor)
                # prim args depend on cursor for adjoint trace (reverse of primal trace)
                for arg in filter(lambda t: t.tracer, primal_cursor.prim.args):
                    adjoint_dep_mapping[arg].append(primal_cursor)
                    traverse_adjoint(arg)

        traverse_adjoint(output)

        # remove tracers before continuing
        for primal, primal_args in adjoint_dep_mapping.items():
            primal.unset_trace()
            map(lambda pa: pa.unset_trace(), primal_args)

        # recurse from primals using deps, caching completed {primal: adjoints}
        adjoint_mapping: Dict[ax.Tensor, ax.Tensor] = {output: ax.scalar(1, output.dtype)}
        # unset trace here since acc_adjoint won't recalculate this adjoint we've made already
        output.unset_trace()
        # use incomplete adjoints {primal: {adjoint_dep: [adjoints]}}
        # use list since a primal's output may be used more than once downstream (e.g. add(a, a))
        incomplete_adjoints: Dict[ax.Tensor, Dict[ax.Tensor, List[ax.Tensor]]] = defaultdict(dict)
        # prevents running backward multiple times and repeat filling incomplete adjoints
        backward_has_run: Set[ax.Primitive] = set()
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
                # go through all adjoint trace dependencies and run backward
                for adjoint_dep in adjoint_dep_mapping[primal_cursor]:
                    backward(adjoint_dep)

                # incomplete cache should be filled now
                adjoint_addends = ax.utils.flatten_list(list(incomplete_adjoints[primal_cursor].values()))
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
            assert primal_cursor.prim is not None
            if primal_cursor.prim in backward_has_run:
                return
            backward_has_run.add(primal_cursor.prim)

            # get (probably incomplete) adjoints of arguments
            if len(primal_cursor.siblings) == 0:
                # no siblings, just add together all adjoints of primals using this primal
                adjoints = [acc_adjoint(primal_cursor)]
            else:
                # require all sibling adjoints in order to run backwards
                # adjoint_dep_mapping keys are all tensors contributing to final scalar value so any siblings
                # not in it must have a zero adjoint
                adjoints = [acc_adjoint(sib) if sib in adjoint_dep_mapping else ax.zeros_like(sib)
                            for sib in primal_cursor.siblings]
            args_adjoints = primal_cursor.prim.backward(adjoints)

            # add adjoints calculated from this backward call to cache
            for arg, arg_adjoint in zip(primal_cursor.prim.args, args_adjoints):
                if primal_cursor in incomplete_adjoints[arg]:
                    incomplete_adjoints[arg][primal_cursor].append(arg_adjoint)
                else:
                    incomplete_adjoints[arg][primal_cursor] = [arg_adjoint]

        for key, primal in ax.utils.tree_flatten(traced_arg):
            grads.append((key, acc_adjoint(primal)))

        return outputs, ax.utils.tree_unflatten(grads)

    return value_and_grad_fn


def grad(fn: Callable, argnum: int = 0) -> callable:
    def grad_fn(*args, **kwargs):
        return value_and_grad(fn, argnum)(*args, **kwargs)[1]

    return grad_fn
