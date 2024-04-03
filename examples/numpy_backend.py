from typing import *

import numpy as np

import axon as ax


class NumpyBackend(ax.Backend):
    eval_registry: Dict[str, callable]
    dtype_map: Dict[ax.DType, np.dtype]

    def __init__(self, eval_overrides=None):
        if eval_overrides is None:
            eval_overrides = {}

        self.eval_registry = {}

        for member in dir(self):
            if member.startswith("impl_"):
                self.eval_registry[member.split("_", 1)[-1]] = getattr(self, member)
        for key, value in eval_overrides.items():
            self.eval_registry[key] = value

        self.dtype_map = {
            ax.Bool: np.bool_,
            ax.UInt8: np.uint8,
            ax.UInt16: np.uint16,
            ax.UInt32: np.uint32,
            ax.UInt64: np.uint64,
            ax.Int8: np.int8,
            ax.Int16: np.int16,
            ax.Int32: np.int32,
            ax.Int64: np.int64,
            ax.Float16: np.float16,
            ax.Float32: np.float32,
            ax.Float64: np.float64,
        }

    def eval_tensor(self, tensor: ax.Tensor):
        if isinstance(tensor.data, (int, float, bool)):
            # promote existing scalar to typed array
            try:
                numpy_dtype = self.dtype_map[tensor.dtype]
                tensor.data = np.array(tensor.data, dtype=numpy_dtype)
            except KeyError as e:
                e.add_note(f"ax.{tensor.dtype} has no numpy equivalent")
        elif tensor.data is not None:
            # data is populated, no need to run
            return
        else:
            # run relevant primitive populate data
            assert tensor.prim is not None, "Tensor to eval must have populated data or "
            # check all args of primitive are evaluated first
            for arg in tensor.prim.args:
                self.eval_tensor(arg)
            self.run_primitive(tensor.prim, [tensor] if len(tensor.siblings) == 0 else tensor.siblings)

    def run_primitive(self, prim: ax.Primitive, outputs: List[ax.Tensor]):
        method = self.eval_registry[type(prim).__name__]
        method(prim, outputs)

    def impl_Add(self, prim: ax.primitives.Add, outputs: List[ax.Tensor]):
        outputs[0].data = np.add(prim.args[0].data, prim.args[1].data)

    def impl_Multiply(self, prim: ax.primitives.Multiply, outputs: List[ax.Tensor]):
        outputs[0].data = np.multiply(prim.args[0].data, prim.args[1].data)

    def impl_Broadcast(self, prim: ax.primitives.Broadcast, outputs: List[ax.Tensor]):
        outputs[0].data = np.broadcast_to(prim.args[0].data, prim.shape)

    def impl_MatMul(self, prim: ax.primitives.MatMul, outputs: List[ax.Tensor]):
        outputs[0].data = np.matmul(prim.args[0].data, prim.args[1].data)

    def impl_Cast(self, prim: ax.primitives.Cast, outputs: List[ax.Tensor]):
        try:
            numpy_dtype = self.dtype_map[prim.dtype]
            outputs[0].data = prim.args[0].data.astype(numpy_dtype)
        except KeyError as e:
            e.add_note(f"ax.{prim.dtype} has no numpy equivalent")

    def impl_Reshape(self, prim: ax.primitives.Reshape, outputs: List[ax.Tensor]):
        outputs[0].data = np.reshape(prim.args[0].data, prim.shape)

    def impl_PermuteDims(self, prim: ax.primitives.PermuteDims, outputs: List[ax.Tensor]):
        outputs[0].data = np.transpose(prim.args[0].data, prim.dims)

    def impl_Negate(self, prim: ax.primitives.Negate, outputs: List[ax.Tensor]):
        outputs[0].data = np.negative(prim.args[0].data)

    def impl_Sum(self, prim: ax.primitives.Sum, outputs: List[ax.Tensor]):
        outputs[0].data = np.sum(prim.args[0].data, axis=prim.axes)

    def impl_Product(self, prim: ax.primitives.Product, outputs: List[ax.Tensor]):
        outputs[0].data = np.prod(prim.args[0].data, axis=prim.axes)

    def impl_Max(self, prim: ax.primitives.Max, outputs: List[ax.Tensor]):
        outputs[0].data = np.max(prim.args[0].data, axis=prim.axes)

    def impl_Min(self, prim: ax.primitives.Min, outputs: List[ax.Tensor]):
        outputs[0].data = np.min(prim.args[0].data, axis=prim.axes)

    def impl_Subtract(self, prim: ax.primitives.Subtract, outputs: List[ax.Tensor]):
        outputs[0].data = np.subtract(prim.args[0].data, prim.args[1].data)

    def impl_Divide(self, prim: ax.primitives.Divide, outputs: List[ax.Tensor]):
        outputs[0].data = np.divide(prim.args[0].data, prim.args[1].data)

    def impl_Maximum(self, prim: ax.primitives.Maximum, outputs: List[ax.Tensor]):
        outputs[0].data = np.maximum(prim.args[0].data, prim.args[1].data)

    def impl_Minimum(self, prim: ax.primitives.Minimum, outputs: List[ax.Tensor]):
        outputs[0].data = np.minimum(prim.args[0].data, prim.args[1].data)

    def impl_Greater(self, prim: ax.primitives.Greater, outputs: List[ax.Tensor]):
        outputs[0].data = np.greater(prim.args[0].data, prim.args[1].data)

    def impl_Lesser(self, prim: ax.primitives.Lesser, outputs: List[ax.Tensor]):
        outputs[0].data = np.less(prim.args[0].data, prim.args[1].data)

    def impl_Equal(self, prim: ax.primitives.Equal, outputs: List[ax.Tensor]):
        outputs[0].data = np.equal(prim.args[0].data, prim.args[1].data)

    def impl_GreaterOrEqual(self, prim: ax.primitives.GreaterOrEqual, outputs: List[ax.Tensor]):
        outputs[0].data = np.greater_equal(prim.args[0].data, prim.args[1].data)

    def impl_LesserOrEqual(self, prim: ax.primitives.LesserOrEqual, outputs: List[ax.Tensor]):
        outputs[0].data = np.less_equal(prim.args[0].data, prim.args[1].data)

    def impl_LogicalNot(self, prim: ax.primitives.LogicalNot, outputs: List[ax.Tensor]):
        outputs[0].data = np.logical_not(prim.args[0].data)

    def impl_LogicalAnd(self, prim: ax.primitives.LogicalAnd, outputs: List[ax.Tensor]):
        outputs[0].data = np.logical_and(prim.args[0].data, prim.args[1].data)

    def impl_LogicalOr(self, prim: ax.primitives.LogicalOr, outputs: List[ax.Tensor]):
        outputs[0].data = np.logical_or(prim.args[0].data, prim.args[1].data)

    def impl_Concatenate(self, prim: ax.primitives.Concatenate, outputs: List[ax.Tensor]):
        outputs[0].data = np.concatenate([arg.data for arg in prim.args], axis=prim.axis)

    def impl_Split(self, prim: ax.primitives.Split, outputs: List[ax.Tensor]):
        split_data = np.split(prim.args[0].data, prim.indices_or_sections, axis=prim.axis)
        for output, data in zip(outputs, split_data):
            output.data = data

    def impl_Slice(self, prim: ax.primitives.Slice, outputs: List[ax.Tensor]):
        outputs[0].data = prim.args[0].data[prim.indices]

    def impl_Sin(self, prim: ax.primitives.Sin, outputs: List[ax.Tensor]):
        outputs[0].data = np.sin(prim.args[0].data)

    def impl_ArcSin(self, prim: ax.primitives.ArcSin, outputs: List[ax.Tensor]):
        outputs[0].data = np.arcsin(prim.args[0].data)

    def impl_Sinh(self, prim: ax.primitives.Sinh, outputs: List[ax.Tensor]):
        outputs[0].data = np.sinh(prim.args[0].data)

    def impl_ArcSinh(self, prim: ax.primitives.ArcSinh, outputs: List[ax.Tensor]):
        outputs[0].data = np.arcsinh(prim.args[0].data)

    def impl_Cos(self, prim: ax.primitives.Cos, outputs: List[ax.Tensor]):
        outputs[0].data = np.cos(prim.args[0].data)

    def impl_ArcCos(self, prim: ax.primitives.ArcCos, outputs: List[ax.Tensor]):
        outputs[0].data = np.arccos(prim.args[0].data)

    def impl_Cosh(self, prim: ax.primitives.Cosh, outputs: List[ax.Tensor]):
        outputs[0].data = np.cosh(prim.args[0].data)

    def impl_ArcCosh(self, prim: ax.primitives.ArcCosh, outputs: List[ax.Tensor]):
        outputs[0].data = np.arccosh(prim.args[0].data)

    def impl_Tan(self, prim: ax.primitives.Tan, outputs: List[ax.Tensor]):
        outputs[0].data = np.tan(prim.args[0].data)

    def impl_ArcTan(self, prim: ax.primitives.ArcTan, outputs: List[ax.Tensor]):
        outputs[0].data = np.arctan(prim.args[0].data)

    def impl_Tanh(self, prim: ax.primitives.Tanh, outputs: List[ax.Tensor]):
        outputs[0].data = np.tanh(prim.args[0].data)

    def impl_ArcTanh(self, prim: ax.primitives.ArcTanh, outputs: List[ax.Tensor]):
        outputs[0].data = np.arctanh(prim.args[0].data)

    def impl_Log(self, prim: ax.primitives.Log, outputs: List[ax.Tensor]):
        outputs[0].data = np.log(prim.args[0].data)

    def impl_Power(self, prim: ax.primitives.Power, outputs: List[ax.Tensor]):
        outputs[0].data = np.power(prim.args[0].data, prim.args[1].data)

    def impl_Sqrt(self, prim: ax.primitives.Sqrt, outputs: List[ax.Tensor]):
        outputs[0].data = np.sqrt(prim.args[0].data)

    def impl_Mask(self, prim: ax.primitives.Mask, outputs: List[ax.Tensor]):
        outputs[0].data = np.where(prim.args[1].data, prim.args[0].data, np.zeros_like(prim.args[0].data))

    def eval(self, tensors: List[Tuple[str, ax.Tensor]]):
        for name, tensor in tensors:
            self.eval_tensor(tensor)


if __name__ == "__main__":
    bknd = NumpyBackend()
    x = ax.fill(1, (128, 32), ax.Float16)
    w = ax.fill(2, (32, 10), ax.Float16)
    b = ax.fill(7, (10,), ax.Float16)

    y = x @ w + b
    ax.print_graph(y)
    ax.eval(y, backend=bknd)
    print(y.data)
