from axon.dtype import DType, UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float16, Float32, Float64, \
    BFloat16, Bool
from axon.tensor import Tensor
import axon.utils

# primitive base class
from axon.primitives import Primitive

# unary
from axon.ops import broadcast, cast, negate, stop_gradient, sqrt, log, wrap_scalar
# manipulation
from axon.ops import reshape, permute_dims, matrix_transpose, array_slice, expand_dims, flatten, squeeze, split
# reduction
from axon.ops import reduce_sum, product, mean, reduce_min, reduce_max
# binary
from axon.ops import broadcast_pair, add, subtract, multiply, divide, maximum, minimum, matmul, power
# binary masking
from axon.ops import greater, greater_or_equal, equal, lesser, lesser_or_equal, mask
# logical bool ops
from axon.ops import logical_or, logical_and, logical_not
# trig
from axon.ops import sin, cos, tan, sinh, cosh, tanh
# inverse trig
from axon.ops import arcsin, arccos, arctan, arcsinh, arccosh, arctanh
# n-ary
from axon.ops import concat, stack
# debugging
from axon.ops import print_graph

# transforms
from axon.transforms import value_and_grad, grad

# expose static methods of tensor
scalar = Tensor.scalar
zeros_like = Tensor.zeros_like
