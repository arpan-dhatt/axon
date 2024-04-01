from axon.dtype import DType, UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float16, Float32, Float64, \
    BFloat16, Bool
from axon.tensor import Tensor

# unary
from axon.ops import broadcast, cast, negate, reshape, permute_dims, matrix_transpose, array_slice, expand_dims
# reduction
from axon.ops import reduce_sum, product, mean, reduce_min, reduce_max
# binary
from axon.ops import add, subtract, multiply, divide, maximum, minimum, matmul
# n-ary
from axon.ops import concat, stack
# debugging
from axon.ops import print_graph

# expose static methods of tensor
scalar = Tensor.scalar
