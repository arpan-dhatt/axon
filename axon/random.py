from typing import *
import math

import axon as ax


def random_bits(shape: Sequence[int], width: int) -> ax.Tensor:
    if width == 1:
        dtype = ax.UInt8
    elif width == 2:
        dtype = ax.UInt16
    elif width == 4:
        dtype = ax.UInt32
    elif width == 8:
        dtype = ax.UInt64
    else:
        raise ValueError(f"Cannot make random bits with element width of {width} bytes")

    return ax.Tensor(tuple(shape), dtype, prim=ax.primitives.RandomBits())


def uniform(lower: float, upper: float, shape: Sequence[int], dtype: ax.DType) -> ax.Tensor:
    assert dtype in [ax.Float64, ax.Float32, ax.Float16, ax.BFloat16], "Uniform distribution must be of real values"
    bits = random_bits(shape, dtype.stride)
    intermediate_dtype = ax.Float32 if dtype in [ax.Float16, ax.BFloat16, ax.Float32] else ax.Float64
    num_range = upper - lower
    return (bits.cast(intermediate_dtype) / (2**(dtype.stride*8)) * num_range + lower).cast(dtype)


def normal(shape: Sequence[int], dtype: ax.DType, loc: float = 0.0, scale: float = 1.0) -> ax.Tensor:
    assert dtype in [ax.Float64, ax.Float32, ax.Float16, ax.BFloat16], "Uniform distribution must be of real values"
    # note: need to make this work without requiring intermediates to happen in Float64
    samples = uniform(math.nextafter(-1.0, math.inf), 1.0, shape, ax.Float64)
    return (ax.wrap_scalar(math.sqrt(2.0), ax.Float64)
            * ax.erfinv(samples) * scale + loc).cast(dtype)
