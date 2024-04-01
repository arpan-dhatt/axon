from typing import *
from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    stride: int
    name: str

    def __repr__(self):
        return self.name


UInt8 = DType(stride=1, name="UInt8")
UInt16 = DType(stride=2, name="UInt16")
UInt32 = DType(stride=4, name="UInt32")
UInt64 = DType(stride=8, name="UInt64")

Int8 = DType(stride=1, name="Int8")
Int16 = DType(stride=2, name="Int16")
Int32 = DType(stride=4, name="Int32")
Int64 = DType(stride=8, name="Int64")

Float16 = DType(stride=2, name="Float16")
Float32 = DType(stride=4, name="Float32")
Float64 = DType(stride=8, name="Float64")

BFloat16 = DType(stride=2, name="BFloat16")

Bool = DType(stride=1, name="Bool")