from typing import *
from abc import ABC, abstractmethod

import axon as ax


class Backend(ABC):

    @abstractmethod
    def tensor(self, data: Any, shape: Tuple[int, ...] = None, dtype: ax.DType = None) -> ax.Tensor:
        pass

    @abstractmethod
    def eval(self, tensors: List[Tuple[str, ax.Tensor]], **kwargs):
        pass