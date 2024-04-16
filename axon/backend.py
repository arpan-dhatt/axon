from typing import *
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from contextvars import ContextVar

import axon as ax


class Backend(AbstractContextManager):

    @abstractmethod
    def tensor(self, data: Any, shape: Tuple[int, ...] = None, dtype: ax.DType = None) -> ax.Tensor:
        pass

    @abstractmethod
    def eval(self, tensors: List[Tuple[str, ax.Tensor]], **kwargs):
        pass

    def __enter__(self):
        self.context_token = ax.context.backend.set(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        ax.context.backend.reset(self.context_token)
