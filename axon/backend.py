from typing import *
from abc import ABC, abstractmethod

import axon as ax


class Backend(ABC):

    @abstractmethod
    def eval(self, tensors: List[Tuple[str, ax.Tensor]]):
        pass