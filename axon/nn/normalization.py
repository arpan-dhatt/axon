import axon as ax
from axon.nn.base import Module


class LayerNorm(Module):
    r"""Applies layer normalization [1] on the inputs.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    [1]: https://arxiv.org/abs/1607.06450

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
        affine (bool): If True learn an affine transform to apply after the
            normalization
        bias (bool): If True include a translation to the affine
            transformation. If set to False the transformation is not really affine
            just scaling.
    """

    def __init__(
        self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True
    ):
        super().__init__()
        if affine:
            self.weight = ax.ones((dims,))
            if bias:
                self.bias = ax.zeros((dims,))
        self.eps = eps
        self.dims = dims

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x):
        weight = self.weight if "weight" in self else None
        bias = self.bias if "bias" in self else None
        return ax.nn.functional.layer_norm(x, weight, bias, self.eps)
