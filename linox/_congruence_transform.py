import plum

# import probnum as pn
# from probnum.linops._arithmetic_fallbacks import ProductLinearOperator
import linox
from linox._arithmetic import ProductLinearOperator
from probnum.typing import LinearOperatorLike

from ._lsqrt import lsqrt


class CongruenceTransform(ProductLinearOperator):
    r""":math:`A B A^\top`"""

    def __init__(self, A: LinearOperatorLike, B: LinearOperatorLike | None = None):
        self._A = linox.aslinop(A)
        self._B = (
            linox.aslinop(B) if B is not None else linox.Identity(self._A.shape[1])
        )

        super().__init__(self._A, self._B, self._A.T)

        self.is_symmetric = self._B.is_symmetric
        self.is_positive_definite = self._B.is_positive_definite

    def _transpose(self) -> linox.LinearOperator:
        return CongruenceTransform(self._A, self._B.T)


@lsqrt.register
def _(A: CongruenceTransform) -> ProductLinearOperator:
    return A._A @ lsqrt(A._B)


@plum.dispatch
def congruence_transform(A, B=None):
    return CongruenceTransform(A, B)


@congruence_transform.dispatch(precedence=1000)
def _(A: linox.Identity, B=None):
    return B if B is not None else A


@congruence_transform.dispatch
def _(A: ProductLinearOperator, B=None):
    for factor in reversed(A.factors):
        B = congruence_transform(factor, B)

    return B


# @congruence_transform.dispatch
# def _(A: linox.BlockDiagonalMatrix, _: linox.Identity | NoneType = None):
#     return linox.BlockDiagonalMatrix(
#         *(congruence_transform(block) for block in A.blocks)
#     )
