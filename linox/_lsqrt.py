"""Linear operator square root.

TODO(2bys): Move main function to lsqrt.
"""

import functools
import jax.numpy as jnp
from linox._linear_operator import LinearOperator
from linox._matrix import Identity, Scalar


@functools.singledispatch
def lsqrt(A: LinearOperator) -> LinearOperator:
    raise NotImplementedError()


@lsqrt.register
def _(A: LinearOperator) -> LinearOperator:
    return A.cholesky(lower=True)


@lsqrt.register
def _(A: Identity) -> Identity:
    return A


@lsqrt.register
def _(A: Scalar) -> Scalar:
    return Scalar(jnp.sqrt(A.scalar))
