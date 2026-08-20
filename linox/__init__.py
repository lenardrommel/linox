"""linox: Linear operators in JAX."""

from . import api
from .api import *  # noqa: F403

# Deprecated "l"-prefixed aliases, kept for backwards compatibility and
# scheduled for removal in 0.0.4. These are renames rather than plain
# re-exports, so they are listed in `__all__` below instead of using the
# redundant `X as X` form that marks an intentional re-export.
from .api import _broadcast_shapes as _broadcast_shapes
from .api import eigh as leigh  # noqa: F401
from .api import eye as identity  # noqa: F401
from .api import inv as linverse  # noqa: F401
from .api import lexp as lexp
from .api import llog as llog
from .api import lpow as lpow
from .api import pinv as lpinverse  # noqa: F401
from .api import solve as lsolve  # noqa: F401
from .api import sqrt as lsqrt  # noqa: F401
from .api import trace as ltrace  # noqa: F401

# Backward compatibility imports for kernel operators
from .operators.kernel import ArrayKernel as ArrayKernel
from .operators.kernel import Kernel as Kernel
from .operators.kernel import KernelLinearOperator as KernelLinearOperator
from .operators.kernel import ToeplitzKernel as ToeplitzKernel
from .operators.kernel import kernel_operator as kernel_operator

__version__ = "0.0.3"


_LEGACY_ALIASES = [
    "identity",
    "leigh",
    "lexp",
    "linverse",
    "llog",
    "lpinverse",
    "lpow",
    "lsolve",
    "lsqrt",
    "ltrace",
]

_KERNEL_EXPORTS = [
    "ArrayKernel",
    "Kernel",
    "KernelLinearOperator",
    "ToeplitzKernel",
    "kernel_operator",
]

# `api.__all__` already carries several of the legacy alias names, so
# de-duplicate rather than concatenating blindly: a name repeated in `__all__`
# breaks star-import tooling. `_broadcast_shapes` stays importable for
# backwards compatibility but is deliberately not advertised as public.
__all__ = sorted({*api.__all__, *_KERNEL_EXPORTS, *_LEGACY_ALIASES})
