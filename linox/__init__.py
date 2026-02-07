"""linox: Linear operators in JAX."""
from . import api

# Version infompatibility aliases (deprecated)
from .api import *
from .api import (
    _broadcast_shapes,  # for backwards compat
    lexp,
    llog,
    lpow,
)
from .api import (
    eigh as leigh,
)
from .api import eye as identity  # maybe?
from .api import inv as linverse
from .api import pinv as lpinverse
from .api import solve as lsolve
from .api import sqrt as lsqrt
from .api import trace as ltrace

# Backward compatibility imports for kernel operators
from .operators.kernel import (
    ArrayKernel,
    Kernel,
    KernelLinearOperator,
    ToeplitzKernel,
    kernel_operator,
)

__version__ = "0.0.3"


# Define __all__ combining api and local exports
__all__ = [*api.__all__, "kernel_operator", "ArrayKernel", "KernelLinearOperator", "Kernel", "ToeplitzKernel", "leigh", "lexp", "llog", "lpow", "identity", "linverse", "lpinverse", "lsolve", "lsqrt", "ltrace"]


# Other legacy names if needed
# from .operators._matrix import Matrix, Identity, etc are already in api
