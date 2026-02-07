"""linox: Linear operators in JAX."""
from . import api

# Version infompatibility aliases (deprecated)
from .api import *
from .api import (
    eigh as leigh,  # log as llog, exp as lexp, pow as lpow # if these names existed
)
from .api import eye as identity  # maybe?
from .api import inv as linverse
from .api import pinv as lpinverse
from .api import solve as lsolve
from .api import sqrt as lsqrt
from .api import trace as ltrace
from .api import _broadcast_shapes  # for backwards compat

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
__all__ = [*api.__all__, "kernel_operator", "ArrayKernel", "KernelLinearOperator", "Kernel", "ToeplitzKernel", "leigh", "identity", "linverse", "lpinverse", "lsolve", "lsqrt", "ltrace"]


# Other legacy names if needed
# from .operators._matrix import Matrix, Identity, etc are already in api
