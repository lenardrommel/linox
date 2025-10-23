# gp.py (moved from tests/gp.py)

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

try:  # Python < 3.11 compatibility
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - only exercised on older interpreters

    class StrEnum(str, Enum):
        """Fallback StrEnum implementation for Python < 3.11."""

        pass


import linox as lo

try:
    from helper.plotting import (
        generate_preprocess_data_1d,
        generate_preprocess_data_2d,
        plot_error_analysis_2d,
        plot_kernel_predictions_1d,
        plot_kernel_predictions_2d,
        plot_marginal_analysis_2d,
        plot_profile_comparison_2d,
        plot_uncertainty_2d,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for package installs
    from tests.helper.plotting import (  # type: ignore[no-redef]
        generate_preprocess_data_1d,
        generate_preprocess_data_2d,
        plot_error_analysis_2d,
        plot_kernel_predictions_1d,
        plot_kernel_predictions_2d,
        plot_marginal_analysis_2d,
        plot_profile_comparison_2d,
        plot_uncertainty_2d,
    )
from linox import (
    AddLinearOperator,
    IsotropicAdditiveLinearOperator,
    Kronecker,
    ProductLinearOperator,
    ScaledLinearOperator,
)
from linox._kernel import ArrayKernel
from linox.kernels.kernel import L2InnerProductKernel, RBFKernel

jax.config.update("jax_enable_x64", True)


class KernelType(StrEnum):
    """Available kernel types."""

    RBF = "rbf"
    SRBF = "srbf"
    L2 = "l2"


class LinearOperation(StrEnum):
    KRONECKER = "kronecker"
    PRODUCT = "product"
    SUM = "sum"


class CombinationStrategy(StrEnum):
    """Strategies for combining kernels."""

    ADDITIVE = "additive"


BIAS = 1.0
LENGTHSCALE_X = jnp.pi / 20
LENGTHSCALE_Y = jnp.pi / 20


KERNEL_REGISTRY = {
    KernelType.RBF: lambda **kwargs: RBFKernel(**kwargs),
    KernelType.L2: lambda **kwargs: L2InnerProductKernel(**kwargs),
}

LINOPS_REGISTRY = {
    LinearOperation.KRONECKER: lambda A, B: Kronecker(A, B),
    LinearOperation.PRODUCT: lambda A, B: ProductLinearOperator(A, B),
    LinearOperation.SUM: lambda A, B: AddLinearOperator(A, B),
}


@dataclass
class DimensionSpec:
    """Specification for a single dimension."""

    name: str
    kernel_type: KernelType
    kernel_params: Dict[str, Any] = field(default_factory=dict)
    is_spatial: bool = True


@dataclass
class StructureConfig:
    """Configuration for the structure of the problem."""

    spatial_dims: List[DimensionSpec]
    function_dims: List[DimensionSpec]
    channel_dims: Optional[List[DimensionSpec]] = None

    @property
    def total_spatial_dims(self) -> int:
        return len(self.spatial_dims)

    @property
    def total_function_dims(self) -> int:
        return len(self.function_dims)


@dataclass
class CombinationConfig:
    """Configuration for kernel combination."""

    strategy: CombinationStrategy
    noise_variance: float = 1e-3
    output_scale: float = 1.0


# The rest of the original tests/gp.py module contains plotting/workflow helpers
# that rely on pytest context. For helper usage, prefer helper/new_gp.py which
# contains the composable GP prior implementation and sampling utilities.

