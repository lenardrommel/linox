# gp.py (moved from tests/gp.py)

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp

try:  # Python < 3.11 compatibility
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - only exercised on older interpreters

    class StrEnum(str, Enum):
        """Fallback StrEnum implementation for Python < 3.11."""


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
    pass
from linox import (
    AddLinearOperator,
    Kronecker,
    ProductLinearOperator,
)
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
    KernelType.RBF: RBFKernel,
    KernelType.L2: L2InnerProductKernel,
}

LINOPS_REGISTRY = {
    LinearOperation.KRONECKER: Kronecker,
    LinearOperation.PRODUCT: ProductLinearOperator,
    LinearOperation.SUM: AddLinearOperator,
}


@dataclass
class DimensionSpec:
    """Specification for a single dimension."""

    name: str
    kernel_type: KernelType
    kernel_params: dict[str, Any] = field(default_factory=dict)
    is_spatial: bool = True


@dataclass
class StructureConfig:
    """Configuration for the structure of the problem."""

    spatial_dims: list[DimensionSpec]
    function_dims: list[DimensionSpec]
    channel_dims: list[DimensionSpec] | None = None

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
