# test_gp.py
#
# Compatibility wrapper exposing the public helpers that used to live in this
# module. The actual implementations now live under the ``helper`` package.

from __future__ import annotations

import jax
import jax.numpy as jnp

try:
    from helper.data import Heat1dDataGenerator, Heat2dDataGenerator
    from helper.plotting import (
        generate_preprocess_data_1d,
        generate_preprocess_data_2d,
        plot_error_analysis_2d,
        plot_kernel_predictions_1d,
        plot_kernel_predictions_2d,
        plot_marginal_analysis_2d,
        plot_profile_comparison_2d,
        plot_samples_2d,
        plot_uncertainty_2d,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when helper not installed
    from tests.helper.data import (  # type: ignore[no-redef]
        Heat1dDataGenerator,
        Heat2dDataGenerator,
    )
    from tests.helper.plotting import (  # type: ignore[no-redef]
        generate_preprocess_data_1d,
        generate_preprocess_data_2d,
        plot_error_analysis_2d,
        plot_kernel_predictions_1d,
        plot_kernel_predictions_2d,
        plot_marginal_analysis_2d,
        plot_profile_comparison_2d,
        plot_samples_2d,
        plot_uncertainty_2d,
    )
from linox.kernels.kernel import L2InnerProductKernel, RBFKernel

jax.config.update("jax_enable_x64", True)

BIAS = 1.0
LENGTHSCALE_X = jnp.pi / 20
LENGTHSCALE_Y = jnp.pi / 20

__all__ = [
    "BIAS",
    "LENGTHSCALE_X",
    "LENGTHSCALE_Y",
    "Heat1dDataGenerator",
    "Heat2dDataGenerator",
    "generate_preprocess_data_1d",
    "generate_preprocess_data_2d",
    "plot_kernel_predictions_1d",
    "plot_kernel_predictions_2d",
    "plot_samples_2d",
    "plot_error_analysis_2d",
    "plot_uncertainty_2d",
    "plot_marginal_analysis_2d",
    "plot_profile_comparison_2d",
    "L2InnerProductKernel",
    "RBFKernel",
]
