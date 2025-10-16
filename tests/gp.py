# gp.py

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


@dataclass
class ModularHParams:
    """Modular hyperparameters that work with any structure.
    To load:
        hparams = ModularHParams(params={}).load_dict(path)

    """

    params: Dict[str, jnp.ndarray]

    @classmethod
    def from_structure(cls, structure_config: StructureConfig) -> "ModularHParams":
        """Initialize hyperparameters based on structure configuration."""
        params = {}

        # Initialize spatial dimension parameters
        for _, dim_spec in enumerate(structure_config.spatial_dims):
            key = f"spatial_{dim_spec.name}"
            if dim_spec.kernel_type == KernelType.RBF:
                params[key] = {"lengthscale": jnp.array(1.0)}
            else:
                params[key] = {"lengthscale": jnp.array(-3.0)}

        # Initialize function dimension parameters
        for _, dim_spec in enumerate(structure_config.function_dims):
            key = f"function_{dim_spec.name}"
            if dim_spec.kernel_type in [KernelType.L2]:
                params[key] = {"bias": jnp.array(-8.0)}
            else:
                params[key] = {"lengthscale": jnp.array(-3.0)}

        # Global parameters
        params["output_scale"] = jnp.array(-3.01)
        params["noise_variance"] = jnp.array(-1.0)

        return cls(params=params)

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return self.params

    @classmethod
    def from_dict(cls, params: Dict) -> "ModularHParams":
        """Create from dictionary."""
        return cls(params=params)

    def to_json(self) -> Dict:
        """Convert to JSON-serializable format."""
        json_params = {}
        for key, value in self.params.items():
            if isinstance(value, dict):
                json_params[key] = {k: float(v) for k, v in value.items()}
            else:
                json_params[key] = float(value)
        return json_params

    @classmethod
    def from_json(cls, json_params: Dict) -> "ModularHParams":
        """Create from JSON format."""
        params = {}
        for key, value in json_params.items():
            if isinstance(value, dict):
                params[key] = {k: jnp.array(v) for k, v in value.items()}
            else:
                params[key] = jnp.array(value)
        return cls(params=params)

    @classmethod
    def from_kernel_config(
        cls, kernel_config: Dict, structure_config: StructureConfig
    ) -> "ModularHParams":
        """Create from old KernelConfig format, converting to new modular format."""
        params = {}

        if "spatial_ls" in kernel_config:
            for dim_spec in structure_config.spatial_dims:
                key = f"spatial_{dim_spec.name}"
                params[key] = {"lengthscale": jnp.array(kernel_config["spatial_ls"])}

        if "function_bias" in kernel_config:
            for dim_spec in structure_config.function_dims:
                key = f"function_{dim_spec.name}"
                params[key] = {"bias": jnp.array(kernel_config["function_bias"])}

        if "output_scale" in kernel_config:
            params["output_scale"] = jnp.array(kernel_config["output_scale"])
        if "sigma" in kernel_config:
            params["noise_variance"] = jnp.array(kernel_config["sigma"])

        return cls(params=params)

    def print_formatted(self, structure_config: Optional[StructureConfig] = None):
        """Print hyperparameters in a formatted way.

        Args:
            structure_config: Optional structure config to provide context
        """
        print("\nOptimized hyperparameters:")
        print("-" * 40)

        # Separate parameters by type
        spatial_params = {}
        function_params = {}
        global_params = {}

        for key, value in self.params.items():
            if key.startswith("spatial_"):
                spatial_params[key] = value
            elif key.startswith("function_"):
                function_params[key] = value
            else:
                global_params[key] = value

        # Print spatial parameters
        if spatial_params:
            print("Spatial kernels:")
            for key, value in spatial_params.items():
                dim_name = key.replace("spatial_", "")
                if isinstance(value, dict):
                    for param_name, param_val in value.items():
                        exp_val = jnp.exp(param_val)
                        print(
                            f"  {dim_name}_{param_name}: {float(param_val):.6f} (exp: {float(exp_val):.6e})"
                        )
                else:
                    print(f"  {dim_name}: {float(value):.6f}")

        # Print function parameters
        if function_params:
            print("\nFunction kernels:")
            for key, value in function_params.items():
                dim_name = key.replace("function_", "")
                if isinstance(value, dict):
                    for param_name, param_val in value.items():
                        exp_val = jnp.exp(param_val)
                        print(
                            f"  {dim_name}_{param_name}: {float(param_val):.6f} (exp: {float(exp_val):.6e})"
                        )
                else:
                    print(f"  {dim_name}: {float(value):.6f}")

        # Print global parameters
        if global_params:
            print("\nGlobal parameters:")
            for key, value in global_params.items():
                if isinstance(value, dict):
                    for param_name, param_val in value.items():
                        exp_val = jnp.exp(param_val)
                        print(
                            f"  {key}_{param_name}: {float(param_val):.6f} (exp: {float(exp_val):.6e})"
                        )
                else:
                    exp_val = jnp.exp(value)
                    print(f"  {key}: {float(value):.6f} (exp: {float(exp_val):.6e})")

        print("-" * 40)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
        """Flatten nested dict using dot-separated keys and cast to numpy arrays."""
        out: Dict[str, np.ndarray] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(ModularHParams._flatten_dict(v, key))
            else:
                arr = np.asarray(v)
                out[key] = arr
        return out

    @staticmethod
    def _unflatten_dict(flat: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Reconstruct nested dict from dot-separated keys; convert to jnp."""
        root: Dict[str, Any] = {}
        for k, v in flat.items():
            if k.startswith("__") and k.endswith("__"):
                continue
            parts = k.split(".")
            cur = root
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = jnp.array(v)
        return root

    def save_dict(self, save_path: Path):
        sp = Path(save_path)
        if sp.suffix == "":
            sp = sp.with_suffix(".npz")
        flat = self._flatten_dict(self.params)
        np.savez_compressed(sp, **flat)

    def load_dict(self, load_path: Path):
        lp = Path(load_path)
        if lp.suffix == "":
            lp = lp.with_suffix(".npz")
        with np.load(lp, allow_pickle=False) as data:
            flat = {k: data[k] for k in data.files}
        params = self._unflatten_dict(flat)
        return ModularHParams(params=params)


class StructureModule:
    """Handles the structure of the problem - dimensions and channels.
    Builds appropriate Kronecker products based on the problem structure.
    """  # noqa: D205

    def __init__(self, config: StructureConfig):
        self.config = config

    def _extract_grid_coordinates(self, grid: jnp.ndarray) -> List[jnp.ndarray]:
        """Extract coordinate vectors from structured grids.

        Args:
            grid: Input grid of various formats

        Returns:
            List of coordinate vectors, one per spatial dimension
        """
        num_spatial_dims = len(self.config.spatial_dims)

        # Add dimension if needed
        if len(grid.shape) < 2:
            grid = grid[..., None]

        # For 2D structured grids (ny, nx, 2)
        if num_spatial_dims == 2 and grid.ndim == 3 and grid.shape[-1] == 2:
            x_coords = grid[0, :, 0:1]  # Shape: (nx, 1)
            y_coords = grid[:, 0, 1:2]  # Shape: (ny, 1)
            return [x_coords, y_coords]

        # For 3D structured grids (nz, ny, nx, 3)
        elif num_spatial_dims == 3 and grid.ndim == 4 and grid.shape[-1] == 3:
            x_coords = grid[0, 0, :, 0:1]  # Shape: (nx, 1)
            y_coords = grid[0, :, 0, 1:2]  # Shape: (ny, 1)
            z_coords = grid[:, 0, 0, 2:3]  # Shape: (nz, 1)
            return [x_coords, y_coords, z_coords]

        # For 1D or general case
        else:
            return [grid[..., i : i + 1] for i in range(num_spatial_dims)]

    def build_kernel(
        self,
        x: jnp.ndarray,
        grid: jnp.ndarray,
        params: dict,
        x2: jnp.ndarray | None = None,
        grid2: jnp.ndarray | None = None,
    ) -> Any:
        if isinstance(params, ModularHParams):
            params = params.to_dict()

        # Build function kernel
        function_kernel = None
        if self.config.function_dims:
            for i, dim_spec in enumerate(self.config.function_dims):
                kernel_params = self._get_kernel_params(
                    params, f"function_{dim_spec.name}", dim_spec
                )
                kernel_fn = KERNEL_REGISTRY[dim_spec.kernel_type](**kernel_params)

                if x2 is not None:
                    array_kernel = ArrayKernel(kernel_fn, x, x2)
                else:
                    array_kernel = ArrayKernel(kernel_fn, x)

                if i == 0:
                    function_kernel = array_kernel
                else:
                    linop_fn = LINOPS_REGISTRY[LinearOperation.KRONECKER]
                    function_kernel = linop_fn(function_kernel, array_kernel)

        # Validate function kernel shape
        if function_kernel is not None:
            expected_shape = (x.shape[0], x2.shape[0] if x2 is not None else x.shape[0])
            assert function_kernel.shape == expected_shape, (
                f"Function kernel shape mismatch: expected {expected_shape}, got {function_kernel.shape}"
            )

        # Build spatial kernel
        spatial_kernel = None
        if self.config.spatial_dims:
            # Extract coordinate vectors from grids
            grid1_coords = self._extract_grid_coordinates(grid)
            grid2_coords = (
                self._extract_grid_coordinates(grid2) if grid2 is not None else None
            )

            # Build spatial kernels for each dimension
            for i, dim_spec in enumerate(self.config.spatial_dims):
                kernel_params = self._get_kernel_params(
                    params, f"spatial_{dim_spec.name}", dim_spec
                )
                kernel_fn = KERNEL_REGISTRY[dim_spec.kernel_type](**kernel_params)

                if grid2_coords is not None:
                    array_kernel = ArrayKernel(
                        kernel_fn, grid1_coords[i], grid2_coords[i]
                    )
                else:
                    array_kernel = ArrayKernel(kernel_fn, grid1_coords[i])

                if i == 0:
                    spatial_kernel = array_kernel
                else:
                    linop_fn = LINOPS_REGISTRY[LinearOperation.KRONECKER]
                    spatial_kernel = linop_fn(spatial_kernel, array_kernel)

        # Combine function and spatial kernels
        if function_kernel is not None and spatial_kernel is not None:
            return Kronecker(function_kernel, spatial_kernel)
        elif spatial_kernel is not None:
            return spatial_kernel
        elif function_kernel is not None:
            return function_kernel
        else:
            raise ValueError("No kernels specified in config")

    def _get_kernel_params(
        self, params: dict, key: str, dim_spec: DimensionSpec
    ) -> dict:
        """Extract and transform kernel parameters."""
        if isinstance(params, ModularHParams):
            params = params.to_dict()
        if key in params:
            raw_params = params[key]
            transformed = {}
            for k, v in raw_params.items():
                if k in ["lengthscale", "bias"]:
                    transformed[k] = jnp.exp(v)
                else:
                    transformed[k] = v
            return transformed
        return dim_spec.kernel_params


class CombinationModule:
    """Handles combination of kernels using various strategies."""

    def __init__(self, config: CombinationConfig) -> None:
        self.config = config

    def apply(self, kernel: Any, params: dict) -> Any:
        """Apply combination strategy (scaling and noise).

        Args:
            kernel: Base kernel to transform
            params: Parameter dictionary
        """
        if isinstance(params, ModularHParams):
            params = params.to_dict()
        output_scale = jnp.exp(
            params.get("output_scale", jnp.log(self.config.output_scale))
        )
        kernel = ScaledLinearOperator(kernel, output_scale)

        return kernel


class ModularGPPrior:
    """Modular GP Prior that uses Structure and Combination modules.
    Compatible with the old optimization interface.
    """  # noqa: D205

    def __init__(  # noqa: D107
        self, structure_config: StructureConfig, combination_config: CombinationConfig
    ) -> None:
        self.structure = StructureModule(structure_config)
        self.combination = CombinationModule(combination_config)
        self.structure_config = structure_config
        self.combination_config = combination_config
        self._params = None

    def build_kernel_modular(
        self,
        x: jnp.ndarray,
        grid: jnp.ndarray,
        params: dict,
        x2: jnp.ndarray | None = None,
        grid2: jnp.ndarray | None = None,
    ) -> Any:
        """Build and scale kernel (auto or cross).

        Args:
            x: First function data
            grid: First spatial grid
            params: Hyperparameters
            x2: Second function data (optional)
            grid2: Second spatial grid (optional)

        Returns:
            Scaled kernel operator
        """
        base_kernel = self.structure.build_kernel(x, grid, params, x2, grid2)
        final_kernel = self.combination.apply(base_kernel, params)
        return final_kernel

    def predict(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        grid_train: jnp.ndarray,
        x_test: jnp.ndarray,
        grid_test: jnp.ndarray,
        params: dict,
        compute_covariance: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Compute GP predictions.

        Args:
            x_train: Training functions (N_train, ...)
            y_train: Training outputs (N_train * n_spatial,)
            grid_train: Training spatial grid
            x_test: Test functions (N_test, ...)
            grid_test: Test spatial grid (can be different resolution)
            params: Hyperparameters
            compute_covariance: Whether to compute predictive covariance

        Returns:
            (pred_mean, pred_cov) where:
            - pred_mean: Predictive mean
            - pred_cov: Predictive covariance (optional)
        """
        # Build training kernel with noise
        K_train = self.build_kernel_modular(x_train, grid_train, params)
        noise_var = jnp.exp(params.get("noise_variance", jnp.log(1e-6)))
        K_train_noisy = IsotropicAdditiveLinearOperator(noise_var, K_train)

        # Solve for coefficients
        y_flat = y_train.reshape(-1, 1)
        alpha = lo.lsolve(K_train_noisy, y_flat)

        # Build cross-kernel
        K_cross = self.build_kernel_modular(
            x_test, grid_test, params, x_train, grid_train
        )

        # Compute predictive mean
        pred_mean = K_cross @ alpha

        # Compute predictive covariance if requested
        pred_cov = None
        if compute_covariance:
            K_test = self.build_kernel_modular(x_test, grid_test, params)
            v = lo.linverse(K_train_noisy) @ K_cross.T
            pred_cov = K_test - K_cross @ v

        return pred_mean.reshape(-1), pred_cov

    def cov(self, x: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
        """Compute the covariance matrix."""
        if self._params is None:
            raise ValueError("Parameters not set.")
        kernel = self.build_kernel_modular(x, grid, self._params)
        return kernel

    def set_params(self, params: dict) -> None:
        """Set parameters for the prior."""
        self._params = params


def get_structure_config_for_data(data_name) -> StructureConfig:
    if "1D" in str(data_name) or "1d" in str(data_name):
        return StructureConfig(
            spatial_dims=[DimensionSpec(name="x", kernel_type=KernelType.RBF)],
            function_dims=[DimensionSpec(name="u", kernel_type=KernelType.L2)],
        )
    elif "2D" in str(data_name) or "2d" in str(data_name):
        return StructureConfig(
            spatial_dims=[
                DimensionSpec(name="x", kernel_type=KernelType.RBF),
                DimensionSpec(name="y", kernel_type=KernelType.RBF),
            ],
            function_dims=[
                DimensionSpec(name="u", kernel_type=KernelType.L2),
            ],
        )
    else:
        raise ValueError(f"Unknown data name: {data_name}")


def solve_1d(
    operator_inputs,  # -----------
    spatial_inputs,  # train_loader
    outputs,  # -----------
    operator_inputs_test,  # -----------
    spatial_inputs_test,  # test_loader
    outputs_test,  # -----------
    spatial_inputs_plot,  # x_plot
    nx,
    nx_plot,
    N_train,
    N_test,
):
    structure_config = get_structure_config_for_data("1D")

    combination_config = CombinationConfig(
        strategy=CombinationStrategy.ADDITIVE, noise_variance=1e-6, output_scale=1.0
    )

    prior = ModularGPPrior(structure_config, combination_config)

    hparams = ModularHParams.from_structure(structure_config)

    K_train = prior.build_kernel_modular(
        operator_inputs, spatial_inputs, hparams.to_dict()
    )

    outputs_flat = outputs.flatten()
    alpha = lo.lpinverse(K_train) @ outputs_flat
    K_cross = prior.build_kernel_modular(
        operator_inputs_test,
        spatial_inputs_plot,
        hparams.to_dict(),
        operator_inputs,
        spatial_inputs,
    )

    pred_mean = K_cross @ alpha
    # pred_mean = pred_mean.reshape(N_test, ny_plot, nx_plot)
    pred_mean_flat = pred_mean.reshape(N_test, -1)

    K_test = prior.build_kernel_modular(
        operator_inputs_test, spatial_inputs_plot, hparams.to_dict()
    )
    v = lo.lpinverse(K_train) @ K_cross.T
    pred_cov = K_test - K_cross @ v
    pred_cov = pred_cov.todense().reshape(N_test, nx_plot, N_test, nx_plot)
    return pred_mean_flat, pred_cov, alpha


def solve_2d(
    operator_inputs,  # -----------
    spatial_inputs,  # train_loader
    outputs,  # -----------
    operator_inputs_test,  # -----------
    spatial_inputs_test,  # test_loader
    outputs_test,  # -----------
    spatial_inputs_plot,  # x_plot
    nx,
    ny,
    nx_plot,
    ny_plot,
    N_train,
    N_test,
):
    structure_config = get_structure_config_for_data("2D")

    combination_config = CombinationConfig(
        strategy=CombinationStrategy.ADDITIVE, noise_variance=1e-6, output_scale=1.0
    )

    prior = ModularGPPrior(structure_config, combination_config)

    hparams = ModularHParams.from_structure(structure_config)

    K_train = prior.build_kernel_modular(
        operator_inputs, spatial_inputs, hparams.to_dict()
    )
    # outputs_flat = outputs.reshape(N_train, ny, nx)
    outputs_flat = outputs.flatten()
    alpha = lo.lpinverse(K_train) @ outputs_flat
    K_cross = prior.build_kernel_modular(
        operator_inputs_test,
        spatial_inputs_plot,
        hparams.to_dict(),
        operator_inputs,
        spatial_inputs,
    )

    pred_mean = K_cross @ alpha
    # pred_mean = pred_mean.reshape(N_test, ny_plot, nx_plot)
    pred_mean_flat = pred_mean.reshape(N_test, -1)

    K_test = prior.build_kernel_modular(
        operator_inputs_test, spatial_inputs_plot, hparams.to_dict()
    )
    v = lo.linverse(K_train) @ K_cross.T
    pred_cov = K_test - K_cross @ v
    pred_cov = pred_cov.todense().reshape(
        N_test, ny_plot * nx_plot, N_test, ny_plot * nx_plot
    )
    return pred_mean_flat, pred_cov, alpha


def run_1d():
    x_range = (0, np.pi)
    nx = 50
    nx_plot = 100
    T = 0.1
    alpha = 0.5
    N_train = 20
    N_test = 3
    (
        operator_inputs,
        spatial_inputs,
        outputs,
        operator_inputs_test,
        spatial_inputs_test,
        outputs_test,
        spatial_inputs_plot,
    ) = generate_preprocess_data_1d(
        x_range=x_range,
        nx=nx,
        T=T,
        alpha=alpha,
        N_train=N_train,
        N_test=N_test,
        nx_plot=nx_plot,
    )

    operator_inputs = operator_inputs.reshape(N_train, nx)
    spatial_inputs = spatial_inputs.reshape(nx, 1)
    operator_inputs_test = operator_inputs_test.reshape(N_test, nx)
    spatial_inputs_test = spatial_inputs_test.reshape(nx, 1)
    spatial_inputs_plot = spatial_inputs_plot.reshape(nx_plot, 1)

    pred_mean, pred_cov, alpha = solve_1d(
        operator_inputs,
        spatial_inputs,
        outputs,
        operator_inputs_test,
        spatial_inputs_test,
        outputs_test,
        spatial_inputs_plot,
        nx,
        nx_plot,
        N_train,
        N_test,
    )

    fig1, axes1 = plot_kernel_predictions_1d(
        pred_mean,
        pred_cov,
        outputs_test,
        operator_inputs_test,
        spatial_inputs_plot,
        spatial_inputs_test,
        nx,
        nx_plot,
        N_test,
    )

    plt.show()


def run_2d():
    x_range = (0, np.pi)
    y_range = (0, np.pi)
    nx = 15
    ny = 15
    nx_plot = 25
    ny_plot = 25
    T = 0.1
    alpha = 0.5
    N_train = 25
    N_test = 3
    (
        operator_inputs,
        spatial_inputs,
        outputs,
        operator_inputs_test,
        spatial_inputs_test,
        outputs_test,
        spatial_inputs_plot,
    ) = generate_preprocess_data_2d(
        x_range=x_range,
        y_range=y_range,
        nx=nx,
        ny=ny,
        T=T,
        alpha=alpha,
        N_train=N_train,
        N_test=N_test,
        nx_plot=nx_plot,
        ny_plot=ny_plot,
    )

    operator_inputs = operator_inputs.reshape(N_train, nx, ny)
    spatial_inputs = spatial_inputs.reshape(nx, ny, 2)
    operator_inputs_test = operator_inputs_test.reshape(N_test, nx, ny)
    spatial_inputs_test = spatial_inputs_test.reshape(nx, ny, 2)
    spatial_inputs_plot = spatial_inputs_plot.reshape(nx_plot, ny_plot, 2)

    pred_mean, pred_cov, alpha = solve_2d(
        operator_inputs,
        spatial_inputs,
        outputs,
        operator_inputs_test,
        spatial_inputs_test,
        outputs_test,
        spatial_inputs_plot,
        nx,
        ny,
        nx_plot,
        ny_plot,
        N_train,
        N_test,
    )

    fig1, axes1 = plot_kernel_predictions_2d(
        pred_mean,
        outputs_test,
        operator_inputs_test,
        spatial_inputs_plot,
        spatial_inputs_test,
        nx,
        ny,
        nx_plot,
        ny_plot,
        N_test,
    )
    fig2, axes2, errors, rel_errors = plot_error_analysis_2d(
        pred_mean,
        outputs_test,
        spatial_inputs_plot,
        spatial_inputs_test,
        nx,
        ny,
        nx_plot,
        ny_plot,
        N_test,
    )
    fig3, axes3 = plot_uncertainty_2d(
        pred_mean, pred_cov, spatial_inputs_plot, nx_plot, ny_plot, N_test
    )

    fig4, axes4 = plot_marginal_analysis_2d(
        pred_mean,
        outputs_test,
        operator_inputs_test,
        spatial_inputs_plot,
        spatial_inputs_test,
        nx,
        ny,
        nx_plot,
        ny_plot,
        N_test,
    )

    fig5, axes5 = plot_profile_comparison_2d(
        pred_mean,
        outputs_test,
        spatial_inputs_plot,
        spatial_inputs_test,
        nx,
        ny,
        nx_plot,
        ny_plot,
        N_test,
        sample_idx=0,
    )
    plt.show()


if __name__ == "__main__":
    # run_1d()
    run_2d()
