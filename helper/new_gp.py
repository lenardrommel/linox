# new_gp.py (moved from tests/new_gp.py)

import functools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

import linox as lo
from helper.gp import (
    BIAS,
    KERNEL_REGISTRY,
    LENGTHSCALE_X,
    LENGTHSCALE_Y,
    LINOPS_REGISTRY,
    CombinationConfig,
    CombinationStrategy,
    KernelType,
    LinearOperation,
)

try:
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
except ModuleNotFoundError:  # pragma: no cover - fallback when helper not on path
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
from linox import (
    AddLinearOperator,
    IsotropicAdditiveLinearOperator,
    Kronecker,
    ProductLinearOperator,
    ScaledLinearOperator,
)
from linox._kernel import ArrayKernel
from linox.kernels.kernel import L2InnerProductKernel, RBFKernel

try:  # Python < 3.11 compatibility
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - only exercised on older interpreters

    class StrEnum(str, Enum):
        """Fallback StrEnum implementation."""

        pass


jax.config.update("jax_enable_x64", True)

Params = Dict[str, Dict[str, jnp.ndarray] | jnp.ndarray]


# Serialization helpers
def _flatten_dict(d: Mapping, prefix: str = "") -> Dict[str, np.ndarray]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(_flatten_dict(v, key)) if isinstance(v, dict) else out.update({
            key: np.asarray(v)
        })
    return out


def _unflatten_dict(flat: Mapping[str, np.ndarray]) -> Dict:
    root = {}
    for k, v in flat.items():
        if k.startswith("__") and k.endswith("__"):
            continue
        parts = k.split(".")
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = jnp.array(v)
    return root


def save_params(path: Path, params: Params) -> None:
    p = Path(path).with_suffix(".npz")
    np.savez_compressed(p, **_flatten_dict(params))


def load_params(path: Path) -> Params:
    p = Path(path).with_suffix(".npz")
    with np.load(p, allow_pickle=False) as data:
        return _unflatten_dict({k: data[k] for k in data.files})


_POSITIVE_PARAM_KEYS = {"lengthscale", "bias"}


def _to_jax_array(value: Any) -> jnp.ndarray:
    if isinstance(value, jnp.ndarray):
        return value
    return jnp.array(value)


def _coerce_params(tree: Mapping[str, Any]) -> Params:
    coerced: Dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, Mapping):
            coerced[key] = _coerce_params(value)
        else:
            coerced[key] = _to_jax_array(value)
    return coerced  # type: ignore[return-value]


@dataclass
class DimensionSpec:
    kernel_type: KernelType
    kernel_params: Dict[str, Any] = field(default_factory=dict)
    is_spatial: bool = True
    name: str = ""
    scope_key: Optional[str] = None


@dataclass
class StructureConfig:
    spatial_dims: List[DimensionSpec]
    function_dims: List[DimensionSpec]
    channel_dims: Optional[List[DimensionSpec]] = None

    def __post_init__(self) -> None:
        for dim in self.spatial_dims:
            dim.scope_key = dim.scope_key or f"spatial_{dim.name}"
            dim.is_spatial = True
        for dim in self.function_dims:
            dim.scope_key = dim.scope_key or f"function_{dim.name}"
            dim.is_spatial = False
        if self.channel_dims:
            for dim in self.channel_dims:
                dim.scope_key = dim.scope_key or f"channel_{dim.name}"


@dataclass
class CombinationConfig:
    strategy: CombinationStrategy
    noise_variance: float = 1e-3
    output_scale: float = 0.10


def params_from_structure(structure: StructureConfig) -> Params:
    params: Dict[str, Any] = {}

    for dim in structure.spatial_dims:
        key = dim.scope_key or f"spatial_{dim.name}"
        defaults = {k: _to_jax_array(v) for k, v in dim.kernel_params.items()}
        if "lengthscale" not in defaults:
            init_val = jnp.pi / 20 if dim.kernel_type == KernelType.RBF else -3.0
            defaults["lengthscale"] = _to_jax_array(init_val)
        params[key] = defaults

    for dim in structure.function_dims:
        key = dim.scope_key or f"function_{dim.name}"
        defaults = {k: _to_jax_array(v) for k, v in dim.kernel_params.items()}
        if dim.kernel_type in KernelType.L2:
            defaults.setdefault("bias", _to_jax_array(-1.0))
        else:
            defaults.setdefault("lengthscale", _to_jax_array(-3.0))
        params[key] = defaults

    # Use unit output scale and small noise by default for examples
    params["output_scale"] = _to_jax_array(0.0)
    params["noise_variance"] = _to_jax_array(-6.0)
    return params  # type: ignore[return-value]


@dataclass
class ModularHParams:
    params: Params

    @classmethod
    def from_structure(cls, structure_config: StructureConfig) -> "ModularHParams":
        return cls(params=params_from_structure(structure_config))

    def to_dict(self) -> Params:
        return self.params

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> "ModularHParams":
        return cls(params=_coerce_params(dict(params)))

    def to_json(self) -> Dict[str, Any]:
        def _to_python(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {k: _to_python(v) for k, v in value.items()}
            arr = np.asarray(value)
            if arr.ndim == 0:
                return float(arr)
            return arr.tolist()

        return {k: _to_python(v) for k, v in self.params.items()}

    @classmethod
    def from_json(cls, json_params: Mapping[str, Any]) -> "ModularHParams":
        def _to_jax(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {k: _to_jax(v) for k, v in value.items()}
            return jnp.array(value)

        return cls(params={k: _to_jax(v) for k, v in json_params.items()})

    @staticmethod
    def from_kernel_config(
        kernel_config: Dict[str, Any], structure_config: StructureConfig
    ) -> "ModularHParams":
        params = params_from_structure(structure_config)

        if "spatial_ls" in kernel_config:
            for dim in structure_config.spatial_dims:
                key = dim.scope_key or f"spatial_{dim.name}"
                params[key]["lengthscale"] = _to_jax_array(kernel_config["spatial_ls"])

        if "function_bias" in kernel_config:
            for dim in structure_config.function_dims:
                key = dim.scope_key or f"function_{dim.name}"
                params[key]["bias"] = _to_jax_array(kernel_config["function_bias"])

        if "output_scale" in kernel_config:
            params["output_scale"] = _to_jax_array(kernel_config["output_scale"])
        if "sigma" in kernel_config:
            params["noise_variance"] = _to_jax_array(kernel_config["sigma"])

        return ModularHParams(params=params)

    def print_formatted(
        self, structure_config: Optional[StructureConfig] = None
    ) -> None:
        spatial_params = {
            k: v for k, v in self.params.items() if k.startswith("spatial_")
        }
        function_params = {
            k: v for k, v in self.params.items() if k.startswith("function_")
        }
        global_params = {
            k: v
            for k, v in self.params.items()
            if not (k.startswith("spatial_") or k.startswith("function_"))
        }

        def _print_block(title: str, block: Mapping[str, Any]) -> None:
            if not block:
                return
            print(title)
            for key, value in block.items():
                if isinstance(value, Mapping):
                    for sub_key, sub_val in value.items():
                        exp_val = jnp.exp(sub_val)
                        print(
                            "  "
                            f"{key.replace('_', ' ')}_{sub_key}: "
                            f"{float(sub_val):.6f} (exp: {float(exp_val):.6e})"
                        )
                else:
                    exp_val = jnp.exp(value)
                    print(
                        "  "
                        f"{key.replace('_', ' ')}: "
                        f"{float(value):.6f} (exp: {float(exp_val):.6e})"
                    )
            print("-" * 40)

        _print_block("Spatial kernels:", spatial_params)
        _print_block("Function kernels:", function_params)
        _print_block("Global parameters:", global_params)

    def save_dict(self, save_path: Path) -> None:
        save_params(Path(save_path), self.params)

    @staticmethod
    def load_dict(load_path: Path) -> "ModularHParams":
        return ModularHParams(params=load_params(Path(load_path)))


class ModularGPPrior:
    def __init__(
        self, structure_config: StructureConfig, combination_config: CombinationConfig
    ) -> None:
        self.structure_config = structure_config
        self.combination_config = combination_config
        self._params: Optional[Params] = None

    @property
    def params(self) -> Params:
        if self._params is None:
            raise ValueError("Parameters not set.")
        return self._params

    @property
    def noise_variance(self) -> jnp.ndarray:
        return self.params.get(
            "noise_variance", jnp.log(self.combination_config.noise_variance)
        )

    def set_params(self, params: Union[Params, ModularHParams]) -> None:
        if isinstance(params, ModularHParams):
            params = params.to_dict()
        self._params = _coerce_params(params)

    def _transformed_scope_params(
        self, params: Params, dim: DimensionSpec
    ) -> Dict[str, jnp.ndarray]:
        raw = params.get(dim.scope_key or "", {})
        transformed: Dict[str, jnp.ndarray] = {}
        if isinstance(raw, Mapping):
            for name, value in raw.items():
                array = _to_jax_array(value)
                if name in _POSITIVE_PARAM_KEYS:
                    transformed[name] = jnp.exp(array)
                else:
                    transformed[name] = array
        defaults = {k: _to_jax_array(v) for k, v in dim.kernel_params.items()}
        defaults.update(transformed)
        return defaults

    def _extract_coords(self, grid: Optional[jnp.ndarray]) -> List[jnp.ndarray]:
        if grid is None:
            return []

        array = jnp.asarray(grid)
        if array.ndim < 2:
            array = array[..., None]

        nd = len(self.structure_config.spatial_dims)
        if nd == 0:
            return []

        if nd == 2 and array.ndim == 3 and array.shape[-1] == 2:
            return [array[0, :, 0:1], array[:, 0, 1:2]]
        if nd == 3 and array.ndim == 4 and array.shape[-1] == 3:
            return [array[0, 0, :, 0:1], array[0, :, 0, 1:2], array[:, 0, 0, 2:3]]

        return [array[..., i : i + 1] for i in range(nd)]

    def _build_axis_kernel(
        self,
        dims: List[DimensionSpec],
        coords: List[jnp.ndarray],
        params: Params,
        coords2: Optional[List[jnp.ndarray]] = None,
    ) -> Optional[ArrayKernel]:
        if not dims:
            return None

        kernels = []
        for i, dim in enumerate(dims):
            kernel_factory = KERNEL_REGISTRY.get(dim.kernel_type)
            if kernel_factory is None:
                raise KeyError(f"Kernel type {dim.kernel_type} is not registered.")
            kernel_params = self._transformed_scope_params(params, dim)
            kernel_fn = kernel_factory(**kernel_params)
            x1 = coords[i]
            x2 = coords2[i] if coords2 is not None else None
            kernels.append(ArrayKernel(kernel_fn, x1, x2))

        return functools.reduce(Kronecker, kernels)

    def build_kernel(
        self,
        x: jnp.ndarray,
        grid: jnp.ndarray,
        params: Union[Params, ModularHParams],
        x2: Optional[jnp.ndarray] = None,
        grid2: Optional[jnp.ndarray] = None,
    ) -> Any:
        if isinstance(params, ModularHParams):
            params_dict = params.to_dict()
        else:
            params_dict = params

        function_inputs = [x] * len(self.structure_config.function_dims)
        function_inputs2 = (
            [x2] * len(self.structure_config.function_dims) if x2 is not None else None
        )
        spatial_coords = self._extract_coords(grid)
        spatial_coords2 = self._extract_coords(grid2) if grid2 is not None else None

        fop = self._build_axis_kernel(
            self.structure_config.function_dims,
            function_inputs,
            params_dict,
            function_inputs2,
        )
        sop = self._build_axis_kernel(
            self.structure_config.spatial_dims,
            spatial_coords,
            params_dict,
            spatial_coords2,
        )

        if fop and sop:
            base = Kronecker(fop, sop)
        else:
            base = fop or sop

        if base is None:
            raise ValueError("No kernels defined in structure configuration.")

        scale_raw = params_dict.get("output_scale")
        if scale_raw is None:
            scale_raw = jnp.log(self.combination_config.output_scale)
        scale = jnp.exp(_to_jax_array(scale_raw))
        kernel = ScaledLinearOperator(base, scale)

        return kernel

    def build_spatial_kernels_list(
        self, grid: jnp.ndarray, params: Optional[Union[Params, ModularHParams]] = None
    ) -> List[ArrayKernel]:
        params_dict = (
            self.params
            if params is None
            else (params.to_dict() if isinstance(params, ModularHParams) else params)
        )
        spatial_coords = self._extract_coords(grid)
        kernels: List[ArrayKernel] = []
        for i, dim in enumerate(self.structure_config.spatial_dims):
            kernel_factory = KERNEL_REGISTRY.get(dim.kernel_type)
            if kernel_factory is None:
                raise KeyError(f"Kernel type {dim.kernel_type} is not registered.")
            kernel_params = self._transformed_scope_params(params_dict, dim)
            kernel_fn = kernel_factory(**kernel_params)
            kernels.append(ArrayKernel(kernel_fn, spatial_coords[i]))
        return kernels

    def build_function_kernels_list(
        self, x: jnp.ndarray, params: Optional[Union[Params, ModularHParams]] = None
    ) -> List[ArrayKernel]:
        params_dict = (
            self.params
            if params is None
            else (params.to_dict() if isinstance(params, ModularHParams) else params)
        )
        kernels: List[ArrayKernel] = []
        for dim in self.structure_config.function_dims:
            kernel_factory = KERNEL_REGISTRY.get(dim.kernel_type)
            if kernel_factory is None:
                raise KeyError(f"Kernel type {dim.kernel_type} is not registered.")
            kernel_params = self._transformed_scope_params(params_dict, dim)
            kernel_fn = kernel_factory(**kernel_params)
            kernels.append(ArrayKernel(kernel_fn, x))
        return kernels

    def cov(
        self,
        x: jnp.ndarray,
        grid: jnp.ndarray,
    ) -> jnp.ndarray:
        K = self.build_kernel(x, grid, self.params)
        return K

    def sample_prior(
        self,
        key: jax.Array,
        x: jnp.ndarray,
        grid: jnp.ndarray,
        params: Optional[Union[Params, ModularHParams]] = None,
        size=(),
        dtype=jnp.float32,
    ) -> jnp.ndarray:
        if params is None:
            params = self.params

        K = self.build_kernel(x, grid, params)

        n_total = K.shape[0]  # This is N_functions * n_spatial_points

        if isinstance(size, int):
            n_samples = size
        elif len(size) == 0:
            n_samples = 1
        else:
            n_samples = np.prod(size)

        # Compute the square root of the kernel
        K_sqrt = lo.lsqrt(K)

        noise = jax.random.normal(key, shape=(n_total, n_samples), dtype=dtype)

        samples = K_sqrt @ noise

        # Transpose to get (n_samples, n_total)
        samples = samples.T
        samples = samples.reshape(n_samples, x.shape[0], *grid.shape[:-1])
        return samples

    def sample_posterior(
        self,
        key: jax.Array,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        grid_train: jnp.ndarray,
        x_test: jnp.ndarray,
        grid_test: jnp.ndarray,
        params: Optional[Union[Params, ModularHParams]] = None,
        size=(),
        dtype=jnp.float32,
    ) -> jnp.ndarray:
        if params is None:
            params = self.params

        # Get posterior mean and covariance using existing predict method
        pred_mean, pred_cov = self.predict(
            x_train, y_train, grid_train, x_test, grid_test, params
        )

        pred_cov_sqrt = lo.lsqrt(pred_cov)

        n_total = pred_cov.shape[0]

        if isinstance(size, int):
            n_samples = size
        elif len(size) == 0:
            n_samples = 1
        else:
            n_samples = np.prod(size)

        noise = jax.random.normal(key, shape=(n_total, n_samples), dtype=dtype)

        # Transform: samples = mean + sqrt(cov) @ noise
        samples = pred_mean[:, None] + (pred_cov_sqrt @ noise)

        # Transpose to get (n_samples, n_total)
        samples = samples.T

        # Reshape to (n_samples, n_functions, *spatial_shape)
        samples = samples.reshape(n_samples, x_test.shape[0], *grid_test.shape[:-1])

        return samples

    def predict(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        grid_train: jnp.ndarray,
        x_test: jnp.ndarray,
        grid_test: jnp.ndarray,
        params: Union[Params, ModularHParams],
    ) -> Tuple[jnp.ndarray, Any]:
        K_train = self.build_kernel(x_train, grid_train, params)
        # Add isotropic noise for numerical stability and better conditioning
        params_dict = params.to_dict() if isinstance(params, ModularHParams) else params
        noise_raw = params_dict.get(
            "noise_variance", jnp.log(self.combination_config.noise_variance)
        )
        noise_var = jnp.exp(_to_jax_array(noise_raw))
        K_train_noisy = lo.IsotropicAdditiveLinearOperator(noise_var, K_train)
        y_flat = y_train.reshape(-1, 1)
        alpha = lo.lsolve(K_train_noisy, y_flat)

        K_cross = self.build_kernel(
            x_test, grid_test, params, x2=x_train, grid2=grid_train
        )
        pred_mean = K_cross @ alpha

        K_test = self.build_kernel(x_test, grid_test, params)
        v = lo.linverse(K_train_noisy) @ K_cross.T
        pred_cov = K_test - K_cross @ v

        return pred_mean.reshape(-1), pred_cov

