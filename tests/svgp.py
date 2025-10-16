# svgp.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsp
import numpy as np
import optax
from jax import tree_util as jtu
from matplotlib import pyplot as plt
from new_gp import (
    CombinationConfig,
    CombinationStrategy,
    ModularGPPrior,
    Params,
    StructureConfig,
    get_structure_config_for_data,
    params_from_structure,
)
from helper.plotting import (
    generate_preprocess_data_2d,
    plot_kernel_predictions_2d,
)


def _project_tril(raw: jnp.ndarray) -> jnp.ndarray:
    """Project a matrix onto the space of lower-triangular matrices with positive diag."""
    lower = jnp.tril(raw)
    diag = jnp.diag(lower)
    diag_pos = jax.nn.softplus(diag) + 1e-6
    lower = lower - jnp.diag(diag) + jnp.diag(diag_pos)
    return lower


@jtu.register_pytree_node_class
@dataclass
class SVGPParams:
    inducing_functions: jnp.ndarray
    inducing_spatial: jnp.ndarray
    variational_mean: jnp.ndarray
    variational_tril: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.inducing_functions,
            self.inducing_spatial,
            self.variational_mean,
            self.variational_tril,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        inducing_functions, inducing_spatial, variational_mean, variational_tril = (
            children
        )
        return cls(
            inducing_functions,
            inducing_spatial,
            variational_mean,
            variational_tril,
        )

    def replace(self, **updates: jnp.ndarray) -> "SVGPParams":
        data = {
            "inducing_functions": self.inducing_functions,
            "inducing_spatial": self.inducing_spatial,
            "variational_mean": self.variational_mean,
            "variational_tril": self.variational_tril,
        }
        data.update(updates)
        return SVGPParams(**data)


class SparseVariationalGP(ModularGPPrior):
    def __init__(
        self,
        structure_config: StructureConfig,
        combination_config: CombinationConfig,
        num_inducing_functions: int,
    ) -> None:
        super().__init__(structure_config, combination_config)
        if num_inducing_functions <= 0:
            raise ValueError("num_inducing_functions must be positive.")
        self.num_inducing_functions = num_inducing_functions
        self.svgp_params: Optional[SVGPParams] = None

    @staticmethod
    def _grid_size(grid: jnp.ndarray) -> int:
        return int(np.prod(np.array(grid.shape[:-1])))

    def _dense_kernel(
        self,
        x: jnp.ndarray,
        grid: jnp.ndarray,
        params: Params,
        x2: Optional[jnp.ndarray] = None,
        grid2: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        op = self.build_kernel(x, grid, params, x2=x2, grid2=grid2)
        return op.todense()

    def initialize_inducing_points(
        self, x_train: jnp.ndarray, grid_train: jnp.ndarray, key: jnp.ndarray
    ) -> SVGPParams:
        n_train = x_train.shape[0]
        num_func = int(min(self.num_inducing_functions, n_train))
        selector = jr.choice(key, n_train, (num_func,), replace=False)
        inducing_functions = x_train[selector]
        inducing_spatial = grid_train
        grid_size = self._grid_size(inducing_spatial)
        latent_dim = num_func * grid_size
        variational_mean = jnp.zeros((latent_dim,))
        variational_tril = jnp.eye(latent_dim)
        return SVGPParams(
            inducing_functions=jnp.array(inducing_functions),
            inducing_spatial=jnp.array(inducing_spatial),
            variational_mean=variational_mean,
            variational_tril=variational_tril,
        )

    def elbo(
        self,
        params: Params,
        svgp_params: SVGPParams,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        grid_batch: jnp.ndarray,
        n_data: int,
    ) -> jnp.ndarray:
        inducing_functions = jax.lax.stop_gradient(svgp_params.inducing_functions)
        inducing_grid = jax.lax.stop_gradient(svgp_params.inducing_spatial)

        Kuu = self._dense_kernel(inducing_functions, inducing_grid, params)
        jitter = 1e-6 * jnp.eye(Kuu.shape[0], dtype=Kuu.dtype)
        Kuu = Kuu + jitter
        L_uu = jnp.linalg.cholesky(Kuu)

        Kuf = self._dense_kernel(
            inducing_functions,
            inducing_grid,
            params,
            x2=x_batch,
            grid2=grid_batch,
        )
        Kff_diag = jnp.diag(self._dense_kernel(x_batch, grid_batch, params))

        m = svgp_params.variational_mean.reshape(-1, 1)
        L_s = _project_tril(svgp_params.variational_tril)
        S = L_s @ L_s.T

        alpha = jsp.solve_triangular(L_uu, m, lower=True)
        Kuu_inv_m = jsp.solve_triangular(L_uu.T, alpha, lower=False)

        C = jsp.solve_triangular(L_uu, Kuf, lower=True)
        pred_var_reduction = jnp.sum(C * C, axis=0)

        Kuu_inv_Kuf = jsp.solve_triangular(L_uu.T, C, lower=False)
        tmp = S @ Kuu_inv_Kuf
        pred_var_increase = jnp.sum(Kuu_inv_Kuf * tmp, axis=0)

        pred_mean = (Kuf.T @ Kuu_inv_m).reshape(-1)
        y_flat = y_batch.reshape(-1)

        noise_log = params.get("noise_variance", jnp.log(1e-3))
        noise_var = jnp.exp(noise_log)

        pred_var = Kff_diag - pred_var_reduction + pred_var_increase + noise_var + 1e-6

        n_batch = y_flat.shape[0]
        scale = n_data / n_batch
        ll_term = (
            -0.5
            * scale
            * (
                jnp.sum(jnp.log(2 * jnp.pi * pred_var))
                + jnp.sum((y_flat - pred_mean) ** 2 / pred_var)
            )
        )

        Kuu_inv_S = jsp.solve_triangular(L_uu, S, lower=True)
        Kuu_inv_S = jsp.solve_triangular(L_uu.T, Kuu_inv_S, lower=False)

        diag_Ls = jnp.diag(L_s)
        kl = 0.5 * (
            jnp.trace(Kuu_inv_S)
            + jnp.sum(m * Kuu_inv_m)
            - m.shape[0]
            + 2 * jnp.sum(jnp.log(jnp.diag(L_uu)))
            - 2 * jnp.sum(jnp.log(jnp.clip(diag_Ls, a_min=1e-12)))
        )

        return ll_term - kl

    def predict_sparse(
        self,
        svgp_params: SVGPParams,
        x_test: jnp.ndarray,
        grid_test: jnp.ndarray,
        params: Params,
        full_cov: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        inducing_functions = svgp_params.inducing_functions
        inducing_grid = svgp_params.inducing_spatial

        Kuu = self._dense_kernel(inducing_functions, inducing_grid, params)
        Kuu = Kuu + 1e-6 * jnp.eye(Kuu.shape[0], dtype=Kuu.dtype)
        L_uu = jnp.linalg.cholesky(Kuu)

        Kus = self._dense_kernel(
            inducing_functions,
            inducing_grid,
            params,
            x2=x_test,
            grid2=grid_test,
        )

        m = svgp_params.variational_mean.reshape(-1, 1)
        L_s = _project_tril(svgp_params.variational_tril)
        S = L_s @ L_s.T

        alpha = jsp.solve_triangular(L_uu, m, lower=True)
        Kuu_inv_m = jsp.solve_triangular(L_uu.T, alpha, lower=False)
        pred_mean = (Kus.T @ Kuu_inv_m).reshape(-1)

        C = jsp.solve_triangular(L_uu, Kus, lower=True)
        Kuu_inv_Kus = jsp.solve_triangular(L_uu.T, C, lower=False)
        tmp = S @ Kuu_inv_Kus
        variance_correction = jnp.sum(Kuu_inv_Kus * tmp, axis=0)

        Kss = self._dense_kernel(x_test, grid_test, params)
        reduction = jnp.sum(C * C, axis=0)
        noise_var = jnp.exp(params.get("noise_variance", jnp.log(1e-3)))

        if full_cov:
            Qss = Kus.T @ jsp.solve_triangular(
                L_uu.T, jsp.solve_triangular(L_uu, Kus, lower=True), lower=False
            )
            cov = Kss - Qss + noise_var * jnp.eye(Kss.shape[0], dtype=Kss.dtype)
            cov = cov + Kus.T @ jsp.solve_triangular(
                L_uu.T,
                jsp.solve_triangular(
                    L_uu, S @ jsp.solve_triangular(L_uu.T, Kus), lower=True
                ),
                lower=False,
            )
            return pred_mean, cov

        diag_Kss = jnp.diag(Kss)
        pred_var = diag_Kss - reduction + variance_correction + noise_var + 1e-6
        return pred_mean, pred_var

    def _optimal_variational_params(
        self,
        params: Params,
        svgp_params: SVGPParams,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        grid_train: jnp.ndarray,
    ) -> SVGPParams:
        inducing_functions = svgp_params.inducing_functions
        inducing_grid = svgp_params.inducing_spatial

        Kuu = self._dense_kernel(inducing_functions, inducing_grid, params)
        jitter = 1e-6 * jnp.eye(Kuu.shape[0], dtype=Kuu.dtype)
        Kuu = Kuu + jitter
        L_uu = jnp.linalg.cholesky(Kuu)

        Kuf = self._dense_kernel(
            inducing_functions,
            inducing_grid,
            params,
            x2=x_train,
            grid2=grid_train,
        )

        y_vec = y_train.reshape(-1, 1)
        noise_log = params.get("noise_variance", jnp.log(1e-3))
        noise_var = jnp.exp(noise_log)

        eye = jnp.eye(Kuu.shape[0], dtype=Kuu.dtype)
        Kuu_inv = jsp.solve_triangular(
            L_uu.T,
            jsp.solve_triangular(L_uu, eye, lower=True),
            lower=False,
        )
        Kuu_inv_Kuf = jsp.solve_triangular(
            L_uu.T,
            jsp.solve_triangular(L_uu, Kuf, lower=True),
            lower=False,
        )

        S_inv = Kuu_inv + (1.0 / noise_var) * (Kuu_inv_Kuf @ Kuu_inv_Kuf.T)
        L_S_inv = jnp.linalg.cholesky(S_inv)
        S = jsp.solve_triangular(
            L_S_inv.T,
            jsp.solve_triangular(L_S_inv, eye, lower=True),
            lower=False,
        )
        L_s = jnp.linalg.cholesky(S + 1e-8 * eye)

        natural = (1.0 / noise_var) * (Kuu_inv_Kuf @ y_vec)
        m = S @ natural

        return svgp_params.replace(
            variational_mean=m.reshape(-1),
            variational_tril=L_s,
        )


def train_svgp(
    svgp: SparseVariationalGP,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    grid_train: jnp.ndarray,
    learning_rate: float = 1e-2,
    num_epochs: int = 200,
    seed: int = 0,
    initial_params: Optional[Params] = None,
    initial_svgp_params: Optional[SVGPParams] = None,
    closed_form: bool = False,
    train_gp: bool = False,
) -> Tuple[Dict[str, Params | SVGPParams], np.ndarray]:
    params = (
        params_from_structure(svgp.structure_config)
        if initial_params is None
        else initial_params
    )
    svgp_params = (
        svgp.initialize_inducing_points(x_train, grid_train, jr.PRNGKey(seed))
        if initial_svgp_params is None
        else initial_svgp_params
    )

    if closed_form:
        updated_svgp = svgp._optimal_variational_params(
            params, svgp_params, x_train, y_train, grid_train
        )
        svgp.svgp_params = updated_svgp
        return {"gp": params, "svgp": updated_svgp}, np.array([], dtype=float)
    svgp_params = svgp._optimal_variational_params(
        params, svgp_params, x_train, y_train, grid_train
    )

    all_params: Dict[str, Params | SVGPParams] = {"gp": params, "svgp": svgp_params}

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(all_params)

    n_total = x_train.shape[0] * SparseVariationalGP._grid_size(grid_train)
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    grid_train = jnp.asarray(grid_train)

    @jax.jit
    def loss_fn(pytree):
        return -svgp.elbo(
            pytree["gp"],
            pytree["svgp"],
            x_train,
            y_train,
            grid_train,
            n_total,
        )

    loss_history = []
    value_and_grad = jax.value_and_grad(loss_fn)

    for epoch in range(num_epochs):
        loss_value, grads = value_and_grad(all_params)
        if not train_gp:
            grads = {
                "gp": jtu.tree_map(jnp.zeros_like, grads["gp"]),
                "svgp": grads["svgp"],
            }
        updates, opt_state = optimizer.update(grads, opt_state, all_params)
        all_params = optax.apply_updates(all_params, updates)
        projected = _project_tril(all_params["svgp"].variational_tril)
        all_params = {
            "gp": all_params["gp"],
            "svgp": all_params["svgp"].replace(variational_tril=projected),
        }
        loss_history.append(float(loss_value))

    svgp.svgp_params = all_params["svgp"]
    return all_params, np.array(loss_history, dtype=float)


def prepare_heat_2d_data(
    *,
    x_range=(0.0, np.pi),
    y_range=(0.0, np.pi),
    nx: int = 12,
    ny: int = 12,
    nx_plot: Optional[int] = None,
    ny_plot: Optional[int] = None,
    T: float = 0.1,
    alpha: float = 0.5,
    n_train: int = 12,
    n_test: int = 3,
) -> Dict[str, jnp.ndarray]:
    nx_plot = nx_plot or nx
    ny_plot = ny_plot or ny

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
        N_train=n_train,
        N_test=n_test,
        nx_plot=nx_plot,
        ny_plot=ny_plot,
    )

    x_train = operator_inputs.reshape(n_train, nx, ny)
    y_train = outputs.reshape(n_train, nx, ny)
    grid_train = spatial_inputs.reshape(nx, ny, 2)

    x_test = operator_inputs_test.reshape(n_test, nx, ny)
    y_test = outputs_test.reshape(n_test, nx, ny)
    grid_test = spatial_inputs_test.reshape(nx, ny, 2)
    grid_plot = spatial_inputs_plot.reshape(nx_plot, ny_plot, 2)

    return {
        "x_train": jnp.asarray(x_train),
        "y_train": jnp.asarray(y_train),
        "grid_train": jnp.asarray(grid_train),
        "x_test": jnp.asarray(x_test),
        "y_test": jnp.asarray(y_test),
        "grid_test": jnp.asarray(grid_test),
        "grid_plot": jnp.asarray(grid_plot),
    }


def run_svgp_2d_demo() -> Dict[str, float]:
    nx = 10
    ny = 10
    nx_plot = nx
    ny_plot = ny
    n_train = 15
    n_test = 3

    data = prepare_heat_2d_data(
        nx=nx,
        ny=ny,
        nx_plot=nx_plot,
        ny_plot=ny_plot,
        n_train=n_train,
        n_test=n_test,
    )

    structure_config = get_structure_config_for_data("2D")
    combination_config = CombinationConfig(
        strategy=CombinationStrategy.ADDITIVE,
        noise_variance=1e-6,
        output_scale=1.0,
    )

    svgp = SparseVariationalGP(
        structure_config=structure_config,
        combination_config=combination_config,
        num_inducing_functions=4,
    )
    params = params_from_structure(structure_config)
    svgp_params_initial = svgp.initialize_inducing_points(
        data["x_train"], data["grid_train"], jr.PRNGKey(0)
    )

    operator_inputs_test = data["x_test"].reshape(n_test, -1)
    outputs_test = data["y_test"].reshape(n_test, -1)
    spatial_inputs_plot = data["grid_plot"].reshape(nx_plot * ny_plot, 2)
    spatial_inputs_test = data["grid_test"].reshape(nx * ny, 2)

    pred_initial, _ = svgp.predict_sparse(
        svgp_params_initial,
        data["x_test"],
        data["grid_test"],
        params,
        full_cov=False,
    )
    pred_initial = pred_initial.reshape(n_test, -1)
    rmse_initial = float(jnp.sqrt(jnp.mean((pred_initial - outputs_test) ** 2)))

    trained_cf, _ = train_svgp(
        svgp,
        data["x_train"],
        data["y_train"],
        data["grid_train"],
        seed=42,
        initial_params=params,
        initial_svgp_params=svgp_params_initial,
        closed_form=True,
    )

    trained, losses = train_svgp(
        svgp,
        data["x_train"],
        data["y_train"],
        data["grid_train"],
        learning_rate=5e-3,
        num_epochs=150,
        seed=42,
        initial_params=params,
        initial_svgp_params=svgp_params_initial,
    )

    pred_mean_cf, _ = svgp.predict_sparse(
        trained_cf["svgp"],
        data["x_test"],
        data["grid_test"],
        trained_cf["gp"],
        full_cov=False,
    )

    pred_mean, _ = svgp.predict_sparse(
        trained["svgp"],
        data["x_test"],
        data["grid_test"],
        trained["gp"],
        full_cov=False,
    )

    pred_mean = pred_mean.reshape(n_test, -1)
    pred_mean_cf = pred_mean_cf.reshape(n_test, -1)
    rmse_trained = float(jnp.sqrt(jnp.mean((pred_mean - outputs_test) ** 2)))
    rmse_cf = float(jnp.sqrt(jnp.mean((pred_mean_cf - outputs_test) ** 2)))

    fig_init, _ = plot_kernel_predictions_2d(
        pred_initial,
        outputs_test,
        operator_inputs_test,
        spatial_inputs_plot,
        spatial_inputs_test,
        nx,
        ny,
        nx_plot,
        ny_plot,
        n_test,
    )
    fig_init.suptitle("SVGP initial predictions (no optimisation)", fontsize=12)

    fig_cf, _ = plot_kernel_predictions_2d(
        pred_mean_cf,
        outputs_test,
        operator_inputs_test,
        spatial_inputs_plot,
        spatial_inputs_test,
        nx,
        ny,
        nx_plot,
        ny_plot,
        n_test,
    )
    fig_cf.suptitle("SVGP closed-form variational solution", fontsize=12)

    fig_trained, _ = plot_kernel_predictions_2d(
        pred_mean,
        outputs_test,
        operator_inputs_test,
        spatial_inputs_plot,
        spatial_inputs_test,
        nx,
        ny,
        nx_plot,
        ny_plot,
        n_test,
    )
    fig_trained.suptitle("SVGP predictions after optimisation", fontsize=12)
    plt.show()

    return {
        "final_elbo": -float(losses[-1]) if len(losses) else float("nan"),
        "rmse_closed_form": rmse_cf,
        "rmse_initial": rmse_initial,
        "rmse_trained": rmse_trained,
    }


if __name__ == "__main__":
    summary = run_svgp_2d_demo()
    print("SVGP training complete.")
    print(f"Final ELBO: {summary['final_elbo']:.4f}")
    print(f"Closed-form RMSE: {summary['rmse_closed_form']:.4f}")
    print(f"Initial RMSE: {summary['rmse_initial']:.4f}")
    print(f"Trained RMSE: {summary['rmse_trained']:.4f}")
