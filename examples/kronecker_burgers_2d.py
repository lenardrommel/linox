#!/usr/bin/env python3
"""Kronecker GP for 2D Burgers Equation with Optimization.

This example demonstrates:
1. 4D Kronecker GP: K = K_function ⊗ K_x ⊗ K_y ⊗ K_t
2. Hyperparameter optimization using optax LBFGS
3. Differentiable log-likelihood using eigendecomposition for 4D tensor structure

Key linox operations demonstrated:
    - lo.Kronecker: 4-level nested Kronecker structure
    - lo.Matrix, lo.Diagonal
    - lo.IsotropicAdditiveLinearOperator
    - lo.lsolve: Linear system solve
    - lo.validate: Operator validation
"""

from __future__ import annotations

import time
from typing import NamedTuple

import exponax as ex
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from optax import tree_utils as otu

import linox as lo

# Enable float64
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Parameter Handling
# =============================================================================


class GPParams(NamedTuple):
    """Raw (unconstrained) GP hyperparameters."""

    ls_func_raw: jax.Array
    ls_x_raw: jax.Array
    ls_y_raw: jax.Array
    ls_t_raw: jax.Array
    noise_var_raw: jax.Array
    output_scale_raw: jax.Array


def sigmoid_transform(raw: jax.Array, low: float, high: float) -> jax.Array:
    """Transform raw parameter to bounded range (low, high)."""
    return low + (high - low) * jax.nn.sigmoid(raw)


def inverse_sigmoid_transform(value: float, low: float, high: float) -> jax.Array:
    """Transform bounded value to raw parameter."""
    x = (value - low) / (high - low)
    return jnp.log(x / (1 - x))


def init_params(
    ls_func: float = 1.0,
    ls_x: float = 0.5,
    ls_y: float = 0.5,
    ls_t: float = 0.1,
    noise_var: float = 0.1,
    output_scale: float = 0.1,
) -> GPParams:
    """Initialize raw parameters."""
    return GPParams(
        ls_func_raw=inverse_sigmoid_transform(ls_func, 0.01, 5.0),
        ls_x_raw=inverse_sigmoid_transform(ls_x, 0.01, 5.0),
        ls_y_raw=inverse_sigmoid_transform(ls_y, 0.01, 5.0),
        ls_t_raw=inverse_sigmoid_transform(ls_t, 0.01, 10.0),
        noise_var_raw=inverse_sigmoid_transform(noise_var, 1e-8, 1.0),
        output_scale_raw=inverse_sigmoid_transform(output_scale, 0.01, 10.0),
    )


def transform_params(params: GPParams) -> dict:
    """Transform raw parameters to constrained values."""
    return {
        "ls_func": sigmoid_transform(params.ls_func_raw, 0.01, 10.0),
        "ls_x": sigmoid_transform(params.ls_x_raw, 0.01, 10.0),
        "ls_y": sigmoid_transform(params.ls_y_raw, 0.01, 10.0),
        "ls_t": sigmoid_transform(params.ls_t_raw, 0.01, 10.0),
        "noise_var": sigmoid_transform(params.noise_var_raw, 1e-8, 1.0),
        "output_scale": sigmoid_transform(params.output_scale_raw, 0.01, 100.0),
    }


# =============================================================================
# Data Generation (exponax 2D)
# =============================================================================


def generate_burgers_2d_data(
    num_points: int = 16,
    n_train: int = 20,
    n_test: int = 3,
    n_time_points: int = 5,
    diffusivity: float = 0.1,
    seed: int = 42,
) -> dict:
    """Generate training and test data from the 2D Burgers equation."""
    key = random.PRNGKey(seed)

    dt = 0.005
    stepper = ex.stepper.Burgers(
        num_spatial_dims=2,
        domain_extent=2 * jnp.pi,
        num_points=num_points,
        dt=dt,
        diffusivity=diffusivity,
        convection_scale=1.0,
    )

    ic_gen = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=2,
        cutoff=2,
        max_one=True,
    )

    subsample = 1
    n_steps = n_time_points * subsample
    rollout_fn = ex.rollout(stepper, n_steps, include_init=True)

    def generate_sample(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key1, key2 = random.split(key)
        u0_x = ic_gen(num_points, key=key1)
        u0_y = ic_gen(num_points, key=key2)
        u0 = jnp.concatenate([u0_x, u0_y], axis=0)

        trajectory = rollout_fn(u0)
        trajectory = trajectory[::subsample, 0, :, :]
        return u0[0], trajectory

    # Training
    train_keys = random.split(key, n_train + 1)
    key = train_keys[0]
    train_keys = train_keys[1:]

    u0_train_list, u_train_list = [], []
    for k in train_keys:
        u0, traj = generate_sample(k)
        u0_train_list.append(u0)
        u_train_list.append(traj)

    u0_train = jnp.stack(u0_train_list)
    u_train = jnp.stack(u_train_list)

    # Test
    test_keys = random.split(key, n_test)
    u0_test_list, u_test_list = [], []
    for k in test_keys:
        u0, traj = generate_sample(k)
        u0_test_list.append(u0)
        u_test_list.append(traj)

    u0_test = jnp.stack(u0_test_list)
    u_test = jnp.stack(u_test_list)

    grid = ex.make_grid(2, 2 * jnp.pi, num_points)
    x = grid[0, :, 0]
    y = grid[1, 0, :]
    t = jnp.arange(n_time_points + 1) * dt * subsample

    return {
        "u0_train": u0_train, "u_train": u_train,
        "u0_test": u0_test, "u_test": u_test,
        "x": x, "y": y, "t": t,
    }


# =============================================================================
# Kernels (Jittable)
# =============================================================================


def rbf_kernel(X1: jax.Array, X2: jax.Array, lengthscale: jax.Array) -> jax.Array:
    sq_dists = (X1[:, None] - X2[None, :]) ** 2
    return jnp.exp(-0.5 * sq_dists / lengthscale**2)


def function_kernel_2d(U1: jax.Array, U2: jax.Array, lengthscale: jax.Array) -> jax.Array:
    """Linear/Dot Product kernel (L2 inner product).
    
    k(u, v) = <u, v> / lengthscale^2
    """
    # Flatten spatial dims: (n, nx, ny) -> (n, nx*ny)
    U1_flat = U1.reshape(U1.shape[0], -1)
    U2_flat = U2.reshape(U2.shape[0], -1)
    
    # Compute pairwise inner products
    dot_prod = jnp.matmul(U1_flat, U2_flat.T)
    
    # Normalize by lengthscale (acting as a variance/scale factor combined with output_scale)
    return dot_prod / lengthscale**2


def build_kronecker_kernel_components(
    u0_train: jax.Array,
    x: jax.Array,
    y: jax.Array,
    t: jax.Array,
    ls_func: jax.Array,
    ls_x: jax.Array,
    ls_y: jax.Array,
    ls_t: jax.Array,
    output_scale: jax.Array,
    jitter: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build component kernels for 4D Kronecker structure."""
    n_func = u0_train.shape[0]
    n_x = x.shape[0]
    n_y = y.shape[0]
    n_t = t.shape[0]

    K_func = function_kernel_2d(u0_train, u0_train, ls_func) + jitter * jnp.eye(n_func)
    # Apply output_scale
    K_func = output_scale * K_func
    K_x = rbf_kernel(x, x, ls_x) + jitter * jnp.eye(n_x)
    K_y = rbf_kernel(y, y, ls_y) + jitter * jnp.eye(n_y)
    K_t = rbf_kernel(t, t, ls_t) + jitter * jnp.eye(n_t)

    return K_func, K_x, K_y, K_t


# =============================================================================
# Optimization (Differentiable NLL)
# =============================================================================


def negative_log_likelihood(
    params: GPParams,
    u0_train: jax.Array,
    u_train: jax.Array,
    x: jax.Array,
    y: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """Compute NLL using 4D Kronecker eigendecomposition."""
    p = transform_params(params)

    # Build components
    K_func, K_x, K_y, K_t = build_kronecker_kernel_components(
        u0_train, x, y, t, p["ls_func"], p["ls_x"], p["ls_y"], p["ls_t"], p["output_scale"]
    )

    n_samples = u_train.size
    n_func, n_t, n_x, n_y = u_train.shape

    # Eigendecomposition of components
    # Add small perturbation to break degeneracy for stable gradients
    1e-6 * jnp.linspace(0, 1, n_x)  # Symmetry breaking
    eig_f, U_f = jnp.linalg.eigh(K_func + 1e-6 * jnp.eye(n_func))
    eig_x, U_x = jnp.linalg.eigh(K_x + jnp.diag(1e-6 * jnp.linspace(0, 1, n_x)))
    eig_y, U_y = jnp.linalg.eigh(K_y + jnp.diag(1e-6 * jnp.linspace(0, 1, n_y)))
    eig_t, U_t = jnp.linalg.eigh(K_t + jnp.diag(1e-6 * jnp.linspace(0, 1, n_t)))

    # Compute Kronecker eigenvalues + noise
    # λ = λ_f ⊗ λ_x ⊗ λ_y ⊗ λ_t
    eig_spatial = jnp.outer(eig_x, eig_y).ravel()  # (nx*ny,)
    eig_st = jnp.outer(eig_spatial, eig_t).ravel() # (nx*ny*nt,)
    eig_kron = jnp.outer(eig_f, eig_st).ravel()    # (nf*nx*ny*nt,)
    eig_noisy = eig_kron + p["noise_var"]

    # Transform y to eigenvector basis: U^T @ y
    # Kronecker ordering: func ⊗ x ⊗ y ⊗ t
    # Data is (n_func, n_t, n_x, n_y). Need to permute to match kron for reshaping?
    # Actually, let's keep it as tensor and contract properly.
    # Target ordering for eigenvalues: f, x, y, t (from outer products above)
    # y_tensor should be permuted to (n_func, n_x, n_y, n_t)
    y_tensor = u_train.transpose(0, 2, 3, 1)  # (f, x, y, t)

    # Contract each mode with U^T
    # Mode f (0):
    proj = jnp.einsum("if,f...->i...", U_f.T, y_tensor)
    # Mode x (1):
    proj = jnp.einsum("jx,ij...->ij...", U_x.T, proj)
    # Mode y (2):
    proj = jnp.einsum("ky,ijk...->ijk...", U_y.T, proj)
    # Mode t (3):
    proj = jnp.einsum("lt,ijkl->ijkl", U_t.T, proj)

    # Flatten (should match ordering f, x, y, t)
    proj_flat = proj.ravel()

    # Quadratic term
    quad = 0.5 * jnp.sum(proj_flat**2 / eig_noisy)

    # Log determinant
    logdet = 0.5 * jnp.sum(jnp.log(eig_noisy))

    # Constant
    const = 0.5 * n_samples * jnp.log(2 * jnp.pi)

    return quad + logdet + const


def predict(
    params: GPParams,
    u0_train: jax.Array,
    u_train: jax.Array,
    u0_test: jax.Array,
    x: jax.Array,
    y: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """Predict for 2D Burgers."""
    p = transform_params(params)
    n_test = u0_test.shape[0]
    n_x, n_y, n_t = x.shape[0], y.shape[0], t.shape[0]
    jitter = 1e-6

    # Build K_train as LO
    K_f, K_x_m, K_y_m, K_t_m = build_kronecker_kernel_components(
        u0_train, x, y, t, p["ls_func"], p["ls_x"], p["ls_y"], p["ls_t"], p["output_scale"]
    )
    
    K_kron = lo.Kronecker(
        lo.Matrix(K_f),
        lo.Kronecker(lo.Matrix(K_x_m), lo.Kronecker(lo.Matrix(K_y_m), lo.Matrix(K_t_m)))
    )
    K_noisy = lo.IsotropicAdditiveLinearOperator(p["noise_var"], K_kron)

    # Solve alpha
    # Data is (n_func, n_t, n_x, n_y).
    # Kronecker ordering is f, x, y, t.
    y_flat = u_train.transpose(0, 2, 3, 1).reshape(-1)
    alpha = lo.lsolve(K_noisy, y_flat[:, None])[:, 0]

    # Cross kernels
    K_f_cross = function_kernel_2d(u0_test, u0_train, p["ls_func"])
    K_f_cross = p["output_scale"] * K_f_cross
    
    # We reuse spatial/temporal kernels (assuming test pts same as train pts for grid)
    # If strictly consistent, we should rebuild K_xx, K_yy, K_tt. 
    # Here we assume prediction is on same grid.
    K_x_cross = rbf_kernel(x, x, p["ls_x"]) + jitter * jnp.eye(n_x)
    K_y_cross = rbf_kernel(y, y, p["ls_y"]) + jitter * jnp.eye(n_y)
    K_t_cross = rbf_kernel(t, t, p["ls_t"]) + jitter * jnp.eye(n_t)

    K_cross = lo.Kronecker(
        lo.Matrix(K_f_cross),
        lo.Kronecker(lo.Matrix(K_x_cross), lo.Kronecker(lo.Matrix(K_y_cross), lo.Matrix(K_t_cross)))
    )

    # Predict
    pred_flat = K_cross @ alpha
    
    # Reshape: (n_test, nx, ny, nt) -> (n_test, nt, nx, ny)
    pred_reshaped = pred_flat.reshape(n_test, n_x, n_y, n_t).transpose(0, 3, 1, 2)
    return pred_reshaped


# =============================================================================
# Plotting
# =============================================================================


def plot_2d_results(
    u0_test: jax.Array,
    u_test: jax.Array,
    u_pred: jax.Array,
    t_idx: int = -1,  # Which time step to show
    save_path: str | None = None,
) -> None:
    """Plot 2D Burgers results at a specific time."""
    n_show = min(2, u0_test.shape[0])

    fig, axes = plt.subplots(3, n_show, figsize=(5 * n_show, 12))
    if n_show == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_show):
        # Initial condition
        im = axes[0, i].imshow(np.asarray(u0_test[i]), cmap="RdBu_r")
        axes[0, i].set_title(f"IC {i+1}: u(x, y, 0)")
        fig.colorbar(im, ax=axes[0, i])

        # True at t
        im = axes[1, i].imshow(np.asarray(u_test[i, t_idx]), cmap="RdBu_r")
        axes[1, i].set_title(f"True u(x, y, t={t_idx})")
        fig.colorbar(im, ax=axes[1, i])

        # Predicted at t
        im = axes[2, i].imshow(np.asarray(u_pred[i, t_idx]), cmap="RdBu_r")
        axes[2, i].set_title(f"Pred u(x, y, t={t_idx})")
        fig.colorbar(im, ax=axes[2, i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()


def run_optimization(
    params: GPParams,
    u0_train: jax.Array,
    u_train: jax.Array,
    x: jax.Array,
    y: jax.Array,
    t: jax.Array,
    max_iter: int = 50,
) -> tuple[GPParams, list]:
    """Run optimization loop."""
    print("\n--- Optimization (4D) ---")
    
    @jax.jit
    def loss_fn(p):
        return negative_log_likelihood(p, u0_train, u_train, x, y, t)
    
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    
    # Compile
    print("  Compiling JIT functions...")
    s = time.time()
    _ = value_and_grad_fn(params)
    print(f"  ✓ Compilation done in {time.time() - s:.2f}s")

    optimizer = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(max_backtracking_steps=10))
    state = optimizer.init(params)
    
    
    history = []
    print(f"  {'Iter':>4} | {'NLL':>12} | {'Grad':>12} | {'scale':>8}")

    for i in range(max_iter):
        loss, grad = value_and_grad_fn(params)
        gnorm = otu.tree_norm(grad)
        history.append(float(loss))
        
        if i % 10 == 0 or i == max_iter-1:
            p = transform_params(params)
            print(f"  {i:4d} | {float(loss):12.4e} | {float(gnorm):12.4e} | {float(p['output_scale']):8.4f}")

        if gnorm < 1e-4:
            print("  ✓ Converged")
            break

        updates, state = optimizer.update(grad, state, params, value=loss, grad=grad, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)

    return params, history


# =============================================================================
# Main
# =============================================================================


def main():
    print("="*60)
    print(" 2D Kronecker Burgers Optimization")
    print("="*60)

    # Data
    print("Generating data...")
    # Smaller grid for optimization demo speed
    data = generate_burgers_2d_data(
        num_points=32, n_train=50, n_test=2, n_time_points=11
    )
    u0_train, u_train = data["u0_train"], data["u_train"]
    u0_test, u_test = data["u0_test"], data["u_test"]
    x, y, t = data["x"], data["y"], data["t"]

    print(f"  Train size: {u_train.shape}")
    print(f"  Total kernel dim: {u_train.size}")

    # Standardize
    u_mean = jnp.mean(u_train)
    u_std = jnp.std(u_train)
    print(f"\n  Mean: {float(u_mean):.4f}")
    print(f"  Std:  {float(u_std):.4f}")
    u_train_norm = (u_train - u_mean) / u_std

    # Params
    params = init_params(output_scale=1.0)

    # Optimize
    params_opt, _ = run_optimization(params, u0_train, u_train_norm, x, y, t, max_iter=30)
    
    p = transform_params(params_opt)
    print("\nOptimized Params:")
    for k, v in p.items():
        print(f"  {k}: {float(v):.4f}")

    # Predict
    print("\nPredicting...")
    # JIT the predict function
    pred_fn = jax.jit(predict)
    u_pred_norm = pred_fn(params_opt, u0_train, u_train_norm, u0_test, x, y, t)
    u_pred = u_pred_norm * u_std + u_mean
    
    rmse = jnp.sqrt(jnp.mean((u_pred - u_test)**2))
    print(f"  RMSE: {float(rmse):.4f}")

    # Plot
    print("\n--- Plotting Results ---")
    plot_2d_results(u0_test, u_test, u_pred, t_idx=-1, save_path="kronecker_gp_burgers_2d.png")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
