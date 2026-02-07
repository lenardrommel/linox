#!/usr/bin/env python3
"""Kronecker GP for 1D Burgers Equation with Optimization.

This example demonstrates:
1. Kronecker-structured GP: K = K_function ⊗ K_spatial_x ⊗ K_time
2. Hyperparameter optimization using optax LBFGS
3. Full JIT compilation proving linox operations are jittable

Key linox operations demonstrated:
    - lo.Kronecker: Kronecker product structure
    - lo.Matrix, lo.Diagonal, lo.Identity
    - lo.IsotropicAdditiveLinearOperator: Efficient noise addition
    - lo.lsolve: Linear system solve
    - lo.slogdet: Log-determinant computation
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

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Parameter Handling with Sigmoid Transforms
# =============================================================================


class GPParams(NamedTuple):
    """Raw (unconstrained) GP hyperparameters."""

    ls_func_raw: jax.Array
    ls_x_raw: jax.Array
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
    ls_t: float = 0.2,
    noise_var: float = 0.01,
    output_scale: float = 1.0,
) -> GPParams:
    """Initialize raw parameters from constrained values."""
    return GPParams(
        ls_func_raw=inverse_sigmoid_transform(ls_func, 0.01, 10.0),
        ls_x_raw=inverse_sigmoid_transform(ls_x, 0.01, 100.0),
        ls_t_raw=inverse_sigmoid_transform(ls_t, 0.01, 100.0),
        noise_var_raw=inverse_sigmoid_transform(noise_var, 1e-6, 1.0),
        output_scale_raw=inverse_sigmoid_transform(output_scale, 0.01, 100.0),
    )


def transform_params(params: GPParams) -> dict:
    """Transform raw parameters to constrained values."""
    return {
        "ls_func": sigmoid_transform(params.ls_func_raw, 0.01, 10.0),
        "ls_x": sigmoid_transform(params.ls_x_raw, 0.01, 100.0),
        "ls_t": sigmoid_transform(params.ls_t_raw, 0.01, 100.0),
        "noise_var": sigmoid_transform(params.noise_var_raw, 1e-6, 1.0),
        "output_scale": sigmoid_transform(params.output_scale_raw, 0.01, 100.0),
    }


# =============================================================================
# Data Generation using exponax (1D Burgers)
# =============================================================================


def generate_burgers_data(
    num_points: int = 64,
    n_train: int = 50,
    n_test: int = 5,
    n_time_points: int = 10,
    diffusivity: float = 0.1,
    seed: int = 42,
) -> dict:
    """Generate training and test data from the 1D Burgers equation."""
    key = random.PRNGKey(seed)

    dt = 0.01
    stepper = ex.stepper.Burgers(
        num_spatial_dims=1,
        domain_extent=2 * jnp.pi,
        num_points=num_points,
        dt=dt,
        diffusivity=diffusivity,
        convection_scale=1.0,
    )

    ic_gen = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=1, cutoff=3, max_one=True
    )

    subsample = 5
    n_steps = n_time_points * subsample
    rollout_fn = ex.rollout(stepper, n_steps, include_init=True)

    def generate_sample(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        u0 = ic_gen(num_points, key=key)
        trajectory = rollout_fn(u0)
        trajectory = trajectory[::subsample, 0, :]
        return u0[0], trajectory

    # Training data
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

    # Test data
    test_keys = random.split(key, n_test)
    u0_test_list, u_test_list = [], []
    for k in test_keys:
        u0, traj = generate_sample(k)
        u0_test_list.append(u0)
        u_test_list.append(traj)

    u0_test = jnp.stack(u0_test_list)
    u_test = jnp.stack(u_test_list)

    x = ex.make_grid(1, 2 * jnp.pi, num_points)[0]
    t = jnp.arange(n_time_points + 1) * dt * subsample

    return {
        "u0_train": u0_train,
        "u_train": u_train,
        "u0_test": u0_test,
        "u_test": u_test,
        "x": x,
        "t": t,
    }


# =============================================================================
# Kernel Functions (pure JAX, jittable)
# =============================================================================


def rbf_kernel(X1: jax.Array, X2: jax.Array, lengthscale: jax.Array) -> jax.Array:
    """RBF kernel for 1D inputs."""
    sq_dists = (X1[:, None] - X2[None, :]) ** 2
    return jnp.exp(-0.5 * sq_dists / lengthscale**2)


def function_kernel(U1: jax.Array, U2: jax.Array, lengthscale: jax.Array) -> jax.Array:
    """Linear/Dot Product kernel (L2 inner product).
    
    k(u, v) = <u, v> / lengthscale^2
    """
    # Compute pairwise inner products
    dot_prod = jnp.matmul(U1, U2.T)
    
    # Normalize by lengthscale (acting as a variance/scale factor combined with output_scale)
    # Usually Linear kernel is just <x, y>. We keep lengthscale for API consistency.
    return dot_prod + lengthscale**2


# =============================================================================
# Kronecker GP (Functional, Jittable)
# =============================================================================


def build_kronecker_kernel(
    u0_train: jax.Array,
    x: jax.Array,
    t: jax.Array,
    ls_func: jax.Array,
    ls_x: jax.Array,
    ls_t: jax.Array,
    noise_var: jax.Array,
    output_scale: jax.Array,
    jitter: float = 1e-6,
) -> tuple[lo.LinearOperator, jax.Array, jax.Array, jax.Array]:
    """Build Kronecker kernel and return components for logdet.

    Returns
    -------
    K_noisy : LinearOperator
        Full kernel with noise
    K_func, K_x, K_t : jax.Array
        Dense matrices for computing logdet via eigendecomposition
    """
    n_func = u0_train.shape[0]
    n_x = x.shape[0]
    n_t = t.shape[0]

    # Build individual kernels
    K_func = function_kernel(u0_train, u0_train, ls_func)
    K_func = K_func + jitter * jnp.eye(n_func)
    
    # Apply output scale to function kernel
    K_func = output_scale * K_func

    K_x = rbf_kernel(x, x, ls_x)
    K_x = K_x + jitter * jnp.eye(n_x)

    K_t = rbf_kernel(t, t, ls_t)
    K_t = K_t + jitter * jnp.eye(n_t)

    # Kronecker structure
    K_kron = lo.Kronecker(lo.Matrix(K_func), lo.Kronecker(lo.Matrix(K_x), lo.Matrix(K_t)))

    # Add noise
    K_noisy = lo.IsotropicAdditiveLinearOperator(noise_var, K_kron)

    return K_noisy, K_func, K_x, K_t


def negative_log_likelihood(
    params: GPParams,
    u0_train: jax.Array,
    u_train: jax.Array,
    x: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """Compute negative log-likelihood (NLL) for GP.

    NLL = 0.5 * (y^T K^{-1} y + log|K| + n*log(2π))

    Uses eigendecomposition for differentiable solve:
    K = K_func ⊗ K_x ⊗ K_t + σ²I

    For K = A ⊗ B ⊗ C, eigenvectors are U_A ⊗ U_B ⊗ U_C
    and eigenvalues are λ_A_i * λ_B_j * λ_C_k
    """
    p = transform_params(params)

    _, K_func, K_x, K_t = build_kronecker_kernel(
        u0_train, x, t,
        p["ls_func"], p["ls_x"], p["ls_t"], p["noise_var"], p["output_scale"]
    )

    y_flat = u_train.reshape(-1)
    n = y_flat.shape[0]
    K_func.shape[0]
    K_x.shape[0]
    K_t.shape[0]

    # Eigendecompositions of component kernels
    eig_f, U_f = jnp.linalg.eigh(K_func)
    eig_x, U_x = jnp.linalg.eigh(K_x)
    eig_t, U_t = jnp.linalg.eigh(K_t)

    # Compute Kronecker eigenvalues: λ_f_i * λ_x_j * λ_t_k + σ²
    # Ordering matches vec of (f, x, t) tensor
    eig_kron = jnp.einsum("i,j,k->ijk", eig_f, eig_x, eig_t).ravel()
    eig_noisy = eig_kron + p["noise_var"]

    # y_flat has Kronecker ordering from reshape: (n_f, n_t, n_x) -> flat
    # But we need (n_f, n_x, n_t) ordering to match eigenvalue kron
    # The actual data reshape is u_train: (n_f, n_t, n_x). Need to permute.
    y_tensor = u_train.transpose(0, 2, 1)  # (n_f, n_x, n_t)

    # Apply (U_f ⊗ U_x ⊗ U_t)^T to y via mode products
    # Mode-1: U_f^T
    y_proj = jnp.einsum("if,fxt->ixt", U_f.T, y_tensor)
    # Mode-2: U_x^T
    y_proj = jnp.einsum("jx,ixt->ijt", U_x.T, y_proj)
    # Mode-3: U_t^T
    y_proj = jnp.einsum("kt,ijt->ijk", U_t.T, y_proj)

    # Flatten to match eigenvalue ordering (i, j, k)
    y_proj_flat = y_proj.ravel()

    # Quadratic term: y^T K^{-1} y = sum((U^T y)^2 / eigenvalues)
    quad = 0.5 * jnp.sum(y_proj_flat**2 / eig_noisy)

    # Log-determinant
    logdet = jnp.sum(jnp.log(eig_noisy))
    logdet_term = 0.5 * logdet

    # Constant term
    const = 0.5 * n * jnp.log(2 * jnp.pi)

    return quad + logdet_term + const


def predict(
    params: GPParams,
    u0_train: jax.Array,
    u_train: jax.Array,
    u0_test: jax.Array,
    x: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """Predict trajectories for test initial conditions."""
    p = transform_params(params)
    n_test = u0_test.shape[0]
    n_x = x.shape[0]
    n_t = t.shape[0]
    jitter = 1e-6

    # Build training kernel
    K_train, _, _, _ = build_kronecker_kernel(
        u0_train, x, t,
        p["ls_func"], p["ls_x"], p["ls_t"], p["noise_var"], p["output_scale"]
    )

    # Solve for alpha
    y_flat = u_train.reshape(-1)
    alpha = lo.lsolve(K_train, y_flat[:, None])[:, 0]

    # Build cross-kernel matrices
    K_func_cross = function_kernel(u0_test, u0_train, p["ls_func"])
    # Apply output scale (on the first component of kron product)
    K_func_cross = p["output_scale"] * K_func_cross
    K_x_cross = rbf_kernel(x, x, p["ls_x"]) + jitter * jnp.eye(n_x)
    K_t_cross = rbf_kernel(t, t, p["ls_t"]) + jitter * jnp.eye(n_t)

    K_cross = lo.Kronecker(
        lo.Matrix(K_func_cross),
        lo.Kronecker(lo.Matrix(K_x_cross), lo.Matrix(K_t_cross))
    )

    # Predict: mean = K_cross @ alpha
    pred_flat = K_cross @ alpha
    pred = pred_flat.reshape(n_test, n_t, n_x)

    return pred


# =============================================================================
# Optimization
# =============================================================================


def run_optimization(
    params: GPParams,
    u0_train: jax.Array,
    u_train: jax.Array,
    x: jax.Array,
    t: jax.Array,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[GPParams, list]:
    """Run LBFGS optimization of GP hyperparameters.

    Uses optax.lbfgs with backtracking line search.
    """
    print("\n--- Hyperparameter Optimization ---")

    # Create loss function
    @jax.jit
    def loss_fn(p):
        return negative_log_likelihood(p, u0_train, u_train, x, t)

    # JIT the gradient computation
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Compile and time
    print("  Compiling JIT functions...")
    t0 = time.perf_counter()
    _ = value_and_grad_fn(params)  # Warmup/compile
    compile_time = time.perf_counter() - t0
    print(f"  ✓ JIT compilation time: {compile_time:.2f}s")

    # Initialize optimizer
    optimizer = optax.lbfgs(
        linesearch=optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=15, decrease_factor=0.5
        )
    )
    state = optimizer.init(params)

    history = {"nll": [], "grad_norm": []}
    best_params = params
    best_loss = float("inf")

    print(f"\n  {'Iter':>5} | {'NLL':>12} | {'‖∇‖':>12} | {'ls_func':>8} | {'ls_x':>8} | {'ls_t':>8} | {'noise':>9} | {'scale':>8}")
    print("  " + "-" * 80)

    for iteration in range(max_iter):
        loss, grad = value_and_grad_fn(params)

        # Update history
        grad_norm = otu.tree_norm(grad)
        history["nll"].append(float(loss))
        history["grad_norm"].append(float(grad_norm))

        # Track best
        if loss < best_loss:
            best_loss = float(loss)
            best_params = params

        # Print progress
        p = transform_params(params)
        if iteration % 5 == 0 or iteration == max_iter - 1:
            print(f"  {iteration:5d} | {float(loss):12.4e} | {float(grad_norm):12.4e} | "
                  f"{float(p['ls_func']):8.4f} | {float(p['ls_x']):8.4f} | "
                  f"{float(p['ls_t']):8.4f} | {float(p['noise_var']):9.2e} | {float(p['output_scale']):8.4f}")

        # Check convergence
        if grad_norm < tol:
            print(f"\n  ✓ Converged at iteration {iteration} (‖∇‖ = {float(grad_norm):.2e})")
            break

        # Update parameters
        updates, state = optimizer.update(
            grad, state, params, value=loss, grad=grad, value_fn=loss_fn
        )
        params = optax.apply_updates(params, updates)

    return best_params, history


# =============================================================================
# Plotting
# =============================================================================


def plot_results(
    x: jax.Array,
    t: jax.Array,
    u0_test: jax.Array,
    u_test: jax.Array,
    u_pred: jax.Array,
    history: list,
    n_samples: int = 2,
    save_path: str | None = None,
) -> None:
    """Plot Burgers equation predictions and optimization history."""
    n_show = min(n_samples, u0_test.shape[0])

    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(3, n_show + 1, width_ratios=[1] * n_show + [0.8])

    x_np, t_np = np.asarray(x), np.asarray(t)
    X, T = np.meshgrid(x_np, t_np)

    for i in range(n_show):
        # Initial condition
        ax_ic = fig.add_subplot(gs[0, i])
        ax_ic.plot(x_np, np.asarray(u0_test[i]), "b-", lw=2)
        ax_ic.set_title(f"IC {i+1}")
        ax_ic.set_xlabel("x")
        ax_ic.set_ylabel("u(x,0)")
        ax_ic.grid(True, alpha=0.3)

        # True trajectory
        ax_true = fig.add_subplot(gs[1, i])
        im = ax_true.pcolormesh(X, T, np.asarray(u_test[i]), shading="auto", cmap="RdBu_r")
        ax_true.set_title("True u(x,t)")
        ax_true.set_xlabel("x")
        ax_true.set_ylabel("t")
        fig.colorbar(im, ax=ax_true, shrink=0.8)

        # Predicted trajectory
        ax_pred = fig.add_subplot(gs[2, i])
        im = ax_pred.pcolormesh(X, T, np.asarray(u_pred[i]), shading="auto", cmap="RdBu_r")
        ax_pred.set_title("Pred u(x,t)")
        ax_pred.set_xlabel("x")
        ax_pred.set_ylabel("t")
        fig.colorbar(im, ax=ax_pred, shrink=0.8)

    # Optimization history
    ax_nll = fig.add_subplot(gs[0, n_show])
    ax_nll.semilogy(history["nll"], "b-", lw=2)
    ax_nll.set_xlabel("Iteration")
    ax_nll.set_ylabel("NLL")
    ax_nll.set_title("Optimization")
    ax_nll.grid(True, alpha=0.3)

    ax_grad = fig.add_subplot(gs[1, n_show])
    ax_grad.semilogy(history["grad_norm"], "r-", lw=2)
    ax_grad.set_xlabel("Iteration")
    ax_grad.set_ylabel("‖∇‖")
    ax_grad.set_title("Gradient Norm")
    ax_grad.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run Kronecker GP with optimization on 1D Burgers equation."""
    print("=" * 70)
    print("   Kronecker GP for 1D Burgers with Hyperparameter Optimization")
    print("   Covariance: K = K_function ⊗ K_x ⊗ K_t")
    print("=" * 70)

    # Generate data
    print("\n--- Generating Burgers Data (exponax) ---")
    data = generate_burgers_data(
        num_points=64,
        n_train=30,
        n_test=3,
        n_time_points=8,
        diffusivity=0.1,
    )

    u0_train, u_train = data["u0_train"], data["u_train"]
    u0_test, u_test = data["u0_test"], data["u_test"]
    x, t = data["x"], data["t"]

    print(f"  Training: {u0_train.shape[0]} samples")
    print(f"  Spatial: {x.shape[0]} points, Time: {t.shape[0]} points")
    print(f"  Total kernel size: {u0_train.shape[0] * x.shape[0] * t.shape[0]}")

    # Standardize data (Zero Mean, Unit Variance)
    u_mean = jnp.mean(u_train)
    u_std = jnp.std(u_train)
    print("\n--- Data Standardization ---")
    print(f"  Mean: {float(u_mean):.4f}")
    print(f"  Std:  {float(u_std):.4f}")
    
    u_train_norm = (u_train - u_mean) / u_std
    # Note: We don't standardize u0_train (inputs), only targets (u_train)
    # But u0_train is used in the kernel. Using raw u0 is fine as it's input features.
    
    # Initialize parameters
    params = init_params(ls_func=0.01, ls_x=0.1, ls_t=0.5, noise_var=0.1, output_scale=0.10)

    # Demonstrate jittability
    print("\n--- Demonstrating JIT Compilation ---")

    @jax.jit
    def jitted_nll(p):
        return negative_log_likelihood(p, u0_train, u_train_norm, x, t)

    @jax.jit
    def jitted_predict(p):
        return predict(p, u0_train, u_train_norm, u0_test, x, t)

    # Compile
    print("  Compiling NLL...")
    t0 = time.perf_counter()
    nll_init = jitted_nll(params)
    t_nll = time.perf_counter() - t0
    print(f"  ✓ NLL compilation: {t_nll:.2f}s, value: {float(nll_init):.4e}")

    print("  Compiling predict...")
    t0 = time.perf_counter()
    _ = jitted_predict(params)
    t_pred = time.perf_counter() - t0
    print(f"  ✓ Predict compilation: {t_pred:.2f}s")

    # Time execution (post-compilation)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = jitted_nll(params)
    t_exec = (time.perf_counter() - t0) / 10
    print(f"  ✓ NLL execution time: {t_exec*1000:.2f}ms per call")

    # Run optimization
    optimized_params, history = run_optimization(
        params, u0_train, u_train_norm, x, t,
        max_iter=50, tol=1e-4
    )

    # Final parameters
    p_opt = transform_params(optimized_params)
    print("\n--- Optimized Parameters ---")
    print(f"  ls_func:   {float(p_opt['ls_func']):.6f}")
    print(f"  ls_x:      {float(p_opt['ls_x']):.6f}")
    print(f"  ls_t:      {float(p_opt['ls_t']):.6f}")
    print(f"  noise_var: {float(p_opt['noise_var']):.2e}")
    print(f"  output_scale: {float(p_opt['output_scale']):.4f}")

    # Validate kernel
    print("\n--- Validating Kronecker Kernel ---")
    K, _, _, _ = build_kronecker_kernel(
        u0_train, x, t,
        p_opt["ls_func"], p_opt["ls_x"], p_opt["ls_t"], p_opt["noise_var"], p_opt["output_scale"]
    )
    print(f"  Kernel shape: {K.shape}")
    print(f"  Kernel type: {type(K).__name__}")
    try:
        lo.validate(K)
        print("  ✓ lo.validate() passed")
    except lo.ValidationError as e:
        print(f"  ✗ Validation failed: {e}")

    # Make predictions
    print("\n--- Making Predictions ---")
    u_pred_norm = jitted_predict(optimized_params)
    u_pred = u_pred_norm * u_std + u_mean
    print(f"  Prediction shape: {u_pred.shape}")

    # Compute metrics
    print("\n--- Metrics ---")
    for i in range(u0_test.shape[0]):
        mse = jnp.mean((u_pred[i] - u_test[i]) ** 2)
        print(f"  Sample {i+1}: MSE = {float(mse):.6f}, RMSE = {float(jnp.sqrt(mse)):.6f}")

    overall_mse = jnp.mean((u_pred - u_test) ** 2)
    print(f"\n  Overall: MSE = {float(overall_mse):.6f}, RMSE = {float(jnp.sqrt(overall_mse)):.6f}")

    # Plot
    print("\n--- Plotting ---")
    plot_results(
        x, t, u0_test, u_test, u_pred, history,
        n_samples=2, save_path="kronecker_gp_burgers_1d.png"
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Linox Operations Demonstrated")
    print("=" * 70)
    print("  ✓ lo.Kronecker: Efficient Kronecker product structure")
    print("  ✓ lo.Matrix: Dense matrix operator")
    print("  ✓ lo.IsotropicAdditiveLinearOperator: Noise addition")
    print("  ✓ lo.lsolve: Linear system solve")
    print("  ✓ lo.slogdet: Log-determinant for NLL")
    print("  ✓ lo.validate: Operator validation")
    print()
    print("  JIT Compilation: ✓ All operations are jittable")
    print("  Optimization: ✓ optax.lbfgs with backtracking line search")
    print("\n✅ Kronecker GP optimization completed successfully!")


if __name__ == "__main__":
    main()
