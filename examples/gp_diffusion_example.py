#!/usr/bin/env python3
"""GP Regression for 1D Diffusion with Hyperparameter Optimization.

This example demonstrates:
1. Dense GP regression for operator learning (u0 -> uT)
2. Hyperparameter optimization using optax (LBFGS)
3. JIT compilation of linox operations

Key linox operations:
    - lo.Matrix: Dense operator
    - lo.IsotropicAdditiveLinearOperator: Noise handling
    - lo.lsolve, lo.slogdet: Jittable linear algebra
"""

from __future__ import annotations

import time
from typing import NamedTuple

import exponax as ex
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import optax
from optax import tree_utils as out

import linox as lo

# Enable float64
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Parameter Handling
# =============================================================================


class GPParams(NamedTuple):
    log_lengthscale: jax.Array
    log_variance: jax.Array
    log_noise: jax.Array


def init_params(
    lengthscale: float = 3.0,
    variance: float = 1.0,
    noise: float = 0.001,
) -> GPParams:
    return GPParams(
        log_lengthscale=jnp.log(jnp.array(lengthscale)),
        log_variance=jnp.log(jnp.array(variance)),
        log_noise=jnp.log(jnp.array(noise)),
    )


def transform_params(params: GPParams) -> dict:
    return {
        "lengthscale": jnp.exp(params.log_lengthscale),
        "variance": jnp.exp(params.log_variance),
        "noise": jnp.exp(params.log_noise),
    }


# =============================================================================
# Data Generation
# =============================================================================


def generate_diffusion_data(
    num_points: int = 64,
    n_train: int = 50,
    n_test: int = 5,
    n_steps: int = 50,
    diffusivity: float = 0.1,
    seed: int = 42,
) -> dict:
    """Generate 1D diffusion data."""
    key = random.PRNGKey(seed)
    stepper = ex.stepper.Diffusion(
        num_spatial_dims=1,
        domain_extent=2 * jnp.pi,
        num_points=num_points,
        dt=0.01,
        diffusivity=diffusivity,
    )
    ic_gen = ex.ic.RandomTruncatedFourierSeries(num_spatial_dims=1, cutoff=5, max_one=True)
    rollout = ex.rollout(stepper, n_steps, include_init=True)

    def sample(k):
        u0 = ic_gen(num_points, key=k)[0]
        traj = rollout(u0[None])[0]
        return u0, traj[-1]

    # Train
    keys = random.split(key, n_train + n_test)
    u0_all, uT_all = jax.vmap(sample)(keys)

    x = ex.make_grid(1, 2 * jnp.pi, num_points)[0]

    return {
        "u0_train": u0_all[:n_train],
        "uT_train": uT_all[:n_train],
        "u0_test": u0_all[n_train:],
        "uT_test": uT_all[n_train:],
        "x": x,
    }


# =============================================================================
# Kernel & Model
# =============================================================================


def build_kernel(X1, X2, lengthscale, variance, jitter=1e-6, add_jitter=False):
    """Dense RBF kernel."""
    # L2 distance between functions
    sq_dists = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    K = variance * jnp.exp(-0.5 * sq_dists / lengthscale**2)

    if add_jitter:
         K = K + jitter * jnp.eye(X1.shape[0])

    return lo.Matrix(K)


def negative_log_likelihood(params: GPParams, u0_train, uT_train):
    """Compute NLL for Dense GP."""
    p = transform_params(params)
    u0_train.shape[0]

    # Kernel
    K_base = build_kernel(u0_train, u0_train, p["lengthscale"], p["variance"], add_jitter=True)
    K = lo.IsotropicAdditiveLinearOperator(p["noise"], K_base)

    # Check if we can use cholesky (Dense SPD)
    # linox.slogdet on IsotropicAdditive might use eigenvalues or dense fallback
    # For dense optimization, let's trust linox dispatch

    # Solve K alpha = Y
    # Y is (n_train, n_x). We treat each x point as independent sample from same GP?
    # Or sum NLL across spatial dims?
    # Typically in operator learning with scalar kernel on u0, we assume independence across output dims
    # or just sum the log likelihoods.

    # Solve
    alpha = lo.lsolve(K, uT_train)  # (n_train, n_x)

    # Quadratic term: tr(Y^T K^{-1} Y) = sum(Y * alpha)
    quad = 0.5 * jnp.sum(uT_train * alpha)

    # Log det: n_x * 0.5 * log|K|
    _sign, logdet = lo.slogdet(K)
    logdet_term = 0.5 * uT_train.shape[1] * logdet

    const = 0.5 * uT_train.size * jnp.log(2 * jnp.pi)

    return quad + logdet_term + const


def predict(params: GPParams, u0_train, uT_train, u0_test):
    p = transform_params(params)
    K_train_base = build_kernel(u0_train, u0_train, p["lengthscale"], p["variance"], add_jitter=True)
    K_train = lo.IsotropicAdditiveLinearOperator(p["noise"], K_train_base)

    alpha = lo.lsolve(K_train, uT_train)

    K_cross = build_kernel(u0_test, u0_train, p["lengthscale"], p["variance"])
    mean = K_cross @ alpha

    return mean


# =============================================================================
# Optimization
# =============================================================================


def run_optimization(params, u0_train, uT_train, max_iter=50):
    print("\n--- Hyperparameter Optimization ---")

    @jax.jit
    def loss(p):
        return negative_log_likelihood(p, u0_train, uT_train)

    val_grad = jax.jit(jax.value_and_grad(loss))

    # Warmup
    print("  Compiling...")
    t0 = time.time()
    _ = val_grad(params)
    print(f"  ✓ Compiled in {time.time()-t0:.2f}s")

    opt = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(max_backtracking_steps=15))
    state = opt.init(params)

    print(f"  {'Iter':>4} | {'NLL':>12} | {'Grad':>12}")

    for i in range(max_iter):
        l, g = val_grad(params)
        gn = out.tree_norm(g)

        if i % 10 == 0 or i == max_iter - 1:
             print(f"  {i:4d} | {float(l):12.4e} | {float(gn):12.4e}")

        if gn < 1e-4:
            print("  ✓ Converged")
            break

        updates, state = opt.update(g, state, params, value=l, grad=g, value_fn=loss)
        params = optax.apply_updates(params, updates)

    return params


# =============================================================================
# Main
# =============================================================================


def main():
    print("="*60)
    print(" 1D Diffusion Optimization (Dense GP)")
    print("="*60)

    data = generate_diffusion_data(n_train=40, n_test=5, num_points=64)
    u0_train, uT_train = data["u0_train"], data["uT_train"]
    u0_test, uT_test = data["u0_test"], data["uT_test"]
    x = data["x"]

    print(f"  Train: {u0_train.shape}")

    # Standardize targets
    uT_mean = jnp.mean(uT_train)
    uT_std = jnp.std(uT_train)
    print(f"\n  uT Mean: {float(uT_mean):.4f}")
    print(f"  uT Std:  {float(uT_std):.4f}")
    uT_train_norm = (uT_train - uT_mean) / uT_std

    params = init_params()
    params_opt = run_optimization(params, u0_train, uT_train_norm, max_iter=50)

    p = transform_params(params_opt)
    print("\nOptimized Params:")
    for k, v in p.items():
        print(f"  {k}: {float(v):.4f}")

    # Predict
    print("\nPredicting...")
    pred_fn = jax.jit(predict)
    pred_mean_norm = pred_fn(params_opt, u0_train, uT_train_norm, u0_test)
    pred_mean = pred_mean_norm * uT_std + uT_mean

    rmse = jnp.sqrt(jnp.mean((pred_mean - uT_test)**2))
    print(f"  RMSE: {float(rmse):.4f}")

    # Plot
    print("\nPlotting...")
    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    idx = 0
    axes[0].plot(x, u0_test[idx], label="u0")
    axes[0].set_title("Input (IC)")
    axes[0].legend()

    axes[1].plot(x, uT_test[idx], 'k-', label="True uT")
    axes[1].plot(x, pred_mean[idx], 'r--', label="Pred uT")
    axes[1].set_title("Output (T=0.5)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("gp_diffusion_optimization.png")
    print("  Saved gp_diffusion_optimization.png")
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
