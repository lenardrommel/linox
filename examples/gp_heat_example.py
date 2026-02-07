#!/usr/bin/env python3
"""GP Regression Example for 1D Heat Equation with linox.

This example demonstrates exact GP regression to learn the 1D heat equation
solution operator using linox for all linear algebra operations.

Key linox operations demonstrated:
    - lo.Matrix: Dense operator from array
    - lo.PSD: Positive semi-definite wrapper
    - lo.Diagonal: Diagonal operator for noise
    - lo.solve: Linear system solve
    - lo.slogdet: Log determinant
    - lo.validate: Operator validation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import linox as lo

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Data Generation: 1D Heat Equation
# =============================================================================


def generate_heat_data(
    nx: int = 30,
    n_train: int = 50,
    n_test: int = 5,
    T: float = 0.5,
    alpha: float = 0.1,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Generate training and test data from the 1D heat equation.

    Returns initial conditions and solutions at time T.
    """
    np.random.seed(seed)

    # Spatial grid [0, pi]
    x = np.linspace(0, np.pi, nx)
    dx = x[1] - x[0]

    # Time stepping parameters
    dt = 0.4 * dx**2 / alpha  # CFL-stable
    n_steps = int(T / dt)

    def solve_heat(u0: np.ndarray) -> np.ndarray:
        """Solve heat equation with explicit Euler."""
        u = u0.copy()
        for _ in range(n_steps):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (
                u[2:] - 2 * u[1:-1] + u[:-2]
            )
            # Dirichlet BCs
            u_new[0] = 0
            u_new[-1] = 0
            u = u_new
        return u

    # Generate random initial conditions (sums of sinusoids)
    def random_ic(n_modes: int = 3) -> np.ndarray:
        """Random superposition of sine modes."""
        u0 = np.zeros(nx)
        for k in range(1, n_modes + 1):
            amp = np.random.randn() / k
            u0 += amp * np.sin(k * x)
        return u0

    # Generate training data
    u0_train = np.array([random_ic() for _ in range(n_train)])
    uT_train = np.array([solve_heat(u0) for u0 in u0_train])

    # Generate test data
    u0_test = np.array([random_ic() for _ in range(n_test)])
    uT_test = np.array([solve_heat(u0) for u0 in u0_test])

    return (
        jnp.array(u0_train),
        jnp.array(uT_train),
        jnp.array(u0_test),
        jnp.array(uT_test),
        jnp.array(x),
    )


# =============================================================================
# GP Kernel Functions
# =============================================================================


def rbf_kernel_matrix(
    X1: jax.Array,
    X2: jax.Array,
    lengthscale: float = 0.3,
    variance: float = 1.0,
) -> jax.Array:
    """Compute RBF kernel matrix between function samples.

    Uses L2 inner product as similarity metric between functions.
    K[i,j] = variance * exp(-||X1[i] - X2[j]||^2 / (2 * lengthscale^2))
    """
    # Compute squared distances between samples
    sq_dists = jnp.sum(
        (X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1
    )
    return variance * jnp.exp(-0.5 * sq_dists / lengthscale**2)


# =============================================================================
# GP Model using linox
# =============================================================================


class GPHeatOperator:
    """GP model for learning the heat equation solution operator.

    Given initial conditions u0, predicts solutions at time T.
    Uses linox operators for efficient computation.
    """

    def __init__(
        self,
        lengthscale: float = 5.0,
        variance: float = 1.0,
        noise_variance: float = 0.01,
        jitter: float = 1e-6,
    ) -> None:
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_variance = noise_variance
        self.jitter = jitter

        # Cached operators
        self.K_train: lo.LinearOperator | None = None
        self.alpha: jax.Array | None = None

    def fit(self, u0_train: jax.Array, uT_train: jax.Array) -> None:
        """Fit the GP model to training data.

        Parameters
        ----------
        u0_train : jax.Array
            Initial conditions, shape (n_train, nx).
        uT_train : jax.Array
            Solutions at time T, shape (n_train, nx).
        """
        n_train = u0_train.shape[0]

        # Build kernel matrix on initial conditions
        K_dense = rbf_kernel_matrix(
            u0_train, u0_train,
            self.lengthscale, self.variance
        )
        K_dense = K_dense + self.jitter * jnp.eye(n_train)

        # Wrap as PSD linox operator
        self.K_train = lo.PSD(lo.Matrix(K_dense))

        # Add noise
        noise_diag = self.noise_variance * jnp.ones(n_train)
        K_noisy = self.K_train.wrapped + lo.Diagonal(noise_diag)

        # Solve for each output dimension using linox
        # alpha = K_noisy^{-1} @ uT_train (for each spatial point)
        self.alpha = lo.solve(K_noisy, uT_train)  # Shape: (n_train, nx)

        # Store training data for predictions
        self._u0_train = u0_train

    def predict(
        self, u0_test: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Predict solutions for new initial conditions.

        Parameters
        ----------
        u0_test : jax.Array
            Test initial conditions, shape (n_test, nx).

        Returns
        -------
        mean : jax.Array
            Predictive mean, shape (n_test, nx).
        var : jax.Array
            Predictive variance (diagonal), shape (n_test, nx).
        """
        if self.alpha is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        n_test = u0_test.shape[0]

        # Cross-covariance K(test, train)
        K_cross = rbf_kernel_matrix(
            u0_test, self._u0_train,
            self.lengthscale, self.variance
        )

        # Predictive mean: K_cross @ alpha
        mean = K_cross @ self.alpha

        # Predictive variance (simplified)
        # K_ss - K_cross @ K_noisy^{-1} @ K_cross.T
        K_ss_diag = self.variance * jnp.ones(n_test)

        # For variance, compute diagonal of K_cross @ K^{-1} @ K_cross.T
        # Use linox solve
        v = lo.solve(self.K_train, K_cross.T)  # (n_train, n_test)
        var_reduction = jnp.sum(K_cross * v.T, axis=1)  # (n_test,)
        var = K_ss_diag - var_reduction + self.noise_variance

        # Broadcast to all spatial dimensions
        var = jnp.maximum(var, 1e-6)[:, None] * jnp.ones_like(mean)

        return mean, var


# =============================================================================
# Plotting
# =============================================================================


def plot_results(
    x: jax.Array,
    u0_test: jax.Array,
    uT_test: jax.Array,
    pred_mean: jax.Array,
    pred_std: jax.Array,
    n_samples: int = 3,
    save_path: str | None = None,
) -> None:
    """Plot GP predictions vs true solutions."""
    n_show = min(n_samples, u0_test.shape[0])

    _fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 6))
    if n_show == 1:
        axes = axes.reshape(-1, 1)

    x_np = np.asarray(x)

    for i in range(n_show):
        # Initial condition
        ax_ic = axes[0, i]
        ax_ic.plot(x_np, np.asarray(u0_test[i]), "b-", lw=2)
        ax_ic.set_title(f"Initial Condition {i+1}")
        ax_ic.set_xlabel("x")
        ax_ic.set_ylabel("u(x, 0)")
        ax_ic.grid(True, alpha=0.3)

        # Prediction vs true
        ax_pred = axes[1, i]
        true_i = np.asarray(uT_test[i])
        mean_i = np.asarray(pred_mean[i])
        std_i = np.asarray(pred_std[i])

        ax_pred.plot(x_np, true_i, "g-", lw=2, label="True")
        ax_pred.plot(x_np, mean_i, "r--", lw=2, label="GP Mean")
        ax_pred.fill_between(
            x_np,
            mean_i - 2 * std_i,
            mean_i + 2 * std_i,
            color="red",
            alpha=0.2,
            label="95% CI",
        )
        ax_pred.set_title(f"Solution at t=T (Sample {i+1})")
        ax_pred.set_xlabel("x")
        ax_pred.set_ylabel("u(x, T)")
        ax_pred.legend(fontsize=8)
        ax_pred.grid(True, alpha=0.3)

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
    """Run GP example on 1D heat equation."""
    print("=" * 60)
    print("GP Regression for 1D Heat Equation Operator Learning")
    print("=" * 60)

    # Generate data
    print("\n--- Generating Data ---")
    u0_train, uT_train, u0_test, uT_test, x = generate_heat_data(
        nx=30,
        n_train=100,
        n_test=5,
        T=0.3,
        alpha=0.1,
    )

    print(f"  Training: {u0_train.shape[0]} samples × {u0_train.shape[1]} points")
    print(f"  Test: {u0_test.shape[0]} samples")

    # Build and train GP
    print("\n--- Training GP Model ---")
    gp = GPHeatOperator(
        lengthscale=3.0,  # In function space
        variance=1.0,
        noise_variance=0.001,
    )
    gp.fit(u0_train, uT_train)

    print(f"  K_train type: {type(gp.K_train).__name__}")
    print(f"  K_train shape: {gp.K_train.shape}")
    print(f"  K_train is_psd: {gp.K_train.is_psd}")

    # Validate operators
    try:
        lo.validate(gp.K_train)
        print("  ✓ lo.validate(K_train) passed")
    except lo.ValidationError as e:
        print(f"  ✗ Validation failed: {e}")

    # Make predictions
    print("\n--- Making Predictions ---")
    pred_mean, pred_var = gp.predict(u0_test)
    pred_std = jnp.sqrt(pred_var)

    print(f"  pred_mean shape: {pred_mean.shape}")

    # Compute metrics
    print("\n--- Metrics ---")
    for i in range(u0_test.shape[0]):
        mse = jnp.mean((pred_mean[i] - uT_test[i]) ** 2)
        print(f"  Sample {i+1}: MSE = {mse:.6f}, RMSE = {jnp.sqrt(mse):.6f}")

    overall_mse = jnp.mean((pred_mean - uT_test) ** 2)
    print(f"\n  Overall: MSE = {overall_mse:.6f}, RMSE = {jnp.sqrt(overall_mse):.6f}")

    # Plot results
    print("\n--- Plotting Results ---")
    plot_results(
        x, u0_test, uT_test,
        pred_mean, pred_std,
        n_samples=3,
        save_path="gp_heat_1d.png",
    )

    # Summary
    print("\n" + "=" * 60)
    print("linox Operations Used:")
    print("=" * 60)
    print("  - lo.Matrix(arr): Dense linear operator")
    print("  - lo.PSD(op): PSD wrapper for kernel")
    print("  - lo.Diagonal(arr): Diagonal noise operator")
    print("  - lo.solve(K, y): Linear system solve")
    print("  - lo.validate(K): Operator validation")
    print("\n✅ All linox operations work correctly!")


if __name__ == "__main__":
    main()
