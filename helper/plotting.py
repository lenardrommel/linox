# plotting.py

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tueplots import bundles
from tueplots.constants.color import rgb


# Helpers to apply tueplots styles cleanly
def make_tueplots_rc(
    *,
    base="icml2022",
    family="sans-serif",
    usetex=False,
    column="full",
    nrows=None,
    ncols=None,
    font_size=24,
    axes_labelsize=22,
    legend_fontsize=22,
    tick_labelsize=22,
    title_size=22,
):
    """Build an rcParams dict from tueplots and optional font overrides.

    Example:
        rc = make_tueplots_rc(column="full", font_size=12, legend_fontsize=10)
        plt.rcParams.update(rc)
        # or
        with plt.rc_context(rc):
            ...
    """
    # Get base bundle
    if not hasattr(bundles, base):
        msg = f"Unknown tueplots bundle: {base}"
        raise ValueError(msg)
    bundle_fn = getattr(bundles, base)
    kwargs = {"family": family, "usetex": usetex, "column": column}
    if nrows is not None:
        kwargs["nrows"] = nrows
    if ncols is not None:
        kwargs["ncols"] = ncols
    rc = dict(bundle_fn(**kwargs))

    # Apply optional overrides
    if font_size is not None:
        rc["font.size"] = font_size
    if axes_labelsize is not None:
        rc["axes.labelsize"] = axes_labelsize
    if legend_fontsize is not None:
        rc["legend.fontsize"] = legend_fontsize
    if tick_labelsize is not None:
        rc["xtick.labelsize"] = tick_labelsize
        rc["ytick.labelsize"] = tick_labelsize
    if title_size is not None:
        rc["axes.titlesize"] = title_size

    return rc


def apply_tueplots_rc(**kwargs) -> None:
    """Apply tueplots rcParams globally via plt.rcParams.update(...)."""
    plt.rcParams.update(make_tueplots_rc(**kwargs))


from .data import Heat1dDataGenerator, Heat2dDataGenerator


def generate_preprocess_data_1d(
    x_range=(0, np.pi),
    nx=25,
    T=1.0,
    alpha=0.5,
    N_train=1000,
    N_test=200,
    nx_plot=50,
):
    """Generate and preprocess 1D heat equation data."""
    heat_gen = Heat1dDataGenerator(
        x_range=x_range,
        nx=nx,
        T=T,
        alpha=alpha,
        N_samples=N_train,
    )
    heat_gen.generate_initial_conditions(n_modes=3, seed=42)
    heat_gen.solve_pde()

    test_heat_gen = Heat1dDataGenerator(
        x_range=x_range,
        nx=nx,
        T=T,
        alpha=alpha,
        N_samples=N_test,
    )

    test_heat_gen.generate_initial_conditions(n_modes=3, seed=43)
    test_heat_gen.solve_pde()

    operator_inputs = jnp.array(heat_gen.u0_samples.reshape(N_train, -1))

    # spatial_inputs
    spatial_inputs = jnp.array(heat_gen.x_grid).reshape(-1, 1)

    # outputs: (N_samples, nx) - flattened solutions
    outputs = jnp.array(heat_gen.data.reshape(N_train, -1))

    # Test data
    operator_inputs_test = jnp.array(test_heat_gen.u0_samples.reshape(N_test, -1))
    spatial_inputs_test = spatial_inputs  # Same spatial grid
    outputs_test = jnp.array(test_heat_gen.data.reshape(N_test, -1))

    # Create dense prediction grid
    x_grid_plot = np.linspace(x_range[0], x_range[1], nx_plot)

    spatial_inputs_plot = x_grid_plot[:, None]

    return (
        operator_inputs,
        spatial_inputs,
        outputs,
        operator_inputs_test,
        spatial_inputs_test,
        outputs_test,
        spatial_inputs_plot,
    )


def generate_preprocess_data_2d(
    x_range=(0, np.pi),
    y_range=(0, np.pi),
    nx=25,
    ny=25,
    T=1.0,
    alpha=0.5,
    N_train=1000,
    N_test=200,
    nx_plot=50,
    ny_plot=50,
):
    """Generate and preprocess 2D heat equation data."""
    # Generate training data
    heat_gen = Heat2dDataGenerator(
        x_range=x_range,
        y_range=y_range,
        nx=nx,
        ny=ny,
        T=T,
        alpha=alpha,
        N_samples=N_train,
    )
    heat_gen.generate_initial_conditions(n_modes_x=3, n_modes_y=3, seed=42)
    heat_gen.solve_pde()

    # Generate test data
    test_heat_gen = Heat2dDataGenerator(
        x_range=x_range,
        y_range=y_range,
        nx=nx,
        ny=ny,
        T=T,
        alpha=alpha,
        N_samples=N_test,
    )
    test_heat_gen.generate_initial_conditions(n_modes_x=3, n_modes_y=3, seed=43)
    test_heat_gen.solve_pde()

    # Flatten 2D data for neural network training
    # operator_inputs: (N_samples, nx*ny) - flattened initial conditions
    operator_inputs = jnp.array(heat_gen.u0_samples.reshape(N_train, -1))

    # spatial_inputs: (nx*ny, 2) - (x,y) coordinate pairs
    x_flat = heat_gen.X.flatten()
    y_flat = heat_gen.Y.flatten()
    spatial_inputs = jnp.array(np.column_stack([x_flat, y_flat]))

    # outputs: (N_samples, nx*ny) - flattened solutions
    outputs = jnp.array(heat_gen.data.reshape(N_train, -1))

    # Test data
    operator_inputs_test = jnp.array(test_heat_gen.u0_samples.reshape(N_test, -1))
    spatial_inputs_test = spatial_inputs  # Same spatial grid
    outputs_test = jnp.array(test_heat_gen.data.reshape(N_test, -1))

    # Create dense prediction grid
    x_grid_plot = np.linspace(x_range[0], x_range[1], nx_plot)
    y_grid_plot = np.linspace(y_range[0], y_range[1], ny_plot)
    X_plot, Y_plot = np.meshgrid(x_grid_plot, y_grid_plot)
    spatial_inputs_plot = jnp.array(
        np.column_stack([X_plot.flatten(), Y_plot.flatten()])
    )

    return (
        operator_inputs,
        spatial_inputs,
        outputs,
        operator_inputs_test,
        spatial_inputs_test,
        outputs_test,
        spatial_inputs_plot,
    )


def plot_kernel_predictions_2d(
    pred_mean_flat,
    outputs_test,
    operator_inputs_test,
    spatial_inputs_plot,
    spatial_inputs_test,
    nx,
    ny,
    nx_plot,
    ny_plot,
    N_test,
    figsize=(15, 10),
):
    """Plot 2D heat equation kernel predictions vs true solutions.

    Args:
        pred_mean_flat: Predictions (N_test, nx_plot*ny_plot)
        outputs_test: True test solutions (N_test, nx*ny)
        operator_inputs_test: Test initial conditions (N_test, nx*ny)
        spatial_inputs_plot: Plot coordinates (nx_plot*ny_plot, 2)
        spatial_inputs_test: Test coordinates (nx*ny, 2)
        nx, ny: Test grid dimensions
        nx_plot, ny_plot: Prediction grid dimensions
        N_test: Number of test samples
    """
    # Reshape data for plotting
    pred_2d = pred_mean_flat.reshape(N_test, ny_plot, nx_plot)
    true_2d = outputs_test.reshape(N_test, ny, nx)
    initial_2d = operator_inputs_test.reshape(N_test, ny, nx)

    # Extract coordinates for plotting
    plot_coords_2d = spatial_inputs_plot.reshape(ny_plot, nx_plot, 2)
    test_coords_2d = spatial_inputs_test.reshape(ny, nx, 2)

    X_plot = plot_coords_2d[:, :, 0]
    Y_plot = plot_coords_2d[:, :, 1]
    X_test = test_coords_2d[:, :, 0]
    Y_test = test_coords_2d[:, :, 1]

    # Create subplots: 3 rows (initial, true, predicted) × N_test columns
    fig, axes = plt.subplots(3, N_test, figsize=figsize)
    if N_test == 1:
        axes = axes.reshape(-1, 1)

    # Find global colorbar limits for consistency
    all_data = [initial_2d, true_2d, pred_2d]
    vmin = min(data.min() for data in all_data)
    vmax = max(data.max() for data in all_data)

    for i in range(N_test):
        # Plot initial condition
        axes[0, i].contourf(
            X_test,
            Y_test,
            initial_2d[i],
            levels=20,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, i].set_title(f"Initial Condition\nSample {i}")
        axes[0, i].set_aspect("equal")
        if i == 0:
            axes[0, i].set_ylabel("y")

        # Plot true solution
        axes[1, i].contourf(
            X_test, Y_test, true_2d[i], levels=20, cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        axes[1, i].set_title(f"True Solution\nSample {i}")
        axes[1, i].set_aspect("equal")
        if i == 0:
            axes[1, i].set_ylabel("y")

        # Plot prediction
        im3 = axes[2, i].contourf(
            X_plot, Y_plot, pred_2d[i], levels=20, cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        axes[2, i].set_title(f"Predicted Solution\nSample {i}")
        axes[2, i].set_xlabel("x")
        axes[2, i].set_aspect("equal")
        if i == 0:
            axes[2, i].set_ylabel("y")

    # Add colorbar to the last column
    divider = make_axes_locatable(axes[2, -1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im3, cax=cax)

    plt.tight_layout()
    return fig, axes


def plot_kernel_predictions_1d_thesis(
    pred_mean_flat,
    pred_cov,
    outputs_test,
    operator_inputs_test,
    spatial_inputs_plot,
    spatial_inputs_test,
    nx,
    nx_plot,
    N_test,
    figsize=(15, 10),
    plot_true_output: bool = True,
    title: str | None = None,
):
    # Reshape
    pred_mean_flat = pred_mean_flat.reshape(N_test, nx_plot)
    outputs_test = outputs_test.reshape(N_test, nx)
    inputs_test = operator_inputs_test.reshape(N_test, nx)

    # Grids
    grid_plot = spatial_inputs_plot.reshape(nx_plot)
    grid_test = spatial_inputs_test.reshape(nx)

    # Ensure ascending for interp (harmless if already sorted)
    sort_test = np.argsort(np.asarray(grid_test))
    grid_test_sorted = jnp.asarray(np.asarray(grid_test)[sort_test])

    sort_plot = np.argsort(np.asarray(grid_plot))
    grid_plot_sorted = jnp.asarray(np.asarray(grid_plot)[sort_plot])

    i = 0  # choose which test case to show

    # Interpolate 25-pt signals to 100-pt plot grid
    ic_interp = jnp.interp(
        grid_plot_sorted, grid_test_sorted, inputs_test[i][sort_test]
    )
    if plot_true_output:
        ytrue_interp = jnp.interp(
            grid_plot_sorted, grid_test_sorted, outputs_test[i][sort_test]
        )

    # GP stats already on plot grid (nx_plot)
    pred_mean_sorted = pred_mean_flat[i][sort_plot]
    variance = jnp.diag(pred_cov[i, :, i, :])
    std_sorted = jnp.sqrt(jnp.maximum(variance, 1e-8))[sort_plot].block_until_ready()

    # ---- Plot only on the 100-pt grid ----
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        np.asarray(grid_plot_sorted),
        np.asarray(ic_interp),
        color=rgb.tue_blue,
        label="Initial Condition (t=0)",
    )

    if plot_true_output:
        ax.plot(
            np.asarray(grid_plot_sorted),
            np.asarray(ytrue_interp),
            color=rgb.tue_green,
            linestyle="--",
            label="True Target",
        )

    ax.plot(
        np.asarray(grid_plot_sorted),
        np.asarray(pred_mean_sorted),
        color=rgb.pn_red,
        label="GP Prediction",
    )

    ax.fill_between(
        np.asarray(grid_plot_sorted),
        np.asarray(pred_mean_sorted - 2 * std_sorted),
        np.asarray(pred_mean_sorted + 2 * std_sorted),
        color=rgb.tue_lightblue,
        alpha=0.2,
        label="95% Confidence",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    if title:
        ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_kernel_predictions_1d(
    pred_mean_flat,
    pred_cov,
    outputs_test,
    operator_inputs_test,
    spatial_inputs_plot,
    spatial_inputs_test,
    nx,
    nx_plot,
    N_test,
    figsize=(15, 10),
    show_train_grid: bool = True,
    grid_tick_fraction: float = 0.07,
    show_axis_grid: bool = False,
):
    # Reshape data for plotting
    pred_mean_flat = pred_mean_flat.reshape(N_test, nx_plot)
    outputs_test = outputs_test.reshape(N_test, nx)
    inputs_test = operator_inputs_test.reshape(N_test, nx)

    # Extract coordinates for plotting
    grid_plot = spatial_inputs_plot.reshape(nx_plot, 1)[:, 0]
    grid_test = spatial_inputs_test.reshape(nx, 1)[:, 0]

    # Create subplots: 3 rows (initial, true, predicted) × N_test columns

    for i in range(N_test):
        variance = jnp.diag(pred_cov[i, :, i, :])
        std_dev = jnp.sqrt(jnp.maximum(variance, 1e-8)).block_until_ready()

        # Plot initial condition
        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        plt.plot(grid_test, inputs_test[i], "r")
        plt.title("Initial Condition (t=0)")
        plt.xlabel("x")
        plt.ylabel("u(x, 0)")
        ax1.grid(show_axis_grid)

        # Plot prediction at time T
        ax = plt.subplot(1, 2, 2)
        ax.plot(grid_test, outputs_test[i], "r", label="True Solution")
        ax.plot(grid_plot, pred_mean_flat[i], "b--", label="GP Prediction")
        plt.fill_between(
            grid_plot,
            pred_mean_flat[i] - 2 * std_dev,
            pred_mean_flat[i] + 2 * std_dev,
            color="blue",
            alpha=0.2,
            label="95% Confidence",
        )

        plt.show()

        # Visually mark training spatial grid locations to highlight uncertainty dips
        if show_train_grid:
            ymin, ymax = ax.get_ylim()
            dh = grid_tick_fraction * (ymax - ymin)
            # short ticks at the bottom of the axis for each training x
            ax.vlines(
                grid_test,
                ymin=ymin,
                ymax=ymin + dh,
                colors="k",
                linewidth=0.5,
                alpha=0.5,
            )

        plt.title(f"Solution at t={1.0}")
        plt.xlabel("x")
        plt.ylabel("u(x, T)")
        plt.legend()
        ax.grid(show_axis_grid)

        plt.tight_layout()
        plt.savefig(f"heat_gp_test_{i}.png")
        plt.show()

    return None, None


def plot_samples_2d(
    samples,
    outputs_test,
    operator_inputs_test,
    spatial_inputs_plot,
    spatial_inputs_test,
    nx,
    ny,
    nx_plot,
    ny_plot,
    N_test,
    test_idx=0,
    figsize=(15, 10),
):
    """Visualize posterior samples alongside the corresponding truth."""
    num_samples = samples.shape[0]
    outputs_test_2d = outputs_test.reshape(N_test, ny, nx)
    operator_inputs_test_2d = operator_inputs_test.reshape(N_test, ny, nx)
    plot_coords_2d = spatial_inputs_plot.reshape(ny_plot, nx_plot, 2)
    test_coords_2d = spatial_inputs_test.reshape(ny, nx, 2)

    X_plot = plot_coords_2d[:, :, 0]
    Y_plot = plot_coords_2d[:, :, 1]
    X_test = test_coords_2d[:, :, 0]
    Y_test = test_coords_2d[:, :, 1]

    fig, axes = plt.subplots(3, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    vmin = min(operator_inputs_test_2d.min(), outputs_test_2d.min(), samples.min())
    vmax = max(operator_inputs_test_2d.max(), outputs_test_2d.max(), samples.max())

    for i in range(num_samples):
        axes[0, i].contourf(
            X_test,
            Y_test,
            operator_inputs_test_2d[test_idx],
            levels=20,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, i].set_title(f"Initial Condition\nSample {i + 1}")
        axes[0, i].set_aspect("equal")
        if i == 0:
            axes[0, i].set_ylabel("y")

        axes[1, i].contourf(
            X_test,
            Y_test,
            outputs_test_2d[test_idx],
            levels=20,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, i].set_title(f"True Solution\nSample {i + 1}")
        axes[1, i].set_aspect("equal")
        if i == 0:
            axes[1, i].set_ylabel("y")

        im3 = axes[2, i].contourf(
            X_plot,
            Y_plot,
            samples[i, test_idx],
            levels=20,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        axes[2, i].set_title(f"Posterior Sample\nSample {i + 1}")
        axes[2, i].set_xlabel("x")
        axes[2, i].set_aspect("equal")
        if i == 0:
            axes[2, i].set_ylabel("y")

    divider = make_axes_locatable(axes[2, -1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im3, cax=cax)
    plt.tight_layout()
    return fig, axes


def plot_error_analysis_2d(
    pred_mean_flat,
    outputs_test,
    spatial_inputs_plot,
    spatial_inputs_test,
    nx,
    ny,
    nx_plot,
    ny_plot,
    N_test,
    figsize=(12, 8),
):
    """Plot absolute and relative error heatmaps for 2D predictions."""
    from scipy.interpolate import griddata

    pred_2d = pred_mean_flat.reshape(N_test, ny_plot, nx_plot)
    true_2d = outputs_test.reshape(N_test, ny, nx)

    plot_coords_2d = spatial_inputs_plot.reshape(ny_plot, nx_plot, 2)
    test_coords_2d = spatial_inputs_test.reshape(ny, nx, 2)

    X_plot = plot_coords_2d[:, :, 0]
    Y_plot = plot_coords_2d[:, :, 1]

    true_interp_2d = np.zeros_like(pred_2d)

    for i in range(N_test):
        points = test_coords_2d.reshape(-1, 2)
        values = true_2d[i].flatten()
        xi = spatial_inputs_plot
        interp_values = griddata(points, values, xi, method="cubic", fill_value=0)
        true_interp_2d[i] = interp_values.reshape(ny_plot, nx_plot)

    errors = pred_2d - true_interp_2d
    relative_errors = np.abs(errors) / (np.abs(true_interp_2d) + 1e-8)

    fig, axes = plt.subplots(2, N_test, figsize=figsize)
    if N_test == 1:
        axes = axes.reshape(-1, 1)

    for i in range(N_test):
        im1 = axes[0, i].contourf(
            X_plot, Y_plot, np.abs(errors[i]), levels=20, cmap="Reds"
        )
        axes[0, i].set_title(f"Absolute Error\nSample {i}")
        axes[0, i].set_aspect("equal")
        if i == 0:
            axes[0, i].set_ylabel("y")

        im2 = axes[1, i].contourf(
            X_plot, Y_plot, relative_errors[i], levels=20, cmap="Reds"
        )
        axes[1, i].set_title(f"Relative Error\nSample {i}")
        axes[1, i].set_xlabel("x")
        axes[1, i].set_aspect("equal")
        if i == 0:
            axes[1, i].set_ylabel("y")

        if i == N_test - 1:
            divider1 = make_axes_locatable(axes[0, i])
            cax1 = divider1.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im1, cax=cax1)

            divider2 = make_axes_locatable(axes[1, i])
            cax2 = divider2.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im2, cax=cax2)

    plt.tight_layout()

    return fig, axes


def plot_uncertainty_2d(
    pred_cov,
    spatial_inputs_plot,
    nx_plot,
    ny_plot,
    N_test,
    figsize=(12, 8),
):
    """Visualize posterior standard deviation for 2D predictions."""
    pred_std = np.zeros((N_test, ny_plot, nx_plot))
    for i in range(N_test):
        cov = pred_cov[i, :, i, :]
        stddev = jnp.sqrt(jnp.maximum(jnp.diag(cov), 1e-12))
        pred_std[i] = stddev.reshape(ny_plot, nx_plot)

    plot_coords_2d = spatial_inputs_plot.reshape(ny_plot, nx_plot, 2)
    X_plot = plot_coords_2d[:, :, 0]
    Y_plot = plot_coords_2d[:, :, 1]

    fig, axes = plt.subplots(1, N_test, figsize=figsize)
    if N_test == 1:
        axes = np.array([axes])

    for i in range(N_test):
        im = axes[i].contourf(X_plot, Y_plot, pred_std[i], levels=20, cmap="Purples")
        axes[i].set_title(f"Posterior Std\nSample {i}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].set_aspect("equal")

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    return fig, axes


def plot_marginal_analysis_2d(
    pred_mean_flat,
    outputs_test,
    spatial_inputs_plot,
    spatial_inputs_test,
    nx,
    ny,
    nx_plot,
    ny_plot,
    N_test,
    figsize=(14, 10),
):
    """Compare predicted and true marginals along several slices."""
    from scipy.interpolate import griddata

    pred_2d = pred_mean_flat.reshape(N_test, ny_plot, nx_plot)
    true_2d = outputs_test.reshape(N_test, ny, nx)

    plot_coords_2d = spatial_inputs_plot.reshape(ny_plot, nx_plot, 2)
    test_coords_2d = spatial_inputs_test.reshape(ny, nx, 2)

    x_plot = plot_coords_2d[0, :, 0]
    y_plot = plot_coords_2d[:, 0, 1]

    true_interp_2d = np.zeros_like(pred_2d)
    for i in range(N_test):
        points = test_coords_2d.reshape(-1, 2)
        values = true_2d[i].flatten()
        xi = spatial_inputs_plot
        interp_values = griddata(points, values, xi, method="linear", fill_value=0)
        true_interp_2d[i] = interp_values.reshape(ny_plot, nx_plot)

    colors = plt.cm.viridis(np.linspace(0, 1, N_test))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes[0, 0].set_title("Horizontal Cross-section (y mid)")
    mid_y_idx = ny_plot // 2
    for i in range(N_test):
        color = colors[i % len(colors)]
        axes[0, 0].plot(
            x_plot,
            pred_2d[i, mid_y_idx, :],
            color=color,
            linewidth=2,
            label=f"Pred {i}",
        )
        axes[0, 0].plot(
            x_plot,
            true_interp_2d[i, mid_y_idx, :],
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"True {i}",
        )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Temperature")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    mid_x_idx = nx_plot // 2
    axes[0, 1].set_title(f"Vertical Cross-section (x = {x_plot[mid_x_idx]:.3f})")
    for i in range(N_test):
        color = colors[i % len(colors)]
        axes[0, 1].plot(
            y_plot,
            pred_2d[i, :, mid_x_idx],
            color=color,
            linewidth=2,
            label=f"Predicted {i}",
        )
        axes[0, 1].plot(
            y_plot,
            true_interp_2d[i, :, mid_x_idx],
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"True {i}",
        )
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("Temperature")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Predicted vs True (All Points)")
    all_pred = pred_2d.flatten()
    all_true = true_interp_2d.flatten()
    mask = np.isfinite(all_pred) & np.isfinite(all_true)
    all_pred_clean = all_pred[mask]
    all_true_clean = all_true[mask]

    axes[1, 0].scatter(all_true_clean, all_pred_clean, alpha=0.6, s=20)
    min_val = min(all_true_clean.min(), all_pred_clean.min())
    max_val = max(all_true_clean.max(), all_pred_clean.max())
    axes[1, 0].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1, 0].set_xlabel("True Values")
    axes[1, 0].set_ylabel("Predicted Values")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if len(all_pred_clean) > 0:
        r2 = 1 - np.sum((all_true_clean - all_pred_clean) ** 2) / np.sum(
            (all_true_clean - np.mean(all_true_clean)) ** 2
        )
        axes[1, 0].text(
            0.05,
            0.95,
            f"R² = {r2:.4f}",
            transform=axes[1, 0].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )
    else:
        r2 = np.nan

    axes[1, 1].set_title("Error Along Diagonal")
    diag_indices = [(i, i) for i in range(min(ny_plot, nx_plot))]
    diag_coords = np.sqrt(2) * np.linspace(0, 1, len(diag_indices))

    for i in range(N_test):
        color = colors[i % len(colors)]
        diag_pred = [pred_2d[i, idx[0], idx[1]] for idx in diag_indices]
        diag_true = [true_interp_2d[i, idx[0], idx[1]] for idx in diag_indices]
        diag_error = np.array(diag_pred) - np.array(diag_true)
        axes[1, 1].plot(
            diag_coords,
            diag_error,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"Error {i}",
        )

    axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1, 1].set_xlabel("Diagonal Position (normalized)")
    axes[1, 1].set_ylabel("Prediction Error")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    np.mean(np.abs(all_pred_clean - all_true_clean))
    np.sqrt(np.mean((all_pred_clean - all_true_clean) ** 2))
    if len(all_pred_clean) > 0:
        pass

    return fig, axes


def plot_profile_comparison_2d(
    pred_mean_flat,
    outputs_test,
    spatial_inputs_plot,
    spatial_inputs_test,
    nx,
    ny,
    nx_plot,
    ny_plot,
    N_test,
    sample_idx=0,
    figsize=(12, 8),
):
    """Detailed profile comparison for a single sample with multiple slices."""
    from scipy.interpolate import griddata

    if sample_idx >= N_test:
        sample_idx = 0

    pred_2d = pred_mean_flat.reshape(N_test, ny_plot, nx_plot)
    true_2d = outputs_test.reshape(N_test, ny, nx)

    plot_coords_2d = spatial_inputs_plot.reshape(ny_plot, nx_plot, 2)
    test_coords_2d = spatial_inputs_test.reshape(ny, nx, 2)

    x_plot = plot_coords_2d[0, :, 0]
    y_plot = plot_coords_2d[:, 0, 1]

    points = test_coords_2d.reshape(-1, 2)
    values = true_2d[sample_idx].flatten()
    xi = spatial_inputs_plot
    interp_values = griddata(points, values, xi, method="linear", fill_value=0)
    true_interp_2d = interp_values.reshape(ny_plot, nx_plot)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"Profile Analysis - Sample {sample_idx}", fontsize=16)

    y_positions = [ny_plot // 4, ny_plot // 2, 3 * ny_plot // 4]
    y_labels = ["Bottom", "Middle", "Top"]

    for i, (y_idx, label) in enumerate(zip(y_positions, y_labels, strict=False)):
        axes[0, i].set_title(f"{label} (y = {y_plot[y_idx]:.3f})")
        axes[0, i].plot(
            x_plot, pred_2d[sample_idx, y_idx, :], "b-", linewidth=2, label="Predicted"
        )
        axes[0, i].plot(
            x_plot, true_interp_2d[y_idx, :], "r--", linewidth=2, label="True"
        )
        axes[0, i].set_xlabel("x")
        axes[0, i].set_ylabel("Temperature")
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

    x_positions = [nx_plot // 4, nx_plot // 2, 3 * nx_plot // 4]
    x_labels = ["Left", "Middle", "Right"]

    for i, (x_idx, label) in enumerate(zip(x_positions, x_labels, strict=False)):
        axes[1, i].set_title(f"{label} (x = {x_plot[x_idx]:.3f})")
        axes[1, i].plot(
            y_plot, pred_2d[sample_idx, :, x_idx], "b-", linewidth=2, label="Predicted"
        )
        axes[1, i].plot(
            y_plot, true_interp_2d[:, x_idx], "r--", linewidth=2, label="True"
        )
        axes[1, i].set_xlabel("y")
        axes[1, i].set_ylabel("Temperature")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes
