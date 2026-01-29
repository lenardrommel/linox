# data.py

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Heat1dDataGenerator:
    """Simple 1D heat equation data generator."""

    def __init__(self, x_range=(0, np.pi), nx=25, T=1.0, alpha=0.5, N_samples=100) -> None:
        self.x_range = x_range
        self.nx = nx
        self.T = T
        self.alpha = alpha
        self.N_samples = N_samples

        # Initialize grid
        self.x_grid = np.linspace(x_range[0], x_range[1], nx)
        self.dx = self.x_grid[1] - self.x_grid[0]

        # Will hold data once generated
        self.u0_samples = None
        self.data = None

    def generate_initial_conditions(self, n_modes=3, seed=42) -> None:
        """Generate random initial conditions using Fourier series."""
        np.random.seed(seed)

        # Generate random coefficients for sine functions
        L = self.x_range[1] - self.x_range[0]

        # Initialize array for initial conditions
        self.u0_samples = np.zeros((self.N_samples, self.nx))

        for i in range(self.N_samples):
            # Generate random coefficients
            coefs = np.random.normal(0, 1, n_modes)

            # Construct initial condition as sum of sine modes
            u0 = np.zeros(self.nx)
            for n, coef in enumerate(coefs, 1):
                u0 += coef * np.sin(n * np.pi * self.x_grid / L)

            self.u0_samples[i] = u0

    def solve_pde(self) -> None:
        """Solve heat equation analytically for all initial conditions."""
        if self.u0_samples is None:
            msg = "Initial conditions not yet generated"
            raise ValueError(msg)

        L = self.x_range[1] - self.x_range[0]
        self.data = np.zeros_like(self.u0_samples)

        # For each sample, solve the heat equation analytically
        for i, u0 in enumerate(self.u0_samples):
            # Compute Fourier coefficients of the initial condition
            coeffs = []
            for n in range(1, 20):  # Use 20 modes for approximation
                bn = (
                    2
                    / L
                    * np.trapezoid(
                        u0 * np.sin(n * np.pi * self.x_grid / L), self.x_grid
                    )
                )
                coeffs.append(bn)

            # Compute solution at time T
            solution = np.zeros(self.nx)
            for n, bn in enumerate(coeffs, 1):
                # Heat equation analytical solution
                solution += (
                    bn
                    * np.exp(-self.alpha * (n * np.pi / L) ** 2 * self.T)
                    * np.sin(n * np.pi * self.x_grid / L)
                )

            self.data[i] = solution


class Heat2dDataGenerator:
    """2D heat equation data generator."""

    def __init__(
        self,
        x_range=(0, np.pi),
        y_range=(0, np.pi),
        nx=25,
        ny=25,
        T=1.0,
        alpha=0.5,
        N_samples=100,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.nx = nx
        self.ny = ny
        self.T = T
        self.alpha = alpha
        self.N_samples = N_samples

        # Initialize grids
        self.x_grid = np.linspace(x_range[0], x_range[1], nx)
        self.y_grid = np.linspace(y_range[0], y_range[1], ny)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]

        # Create meshgrid for 2D operations
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

        # Will hold data once generated
        self.u0_samples = None
        self.data = None

    def generate_initial_conditions(self, n_modes_x=3, n_modes_y=3, seed=42) -> None:
        """Generate random initial conditions using 2D Fourier series."""
        np.random.seed(seed)

        Lx = self.x_range[1] - self.x_range[0]
        Ly = self.y_range[1] - self.y_range[0]

        # Initialize array for initial conditions
        self.u0_samples = np.zeros((self.N_samples, self.ny, self.nx))

        for i in range(self.N_samples):
            # Generate random coefficients for 2D modes
            coefs = np.random.normal(0, 1, (n_modes_y, n_modes_x))

            # Construct initial condition as sum of 2D sine modes
            u0 = np.zeros((self.ny, self.nx))
            for m in range(1, n_modes_x + 1):
                for n in range(1, n_modes_y + 1):
                    u0 += (
                        coefs[n - 1, m - 1]
                        * np.sin(m * np.pi * self.X / Lx)
                        * np.sin(n * np.pi * self.Y / Ly)
                    )

            self.u0_samples[i] = u0

    def solve_pde(self) -> None:
        """Solve 2D heat equation analytically for all initial conditions."""
        if self.u0_samples is None:
            msg = "Initial conditions not yet generated"
            raise ValueError(msg)

        Lx = self.x_range[1] - self.x_range[0]
        Ly = self.y_range[1] - self.y_range[0]

        self.data = np.zeros_like(self.u0_samples)

        # For each sample, solve the 2D heat equation analytically
        for i, u0 in enumerate(self.u0_samples):
            solution = np.zeros((self.ny, self.nx))

            # Use finite number of modes for approximation
            for m in range(1, 15):  # modes in x direction
                for n in range(1, 15):  # modes in y direction
                    # Compute 2D Fourier coefficient A_mn
                    sin_m = np.sin(m * np.pi * self.X / Lx)
                    sin_n = np.sin(n * np.pi * self.Y / Ly)
                    integrand = u0 * sin_m * sin_n

                    # 2D integration using trapezoidal rule
                    A_mn = (4 / (Lx * Ly)) * np.trapezoid(
                        np.trapezoid(integrand, self.x_grid), self.y_grid
                    )

                    # Add contribution to solution at time T
                    eigenvalue = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2
                    decay_factor = np.exp(-self.alpha * eigenvalue * self.T)
                    solution += A_mn * decay_factor * sin_m * sin_n

            self.data[i] = solution

    def plot_solution(self, sample_idx=0, figsize=(12, 5)):
        """Plot initial condition and solution for a specific sample."""
        if self.u0_samples is None or self.data is None:
            msg = "Data not yet generated. Run generate_initial_conditions() and solve_pde() first."
            raise ValueError(
                msg
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot initial condition
        im1 = ax1.contourf(
            self.X, self.Y, self.u0_samples[sample_idx], levels=20, cmap="RdBu_r"
        )
        ax1.set_title(f"Initial Condition (t=0)\nSample {sample_idx}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")

        # Add colorbar for initial condition
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im1, cax=cax1)

        # Plot solution at time T
        im2 = ax2.contourf(
            self.X, self.Y, self.data[sample_idx], levels=20, cmap="RdBu_r"
        )
        ax2.set_title(f"Solution at t={self.T}\nα={self.alpha}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_aspect("equal")

        # Add colorbar for solution
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax2)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_multiple_solutions(self, n_samples=4, figsize=(15, 12)):
        """Plot initial conditions and solutions for multiple samples."""
        if self.u0_samples is None or self.data is None:
            msg = "Data not yet generated. Run generate_initial_conditions() and solve_pde() first."
            raise ValueError(
                msg
            )

        n_samples = min(n_samples, self.N_samples)
        fig, axes = plt.subplots(2, n_samples, figsize=figsize)

        for i in range(n_samples):
            # Plot initial condition
            axes[0, i].contourf(
                self.X, self.Y, self.u0_samples[i], levels=15, cmap="RdBu_r"
            )
            axes[0, i].set_title(f"Initial (t=0)\nSample {i}")
            axes[0, i].set_aspect("equal")
            if i == 0:
                axes[0, i].set_ylabel("y")

            # Plot solution
            axes[1, i].contourf(
                self.X, self.Y, self.data[i], levels=15, cmap="RdBu_r"
            )
            axes[1, i].set_title(f"Solution (t={self.T})\nSample {i}")
            axes[1, i].set_xlabel("x")
            axes[1, i].set_aspect("equal")
            if i == 0:
                axes[1, i].set_ylabel("y")

        # Add colorbars
        plt.tight_layout()
        return fig, axes


def get_data(name: str):
    if name == "heat1d":
        data_gen = Heat1dDataGenerator(nx=50, N_samples=20, T=0.1, alpha=0.5)
        data_gen.generate_initial_conditions(n_modes=5)
        data_gen.solve_pde()
        return data_gen.x_grid, data_gen.u0_samples, data_gen.data
    if name == "heat2d":
        data_gen = Heat2dDataGenerator(nx=30, ny=30, N_samples=10, T=0.1, alpha=0.5)
        data_gen.generate_initial_conditions(n_modes_x=5, n_modes_y=5)
        data_gen.solve_pde()
        return data_gen.x_grid, data_gen.y_grid, data_gen.u0_samples, data_gen.data
    msg = f"Unknown dataset name: {name}"
    raise ValueError(msg)
