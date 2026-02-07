# Toeplitz Solver Implementation

Linox provides a JAX-compatible Toeplitz solver in `linox.linalg._toeplitz_solve`.

## Hybrid Approach

The solver uses a **hybrid approach** to combine the performance of SciPy's compiled Levinson recursion with JAX's composability:

1.  **Forward Pass**: Calls `scipy.linalg.solve_toeplitz` via `jax.pure_callback`.
    - This executes the efficient C implementation of Levinson recursion ($O(n^2)$).
    - It runs on the CPU host (callback).

2.  **Backward Pass (Gradients)**: Implements a custom VJP (Vector-Jacobian Product) in pure JAX.
    - The gradient of a Toeplitz solve $Tx = b$ involves solving transposed Toeplitz systems.
    - We re-use the Toeplitz structure to compute gradients efficiently without densifying ($O(n^2)$).

## Limitations

- **Device**: The forward pass uses a CPU callback. This is efficient for most workflows but may incur transfer overhead if data is on GPU/TPU.
- **Batching**: Supports `vmap` via sequential execution of the callback.

## Future Work

- **Pure JAX FFT Solver**: For very large $n$, an FFT-based $O(n \log n)$ solver (superfast Toeplitz solver) implemented in pure JAX would be preferred for GPU acceleration. This is planned for future milestones.
