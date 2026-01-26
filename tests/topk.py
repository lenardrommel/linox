# topk.py

"""Memory-efficient top-k eigendecomposition for Kronecker products.

Standalone version - no linox dependency.
Avoids O(∏nᵢ²) memory by never forming the full Kronecker product.
"""

import heapq
from collections.abc import Sequence
from typing import Callable

import jax
import jax.numpy as jnp

# ============================================================================
# Matrix-free operators for selected Kronecker eigenvectors
# ============================================================================


def make_kronecker_selected_eigvecs_ops(
    factor_vecs: list[jax.Array],  # List of (n_i, n_i) eigenvector matrices
    selected_indices: list[tuple[int, ...]],  # k tuples of indices
    sort_indices: list[jax.Array],  # Sorting permutations for each factor
) -> tuple[Callable, Callable, tuple[int, int]]:
    """Build matrix-free matvec/rmatvec for selected Kronecker eigenvectors.

    Returns:
        mv: Q_k @ alpha function
        rmv: Q_k.T @ v function
        shape: (n_total, k)
    """
    d = len(factor_vecs)
    k = len(selected_indices)
    factor_dims = [Q.shape[0] for Q in factor_vecs]
    n_total = int(jnp.prod(jnp.array(factor_dims)))

    # Precompute gathered columns for each factor: (n_i, k)
    gathered = []
    for i in range(d):
        idx_for_factor = jnp.array([
            sort_indices[i][sel[i]] for sel in selected_indices
        ])
        gathered.append(factor_vecs[i][:, idx_for_factor])

    if d == 2:
        # Optimized 2-factor case
        # kron(qA, qB)[i*nB + j] = qA[i] * qB[j]
        UA, UB = gathered[0], gathered[1]  # (nA, k), (nB, k)
        nA, nB = factor_dims[0], factor_dims[1]

        def mv(alpha: jax.Array) -> jax.Array:
            """Q_k @ alpha: result[i*nB + j] = sum_l UA[i,l] * UB[j,l] * alpha[l]"""
            squeeze = alpha.ndim == 1
            if squeeze:
                alpha = alpha[:, None]
            # Y[i, j, b] = sum_l UA[i, l] * alpha[l, b] * UB[j, l]
            Y = jnp.einsum("il,lb,jl->ijb", UA, alpha, UB)  # (nA, nB, batch)
            result = Y.reshape((nA * nB, -1))
            return result.squeeze(-1) if squeeze else result

        def rmv(v: jax.Array) -> jax.Array:
            """Q_k.T @ v: result[l] = sum_{i,j} UA[i,l] * UB[j,l] * v[i*nB + j]"""
            squeeze = v.ndim == 1
            if squeeze:
                v = v[None, :]
            X = v.reshape((v.shape[0], nA, nB))  # (batch, nA, nB)
            # result[b, l] = sum_{i,j} UA[i,l] * X[b,i,j] * UB[j,l]
            T = jnp.einsum("il,bij->blj", UA, X)  # (batch, k, nB)
            result = jnp.einsum("blj,jl->bl", T, UB)  # (batch, k)
            return result.squeeze(0) if squeeze else result

        return mv, rmv, (n_total, k)

    # General d-factor case
    def mv(alpha: jax.Array) -> jax.Array:
        squeeze = alpha.ndim == 1
        if squeeze:
            alpha = alpha[:, None]

        result = jnp.zeros((n_total, alpha.shape[1]), dtype=factor_vecs[0].dtype)
        for l in range(k):
            vec_l = gathered[0][:, l]
            for i in range(1, d):
                vec_l = jnp.kron(vec_l, gathered[i][:, l])
            result = result + vec_l[:, None] * alpha[l, :]

        return result.squeeze(-1) if squeeze else result

    def rmv(v: jax.Array) -> jax.Array:
        squeeze = v.ndim == 1
        if squeeze:
            v = v[None, :]

        result = jnp.zeros((v.shape[0], k), dtype=factor_vecs[0].dtype)
        for l in range(k):
            vec_l = gathered[0][:, l]
            for i in range(1, d):
                vec_l = jnp.kron(vec_l, gathered[i][:, l])
            result = result.at[:, l].set(v @ vec_l)

        return result.squeeze(0) if squeeze else result

    return mv, rmv, (n_total, k)


def make_kronecker_inv_sqrt_ops(
    factor_vecs: list[jax.Array],
    eigenvalues: jax.Array,
    selected_indices: list[tuple[int, ...]],
    sort_indices: list[jax.Array],
    eps: float = 1e-12,
) -> tuple[Callable, Callable]:
    """Build D_k @ alpha = Q_k @ diag(1/sqrt(λ)) @ alpha operators."""

    inv_sqrt_eigs = jnp.where(eigenvalues > eps, 1.0 / jnp.sqrt(eigenvalues), 0.0)
    mv_q, rmv_q, shape = make_kronecker_selected_eigvecs_ops(
        factor_vecs, selected_indices, sort_indices
    )

    def mv(alpha: jax.Array) -> jax.Array:
        scaled = inv_sqrt_eigs * alpha
        return mv_q(scaled)

    def rmv(v: jax.Array) -> jax.Array:
        qt_v = rmv_q(v)
        return inv_sqrt_eigs * qt_v

    return mv, rmv


# ============================================================================
# Heap-based top-k selection
# ============================================================================


def topk_eigh_kronecker(
    factors: Sequence[jax.Array],
    k: int,
    *,
    largest: bool = True,
) -> tuple[jax.Array, Callable, Callable, dict]:
    """Compute top-k or bottom-k eigenvalues/vectors of a Kronecker product.

    Uses heap-based best-first search. Avoids O(∏nᵢ²) memory.

    Note: Assumes PSD factors (non-negative eigenvalues).

    Args:
        factors: Sequence of symmetric PSD matrices.
        k: Number of eigenvalues to return.
        largest: If True, return largest k; else smallest k.

    Returns:
        eigenvalues: Array of shape (k,)
        mv: Q_k @ alpha function (matrix-free)
        rmv: Q_k.T @ v function (matrix-free)
        info: Dict with factor decompositions and indices for further use
    """
    d = len(factors)

    # Eigendecomposition of each factor
    factor_eigs = []
    factor_vecs = []
    sort_indices = []

    for A in factors:
        w, Q = jnp.linalg.eigh(A)
        order = jnp.argsort(-w) if largest else jnp.argsort(w)
        factor_eigs.append(w[order])
        sort_indices.append(order)
        factor_vecs.append(Q)

    sizes = [len(w) for w in factor_eigs]

    def compute_eigenvalue(indices: tuple[int, ...]) -> float:
        prod = 1.0
        for i, idx in enumerate(indices):
            prod = prod * factor_eigs[i][idx]
        return float(prod)

    # Heap search
    initial_idx = tuple(0 for _ in range(d))
    initial_val = compute_eigenvalue(initial_idx)

    heap = [(-initial_val, initial_idx)] if largest else [(initial_val, initial_idx)]
    visited = {initial_idx}

    eigenvalues = []
    selected_indices = []

    while len(eigenvalues) < k and heap:
        if largest:
            neg_val, idx = heapq.heappop(heap)
            val = -neg_val
        else:
            val, idx = heapq.heappop(heap)

        eigenvalues.append(val)
        selected_indices.append(idx)

        for dim in range(d):
            new_idx = list(idx)
            new_idx[dim] += 1
            new_idx = tuple(new_idx)

            if new_idx[dim] < sizes[dim] and new_idx not in visited:
                visited.add(new_idx)
                new_val = compute_eigenvalue(new_idx)
                if largest:
                    heapq.heappush(heap, (-new_val, new_idx))
                else:
                    heapq.heappush(heap, (new_val, new_idx))

    # Build matrix-free operators
    mv, rmv, shape = make_kronecker_selected_eigvecs_ops(
        factor_vecs, selected_indices, sort_indices
    )

    info = {
        "factor_eigs": factor_eigs,
        "factor_vecs": factor_vecs,
        "sort_indices": sort_indices,
        "selected_indices": selected_indices,
        "shape": shape,
    }

    return jnp.array(eigenvalues), mv, rmv, info


def build_dense_eigenvectors(info: dict) -> jax.Array:
    """Build dense eigenvector matrix from info dict (for verification only)."""
    factor_vecs = info["factor_vecs"]
    sort_indices = info["sort_indices"]
    selected_indices = info["selected_indices"]
    d = len(factor_vecs)

    cols = []
    for idx_tuple in selected_indices:
        vec = factor_vecs[0][:, sort_indices[0][idx_tuple[0]]]
        for i in range(1, d):
            col_idx = sort_indices[i][idx_tuple[i]]
            vec = jnp.kron(vec, factor_vecs[i][:, col_idx])
        cols.append(vec)

    return jnp.stack(cols, axis=1)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Create symmetric PSD matrices
    A = jax.random.normal(k1, (4, 4))
    A = A @ A.T + jnp.eye(4) * 0.1
    B = jax.random.normal(k2, (3, 3))
    B = B @ B.T + jnp.eye(3) * 0.1

    print("=== Two-factor Kronecker product (matrix-free) ===")
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    full_kron = jnp.kron(A, B)
    print(f"Full Kronecker size: {full_kron.shape[0]} x {full_kron.shape[0]}")

    # Full eigendecomposition for comparison
    full_eigs, full_Q = jnp.linalg.eigh(full_kron)

    k = 5
    print(f"\nComputing top-{k} eigenvalues using heap algorithm (matrix-free)...")
    topk_vals, mv, rmv, info = topk_eigh_kronecker([A, B], k=k, largest=True)

    print(f"\nTop-{k} via heap:       {topk_vals}")
    print(f"Top-{k} via full eigh:  {jnp.sort(full_eigs)[::-1][:k]}")

    print(f"\nBottom-{k} eigenvalues...")
    botk_vals, mv_bot, rmv_bot, _ = topk_eigh_kronecker([A, B], k=k, largest=False)

    print(f"Bottom-{k} via heap:       {botk_vals}")
    print(f"Bottom-{k} via full eigh:  {jnp.sort(full_eigs)[:k]}")

    print("\n=== Verification: Eigenvector correctness (matrix-free matvec) ===")
    for i in range(min(3, k)):
        lam = topk_vals[i]
        e_i = jnp.zeros(k).at[i].set(1.0)
        v = mv(e_i)  # Matrix-free!
        Av = full_kron @ v
        residual = jnp.linalg.norm(Av - lam * v)
        print(f"Eigenpair {i}: λ={lam:.6f}, ||Av - λv|| = {residual:.2e}")

    print("\n=== Verify transpose operation ===")
    test_vec = jax.random.normal(k1, (full_kron.shape[0],))
    qt_v = rmv(test_vec)
    print(f"Q_k.T @ v shape: {qt_v.shape} (should be ({k},))")

    # Verify Q.T @ v against dense
    Q_dense = build_dense_eigenvectors(info)
    qt_v_dense = Q_dense.T @ test_vec
    print(
        f"||Q_k.T @ v (matrix-free) - Q_k.T @ v (dense)||: {jnp.linalg.norm(qt_v - qt_v_dense):.2e}"
    )

    print("\n=== Three-factor Kronecker product ===")
    C = jax.random.normal(k3, (3, 3))
    C = C @ C.T + jnp.eye(3) * 0.1

    full_kron_3 = jnp.kron(jnp.kron(A, B), C)
    print(f"Shapes: A={A.shape}, B={B.shape}, C={C.shape}")
    print(f"Full Kronecker size: {full_kron_3.shape[0]} x {full_kron_3.shape[0]}")

    full_eigs_3, _ = jnp.linalg.eigh(full_kron_3)

    k = 8
    print(f"\nComputing top-{k} eigenvalues...")
    topk_vals_3, mv_3, rmv_3, info_3 = topk_eigh_kronecker([A, B, C], k=k, largest=True)

    print(f"\nTop-{k} via heap:       {topk_vals_3}")
    print(f"Top-{k} via full eigh:  {jnp.sort(full_eigs_3)[::-1][:k]}")

    print("\n=== Verify 3-factor eigenvectors ===")
    for i in range(min(3, k)):
        lam = topk_vals_3[i]
        e_i = jnp.zeros(k).at[i].set(1.0)
        v = mv_3(e_i)
        Av = full_kron_3 @ v
        residual = jnp.linalg.norm(Av - lam * v)
        print(f"Eigenpair {i}: λ={lam:.6f}, ||Av - λv|| = {residual:.2e}")

    print("\n=== Large-scale example (NO densification!) ===")
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)

    D = jax.random.normal(k1, (50, 50))
    D = D @ D.T + jnp.eye(50) * 0.1
    E = jax.random.normal(k2, (50, 50))
    E = E @ E.T + jnp.eye(50) * 0.1
    F = jax.random.normal(k3, (50, 50))
    F = F @ F.T + jnp.eye(50) * 0.1

    full_size = 50 * 50 * 50
    print(f"3 factors of 50x50 → Kronecker size: {full_size} x {full_size}")
    print(f"Dense Kronecker would need: {full_size**2 * 8 / 1e9:.2f} GB")
    print("(We never build it!)")

    k_large = 20
    topk_large, mv_large, rmv_large, info_large = topk_eigh_kronecker(
        [D, E, F], k=k_large, largest=True
    )
    print(f"\nTop-{k_large} eigenvalues computed efficiently:")
    print(f"  {topk_large[:5]} ...")

    print("\nVerify with matrix-free Kronecker matvec:")

    def kron_matvec(factors, v):
        """Matrix-free (A ⊗ B ⊗ C) @ v"""
        # Using vec property: kron(A,B,C) @ vec(X) involves reshaping
        d = len(factors)
        dims = [f.shape[0] for f in factors]

        # Reshape v to tensor
        X = v.reshape(dims)

        # Apply each factor along its axis
        for i, F in enumerate(factors):
            X = jnp.tensordot(F, X, axes=([1], [i]))
            # Move the result axis back to position i
            X = jnp.moveaxis(X, 0, i)

        return X.reshape(-1)

    e_0 = jnp.zeros(k_large).at[0].set(1.0)
    v0 = mv_large(e_0)
    Av0 = kron_matvec([D, E, F], v0)
    residual = jnp.linalg.norm(Av0 - topk_large[0] * v0)
    print(f"  Eigenpair 0: λ={topk_large[0]:.6f}, ||Av - λv|| = {residual:.2e}")

    print("\n=== Memory comparison ===")
    print(f"Dense eigenvector matrix would be: {full_size * k_large * 8 / 1e6:.2f} MB")
    print(
        f"Our operator stores: {3 * 50 * k_large * 8 / 1e3:.2f} KB (factor columns only)"
    )

    print("\n=== Inverse sqrt operator (for whitening) ===")
    mv_inv_sqrt, rmv_inv_sqrt = make_kronecker_inv_sqrt_ops(
        info["factor_vecs"], topk_vals, info["selected_indices"], info["sort_indices"]
    )
    z = jax.random.normal(k2, (5,))
    whitened = mv_inv_sqrt(z)
    print(f"D_k @ z shape: {whitened.shape}")
