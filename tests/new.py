# new.py

import heapq
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

import linox
from linox import Kronecker, LinearOperator, Matrix
from linox._arithmetic import leigh
from linox.utils import as_linop


def topk_eigh_kronecker(
    factors: Sequence[LinearOperator | jax.Array],
    k: int,
    *,
    largest: bool = True,
) -> tuple[jax.Array, LinearOperator]:
    """Compute top-k or bottom-k eigenvalues/vectors of a Kronecker product.

    Uses a heap-based best-first search on the monotone grid of eigenvalue
    products. Avoids O(∏nᵢ) enumeration when k << ∏nᵢ.

    Args:
        factors: Sequence of symmetric LinearOperators or arrays.
        k: Number of eigenvalues to return.
        largest: If True, return largest k eigenvalues; else smallest k.

    Returns:
        eigenvalues: Array of shape (k,) with the k largest/smallest eigenvalues.
        eigenvectors: LinearOperator representing the k eigenvectors (lazy Kronecker).
    """
    factors = [as_linop(f) for f in factors]
    d = len(factors)

    factor_eigs = []
    factor_vecs = []
    sort_indices = []

    for A in factors:
        w, Q = leigh(A)
        order = jnp.argsort(-w) if largest else jnp.argsort(w)
        factor_eigs.append(w[order])
        sort_indices.append(order)
        factor_vecs.append(Q)

    sizes = [len(w) for w in factor_eigs]

    def compute_eigenvalue(indices: tuple[int, ...]) -> float:
        prod = 1.0
        for i, idx in enumerate(indices):
            prod = prod * factor_eigs[i][idx]
        return prod

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

    eigenvectors = _build_kronecker_eigenvectors(
        factor_vecs, sort_indices, selected_indices
    )

    return jnp.array(eigenvalues), eigenvectors


def _build_kronecker_eigenvectors(
    factor_vecs: list[LinearOperator],
    sort_indices: list[jax.Array],
    selected_indices: list[tuple[int, ...]],
) -> LinearOperator:
    """Build eigenvector matrix from selected Kronecker indices.

    For each selected (i, j, ...) tuple, the eigenvector is:
        Q_A[:, sort_A[i]] ⊗ Q_B[:, sort_B[j]] ⊗ ...

    Returns a Matrix wrapping the dense (n, k) eigenvector array.
    """
    d = len(factor_vecs)
    eigvecs = []
    for idx_tuple in selected_indices:
        vec = factor_vecs[0].todense()[:, sort_indices[0][idx_tuple[0]]]
        for i in range(1, d):
            col_idx = sort_indices[i][idx_tuple[i]]
            # vec = linox.kron(vec, )
            vec = jnp.kron(vec, factor_vecs[i].todense()[:, col_idx])
        eigvecs.append(vec)

    return Matrix(jnp.stack(eigvecs, axis=1))


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    A = jax.random.normal(k1, (4, 4))
    A = A @ A.T + jnp.eye(4) * 0.1
    B = jax.random.normal(k2, (3, 3))
    B = B @ B.T + jnp.eye(3) * 0.1

    print("=== Two-factor Kronecker product ===")
    op_A = Matrix(A)
    op_B = Matrix(B)
    linop = Kronecker(A=op_A, B=op_B)

    print(f"A shape: {op_A.shape}, B shape: {op_B.shape}")
    print(f"Full Kronecker size: {linop.shape[0]}")

    full_eigs, full_Q = leigh(linop)

    k = 5
    print(f"\nComputing top-{k} eigenvalues using heap algorithm...")
    topk_vals, topk_vecs = topk_eigh_kronecker([op_A, op_B], k=k, largest=True)

    print(f"\nTop-{k} via heap:      {topk_vals}")
    print(f"Top-{k} via full leigh: {jnp.sort(full_eigs)[::-1][:k]}")

    print(f"\nBottom-{k} eigenvalues...")
    botk_vals, botk_vecs = topk_eigh_kronecker([op_A, op_B], k=k, largest=False)

    print(f"Bottom-{k} via heap:      {botk_vals}")
    print(f"Bottom-{k} via full leigh: {jnp.sort(full_eigs)[:k]}")

    print("\n\n=== Verification: Eigenvector correctness (matrix-free) ===")
    for i in range(min(3, k)):
        lam = topk_vals[i]
        v = topk_vecs.todense()[:, i]
        Av = linop @ v
        residual = jnp.linalg.norm(Av - lam * v)
        print(f"Eigenpair {i}: λ={lam:.6f}, ||Av - λv|| = {residual:.2e}")

    print("\n\n=== Three-factor Kronecker product ===")
    C = jax.random.normal(k3, (3, 3))
    C = C @ C.T + jnp.eye(3) * 0.1
    op_C = Matrix(C)

    linop_3 = Kronecker(Kronecker(op_A, op_B), op_C)
    print(f"Shapes: A={op_A.shape}, B={op_B.shape}, C={op_C.shape}")
    print(f"Full Kronecker size: {linop_3.shape[0]}")

    full_eigs_3, _ = leigh(linop_3)

    k = 8
    print(f"\nComputing top-{k} eigenvalues...")
    topk_vals_3, topk_vecs_3 = topk_eigh_kronecker(
        [op_A, op_B, op_C], k=k, largest=True
    )

    print(f"\nTop-{k} via heap:      {topk_vals_3}")
    print(f"Top-{k} via full leigh: {jnp.sort(full_eigs_3)[::-1][:k]}")

    print("\n\n=== Verify 3-factor eigenvectors (matrix-free matvec) ===")
    linop_3_flat = Kronecker(op_A, Kronecker(op_B, op_C))
    for i in range(min(3, k)):
        lam = topk_vals_3[i]
        v = topk_vecs_3.todense()[:, i]
        Av = linop_3_flat @ v
        residual = jnp.linalg.norm(Av - lam * v)
        print(f"Eigenpair {i}: λ={lam:.6f}, ||Av - λv|| = {residual:.2e}")

    print("\n\n=== Large-scale example (no densification needed) ===")
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)

    D = jax.random.normal(k1, (50, 50))
    D = D @ D.T + jnp.eye(50) * 0.1
    E = jax.random.normal(k2, (50, 50))
    E = E @ E.T + jnp.eye(50) * 0.1
    F = jax.random.normal(k3, (50, 50))
    F = F @ F.T + jnp.eye(50) * 0.1

    op_D, op_E, op_F = Matrix(D), Matrix(E), Matrix(F)

    full_size = 50 * 50 * 50
    print(f"3 factors of 50x50 → Kronecker size: {full_size}")
    print(f"Dense Kronecker would need: {full_size**2 * 8 / 1e9:.2f} GB")
    print("(We never build it!)")

    k_large = 20
    topk_large, vecs_large = topk_eigh_kronecker(
        [op_D, op_E, op_F], k=k_large, largest=True
    )
    print(f"\nTop-{k_large} eigenvalues computed efficiently:")
    print(f"  {topk_large[:5]} ...")

    print("\nVerify with matrix-free matvec:")
    linop_large = Kronecker(op_D, Kronecker(op_E, op_F))
    v0 = vecs_large.todense()[:, 0]
    Av0 = linop_large @ v0
    residual = jnp.linalg.norm(Av0 - topk_large[0] * v0)
    print(f"  Eigenpair 0: λ={topk_large[0]:.6f}, ||Av - λv|| = {residual:.2e}")
