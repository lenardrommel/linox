<div align="center">
  <img src="linox_logo.png" alt="linox logo" width="180" />
</div>

<h1 align="center"><code>linox</code></h1>

<p align="center">
  <strong>Structured, matrix-free linear algebra in JAX.</strong>
</p>

<p align="center">
  <a href="https://linox.lenardrommel.com">Documentation</a> ·
  <a href="https://github.com/lenardrommel/linox">Source</a> ·
  <a href="https://github.com/lenardrommel/linox/issues">Issues</a>
</p>

---

`linox` represents a linear map as an **operator** rather than a matrix. An operator
knows how to apply itself, and often knows more than that — that it is diagonal, a
Kronecker product, a low-rank update — and uses that structure to avoid ever forming
the dense matrix.

```python
import jax.numpy as jnp
import linox

d = linox.Diagonal(jnp.array([1.0, 2.0, 3.0, 4.0]))
op = d + 0.5 * linox.Identity(4)   # nothing is computed yet

x = op @ jnp.ones(4)               # one elementwise multiply, no 4x4 matrix
assert x.shape == (4,)
```

Two 1000×1000 Kronecker factors describe a 10⁶×10⁶ matrix. `linox` solves against it
without allocating one.

## Install

```bash
pip install linox
```

Python 3.10+. From source:

```bash
git clone https://github.com/lenardrommel/linox.git
cd linox
pip install -e ".[test]"
```

## What it looks like

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (3, 3))
b = jax.random.normal(jax.random.fold_in(key, 1), (3, 3))
spd_a, spd_b = a @ a.T + 3 * jnp.eye(3), b @ b.T + 3 * jnp.eye(3)

# A 9x9 operator held as two 3x3 factors.
kron = linox.Kronecker(linox.Matrix(spd_a), linox.Matrix(spd_b))

x = linox.solve(kron, jnp.ones(9))          # solved through the factors
sign, logdet = linox.slogdet(kron)          # det(A(x)B) = det(A)^n det(B)^m

assert jnp.linalg.norm(jnp.kron(spd_a, spd_b) @ x - jnp.ones(9)) < 1e-8
assert jnp.allclose(logdet, jnp.linalg.slogdet(jnp.kron(spd_a, spd_b))[1], atol=1e-8)
```

Arithmetic is lazy and rewrites itself. `A + s * I` becomes a single operator with
fast spectral methods rather than a generic sum:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = linox.Matrix(dense @ dense.T + 4 * jnp.eye(4))

regularised = spd + 0.1 * linox.Identity(4)
assert type(regularised).__name__ == "IsotropicAdditiveLinearOperator"
```

Failure is reported rather than hidden. A singular system raises instead of returning
finite nonsense:

```python
import jax
import jax.numpy as jnp
import linox

u = jax.random.normal(jax.random.PRNGKey(0), (6, 3))
singular = linox.Matrix(u @ u.T)            # rank 3

try:
    linox.solve(singular, jnp.ones(6))
    raise AssertionError("expected a failure")
except linox.LinearSolveError:
    pass

x, info = linox.solve(singular, jnp.ones(6), throw=False, return_info=True)
assert info.result != linox.RESULTS.successful
```

And it is all JAX — operators are pytrees, so `jit`, `grad` and `vmap` work:

```python
import jax
import jax.numpy as jnp
import linox

key = jax.random.PRNGKey(0)
dense = jax.random.normal(key, (4, 4))
spd = dense @ dense.T + 4 * jnp.eye(4)
b = jnp.ones(4)

assert jax.jit(lambda m, v: linox.solve(linox.Matrix(m), v))(spd, b).shape == (4,)
assert jax.grad(lambda v: linox.solve(linox.Matrix(spd), v).sum())(b).shape == (4,)
assert jax.vmap(lambda v: linox.solve(linox.Matrix(spd), v))(jnp.ones((5, 4))).shape == (5, 4)
```

## What is in it

**Operators** — dense, diagonal, identity, zero, ones, scalar, permutation, Toeplitz,
Kronecker, block, low-rank and diagonal-plus-low-rank, eigendecomposed, kernel
matrices, and the composites that arithmetic produces.

**Solvers** — structured exact solves, preconditioned conjugate gradients, LSMR, and
pseudo-inverses, all reporting whether they succeeded.

**Decompositions** — `eigh`, `svd` (full or matrix-free partial), `qr`, `cholesky`,
square roots.

**Matrix-free algorithms** — Lanczos, Arnoldi, Hutchinson trace and diagonal
estimation, stochastic Lanczos quadrature for log-determinants, partial SVD.

Full listings are in the [API reference](https://linox.lenardrommel.com/reference/api/).

## Documentation

| | |
|---|---|
| [Quickstart](https://linox.lenardrommel.com/quickstart/) | Five minutes end to end |
| [Linear operators](https://linox.lenardrommel.com/concepts/linear-operators/) | The model |
| [Structure and laziness](https://linox.lenardrommel.com/concepts/structure-and-laziness/) | Why `A + s*I` is not a sum |
| [Choosing a method](https://linox.lenardrommel.com/guides/choosing-a-method/) | Exact vs approximate |
| [Avoiding densification](https://linox.lenardrommel.com/guides/avoiding-densification/) | Why it got slow |
| [Kronecker GP](https://linox.lenardrommel.com/examples/kronecker-gp/) | A worked example |

## Status

Alpha, and the API is still moving. The [roadmap](https://linox.lenardrommel.com/development/roadmap/)
records what is settled, what is known to be rough, and where the library is behind
[lineax](https://github.com/patrick-kidger/lineax).

Every Python example in this README and on the documentation site is executed by the
test suite, so they do not silently rot.

## Related work

[`matfree`](https://github.com/pnkraemer/matfree) by Nicholas Krämer provides
matrix-free linear algebra in JAX — randomised and deterministic trace estimation,
matrix functions and factorisations. The Lanczos, Arnoldi, Hutchinson and SLQ
implementations here follow its approach closely, and it is the better choice if you
want those algorithms without an operator abstraction on top.

[`lineax`](https://github.com/patrick-kidger/lineax) by Patrick Kidger is the closest
neighbour: a linear operator and solver library for JAX. It is stronger on solver
rigour — more solvers, preconditioning, PyTree-structured operators, and a custom
primitive for stable least-squares gradients. `linox` is stronger on structure:
Kronecker, Toeplitz, kernel, block and low-rank operators with algorithms that
exploit them.

[`probnum.linops`](https://probnum.readthedocs.io/en/latest/api/linops.html) is the
NumPy-based ancestor of this design.

## Contributing

Issues and pull requests are welcome at
[github.com/lenardrommel/linox](https://github.com/lenardrommel/linox).

```bash
pip install -e ".[test]"
pre-commit install
pytest
```

CI runs the test suite on Python 3.10–3.12, the pre-commit hooks, and a strict
documentation build. All three must pass.

## Citation

If `linox` is useful in your research, please cite the repository:

```bibtex
@software{linox,
  author  = {Weber, Tobias and Rommel, Lenard},
  title   = {linox: Structured, matrix-free linear algebra in JAX},
  url     = {https://github.com/lenardrommel/linox},
  year    = {2026},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
