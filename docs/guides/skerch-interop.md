# skerch interop

[skerch](https://github.com/andres-fr/skerch) implements sketched decompositions —
randomised low-rank and Hermitian approximations built from matrix-vector products.
Since that is exactly what a linox operator provides, the two compose.

```bash title="not-executed"
pip install "linox[interop]"
```

The extra pulls in skerch and torch. It is deliberately separate from `dev`: torch
and h5py are a lot of install for a JAX library whose own test suite never touches
them.

## The adapter

```python title="not-executed"
from linox.interop.skerch import LinoxLinOp

op = linox.Kronecker(linox.Matrix(a), linox.Matrix(b))
sketchable = LinoxLinOp(op)
```

skerch expects an object with `.shape` and the two matmul protocols — `__matmul__`
for `A @ x` and `__rmatmul__` for `x @ A`. `LinoxLinOp` supplies them, converting
between torch tensors and JAX arrays at the boundary.

## Why it stays matrix-free

The adjoint route matters here. `__rmatmul__` goes through `operator.T`, which for a
linox operator either uses a structured transpose or derives the adjoint from the
forward matvec — either way without materialising the matrix. A sketching algorithm
that densified its operand would defeat its own purpose.

## Caveats

**Real operators only.** The adjoint identity used in the adapter assumes a plain
transpose. Complex operators need the conjugate transpose, and the adapter guards
against being used with one rather than returning a wrong answer.

**Two array libraries.** Every crossing converts between torch and JAX. For a
sketch, the matvecs dominate and the conversions are noise; for a tight loop, they
are not.

**Not differentiable end to end.** Gradients do not flow through the torch boundary.
Use it for the sketch, then bring the factors back into JAX.

## When to reach for it

Sketching earns its place when you want a low-rank approximation of an operator that
is too large to decompose exactly and too unstructured for the exact shortcuts — the
gap between `linox.svd(op, k=...)`, which is Krylov-based and deterministic, and
doing nothing. Randomised methods trade a little accuracy for a fixed, predictable
number of matvecs.

For a Kronecker product or a diagonal-plus-low-rank operator, the exact structured
routes are better. Reach for sketching when there is no structure to exploit.
