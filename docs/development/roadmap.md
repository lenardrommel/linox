# Roadmap

## Settled

- Operator model, lazy arithmetic and the rewrite rules
- Dispatch on operator type via plum
- `Solution` / `RESULTS` outcome reporting on `solve`
- Preconditioned conjugate gradients
- Left-square-root contract for `sqrt`
- float64 preservation
- Matrix-free transpose for operators without a structured one

## In progress

**API surface tidy-up.** Several inconsistencies remain: `LowRank.shape` reports a
square shape for rectangular factors, `diagonal(Toeplitz)` returns an operator where
every other dispatch returns an array, and `CongruenceTransform` is defined twice
with the specialised `diagonal` fast path bound to the shadowed class.

**Import-time side effects.** linox calls `jax.config.update("jax_enable_x64", True)`
at import, changing the dtype policy of the importing program. This should go; the
hesitation is that removing it changes default dtypes for anyone relying on it.

**`config.warn` is a no-op.** Its body is `if _DEBUG: pass`, so densification
warnings are silent unless a debug hook is installed.

## Planned

**Tag propagation.** Which structural properties survive transpose and inversion is
currently decided ad hoc per operator. lineax centralises this in
`transpose_tags`/`invert_tags` rules — for example, tridiagonality survives
transposition but not inversion. The `Sym`/`PSD`/`SPD` wrappers are groping toward
the same thing.

**Inspectable `method="auto"`.** You cannot currently ask which path `auto` will
take without reading `config.resolve_method`. lineax exposes `select_solver`.

**More iterative solvers.** BiCGStab and GMRES for non-symmetric operators; linox
has CG and LSMR only.

## Considered, not planned

**PyTree-structured operators.** lineax lets in/out spaces be arbitrary pytrees.
Valuable, and a large change to every shape computation in the library.

**A custom AD primitive for solves.** lineax defines `linear_solve_p` with
hand-written JVP and transpose rules, which is what makes its least-squares gradients
stable. linox currently differentiates the underlying factorisation.

## Where linox differs from lineax

Worth being explicit, since the two overlap. linox is stronger on **structure**:
Kronecker, Toeplitz, kernel, block and low-rank operators, and matrix-free spectral
methods, none of which lineax has. lineax is stronger on **solver rigour**: a wider
solver set, preconditioning, PyTree structure, and a more careful AD story.

The gap has narrowed — outcome reporting and preconditioned CG were both taken from
lineax's design — but the remaining items above are real.
