# Architecture

```
linox/
├── operators/     operator classes and their dispatched implementations
├── linalg/        algorithms: solving, decompositions, approximations
│   └── approx/    matrix-free: cg, lanczos, arnoldi, lsmr, hutchinson, slq
├── utils/         conversion, validation, graph inspection
├── structure/     operator IR, canonicalisation, fingerprinting
├── cache/         session-scoped analysis cache
├── debug/         tracing
├── interop/       adapters to other libraries
├── api.py         the public functional surface
└── config.py      settings and method resolution
```

## Three layers

**Operators** know their shape and how to apply themselves. They store structure,
not results.

**Dispatch** connects an operation to an implementation, keyed on operator type via
plum. `solve` is a generic with a method per operator; adding an operator means
adding methods, not editing a switch. This is the same argument lineax makes for
`singledispatch`: end users can add operators *and* operations without touching the
core.

**The API layer** (`api.py`) is a thin, documented surface over the generics. It
resolves `method=`, validates it, and adds the outcome reporting that the raw
dispatches do not carry.

## Why arithmetic rewrites

`smart_add` inspects a sum for patterns worth specialising — `s·I + A`, `D + USUᵀ` —
and returns a structured operator instead of a generic sum. The payoff is that
downstream operations then find a fast dispatch. The cost is that the type you get
is not always the type you wrote, which is why `linop_graph` exists.

## Naming

Two spellings coexist:

- **`linox.solve`, `linox.sqrt`** — the public API, with `method=` and outcome
  reporting.
- **`lsolve`, `lsqrt`, `linverse`** — the underlying plum generics.

The `l`-prefixed names are deprecated for public use but remain the extension point:
to teach linox about a new operator, register methods on the generics.

## Invariants

Things the test suite enforces, and that new code must not break:

1. **Matrix-free stays matrix-free.** Operations documented as such must not
   densify. Asserted per operator in the conformance suite.
2. **The adjoint identity.** `⟨Ax, y⟩ = ⟨x, Aᵀy⟩` for every operator.
3. **`sqrt` returns a left square root.** `S Sᵀ = A`.
4. **Rewrites preserve meaning.** A rewritten expression equals the plain arithmetic.
5. **Failure is reported.** A solve that did not converge says so.
6. **No stacked dispatch decorators.** They leak one generic's signature into
   another; a test scans for the pattern.

## Testing

Three kinds, in increasing order of what they catch:

- **Unit tests** per operator and algorithm.
- **The conformance suite** (`tests/test_operator_conformance.py`), which asserts
  mathematical properties against every operator in a registry. Adding an operator
  opts it into all of them.
- **Documentation examples** (`tests/test_docs_examples.py`), which execute every
  Python block on this site.

The conformance suite exists because three silently-wrong results reached a release
despite extensive unit tests: each test asserted the same thing the implementation
did. A property test cannot agree with a buggy implementation, because it never
mentions one.
