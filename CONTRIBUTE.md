# Contributing to linox

Thanks for your interest in contributing! This document explains how to set up the project locally, run tests and linters, build docs, and open a PR.

---

## Quick start

### 1) Prerequisites
- Python **>= 3.10** (3.11 recommended)
- `uv` installed
- (Optional) VS Code + Python + Ruff extensions

### 2) Clone + create environment
```bash
git clone <YOUR_REPO_URL>
cd linox

uv lock
uv sync
````

> `uv sync` installs the project and the default dependency groups (configured via `[tool.uv].default-groups`).

---

## Project layout

* `linox/` — main library package
* `helper/` — auxiliary package used by linox
* `tests/` — pytest test suite
* `docs/` — MkDocs documentation source (Markdown pages)

⚠️ If you have a top-level folder like `site/` in the repo root, ensure it is **excluded from packaging** (setuptools discovery), or keep an explicit package list (e.g. `packages = ["linox", "helper"]`) so it doesn’t get detected as a Python package.

---

## Development workflow

### Install dev tools (if not in your environment)

```bash
uv sync --group dev
```

---

## Code style & formatting

We use **Ruff** for linting + formatting.

### Format code

```bash
uv run ruff format .
```

### Lint (and auto-fix what’s safe)

```bash
uv run ruff check .
uv run ruff check --fix .
```

### Notes on linting rules

* We use **NumPy-style docstrings**.
* Type hints are encouraged and often enforced.
* Some “internal” modules may allow private member access (e.g. `_todense`) via per-file ignores.
* Prefer targeted `# noqa: <RULE>` only when there is a clear reason (e.g. intentional lazy import / circular dependency avoidance).

**Tooling config reminders**

* In `pyproject.toml` (Ruff):

  * `pydocstyle` should be set to NumPy convention:

    ```toml
    [tool.ruff.lint.pydocstyle]
    convention = "numpy"
    ```
* In `mkdocs.yml` (mkdocstrings/griffe):

  * Use NumPy docstring style:

    ```yml
    plugins:
      - mkdocstrings:
          handlers:
            python:
              options:
                docstring_style: numpy
    ```

---

## Imports

### Default behavior

Ruff can organize imports as part of lint/fix and on-save in many editors.

### “Float imports to top” (occasional manual cleanups)

If you want the `isort` feature that moves imports from the middle of a file to the top:

```bash
uv run isort --float-to-top path/to/file.py
# or for the whole repo:
uv run isort --float-to-top .
```

---

## Testing

We use **pytest**.

### Run tests

```bash
uv run pytest
```

### Run tests with coverage (HTML + XML configured in pyproject)

```bash
uv run pytest --cov
```

### Troubleshooting: “No module named pytest”

If your editor/test runner can’t find pytest, you likely installed only the base dependencies. Sync the dev/test group:

```bash
uv sync --group dev
# or
uv sync --group test
```

---

## Pre-commit hooks

If you use `pre-commit`, install hooks once:

```bash
uv run pre-commit install
```

Run hooks on all files:

```bash
uv run pre-commit run --all-files
```

---

## Documentation

Docs are built with **MkDocs Material** + **mkdocstrings** (API docs from docstrings).

### Serve docs locally

```bash
uv run mkdocs serve
```

### Build docs

```bash
uv run mkdocs build
```

### Where documentation lives

* Narrative/guide content: `docs/*.md`
* API reference: generated from **docstrings** via `mkdocstrings` (pages containing `::: linox...`)

### Common doc build warnings

* **Broken links**: MkDocs resolves links only to files under `docs/` (and/or included in `nav:`).
* **griffe warnings**: usually missing type hints or docstring formatting/indentation issues.

---

## Writing docs & docstrings

### Docstrings (NumPy style)

Use NumPy convention consistently:

```py
def f(x: int) -> int:
    """Short summary.

    Parameters
    ----------
    x
        Description of `x`.

    Returns
    -------
    int
        Description of the return value.
    """
    return x
```

Tips:

* Keep the first line a short summary.
* Use `Parameters`, `Returns`, `Raises`, `Notes`, `Examples` when useful.
* For JAX-heavy code, consider documenting **shape**, **dtype**, and **batching** expectations.

### Markdown docs

* Prefer small, runnable examples.
* Mention shapes/dtypes when relevant (JAX).
* If you add a new user-facing feature, add both:

  * a docstring for API reference, and
  * a short usage section in a relevant `docs/*.md` page.

---

## Making changes

### Branching

```bash
git checkout -b feature/my-change
```

### Before opening a PR

Run:

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pytest
uv run mkdocs build
```

---

## Pull Requests

In your PR description, include:

* What changed and why
* Any API changes / breaking behavior
* How you tested it (commands + environment)
* Links to issues/discussions if applicable

---

## License & code of conduct

* Contributions are assumed to be under the project’s license.
* Be kind and constructive in reviews and discussions.

```

If you want, paste your current `mkdocs.yml` and I’ll show the exact `mkdocstrings` block for NumPy docstrings (and help silence the griffe warnings cleanly).
::contentReference[oaicite:0]{index=0}
```
