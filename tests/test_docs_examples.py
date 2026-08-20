"""Execute every Python example in the documentation.

Documentation that lies is worse than none, and this project has form: the
README once listed 27 exports that did not exist, and several docstrings
promised a `ValueError` that was never raised. Prose cannot be tested, but
examples can, so every ```python block in `docs/` is executed here.

To exclude a block -- pseudo-code, or something requiring an optional
dependency -- tag the fence:

    ```python title="not-executed"
"""

from __future__ import annotations

import pathlib
import re
import textwrap

import pytest

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

#: ```python ... ``` blocks, capturing the info string so fences can opt out.
#: Leading whitespace is allowed so that examples nested inside admonitions are
#: tested too -- they were silently skipped while the regex anchored at column 0.
_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)```python(?P<info>[^\n]*)\n(?P<code>.*?)^(?P=indent)```",
    re.M | re.S,
)

#: Marks a block as illustrative rather than executable.
_SKIP = "not-executed"


def _blocks():
    """Yield (page, index, code) for every executable example in the docs."""
    for path in sorted(DOCS.rglob("*.md")):
        text = path.read_text()
        for i, match in enumerate(_BLOCK.finditer(text)):
            if _SKIP in match.group("info"):
                continue
            rel = path.relative_to(DOCS)
            code = textwrap.dedent(match.group("code"))
            yield pytest.param(str(rel), i, code, id=f"{rel}#{i}")


CASES = list(_blocks())


def test_documentation_contains_examples() -> None:
    """Guard the harness itself: a broken regex must not silently pass."""
    assert len(CASES) > 20, f"only found {len(CASES)} examples; is the regex right?"


@pytest.mark.parametrize(("page", "index", "code"), CASES)
def test_example_runs(page: str, index: int, code: str) -> None:
    """Each block executes standalone, so examples must be self-contained."""
    namespace: dict = {"__name__": "__doc_example__"}
    try:
        exec(compile(code, f"docs/{page}#{index}", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"docs/{page}, example {index} failed: {type(exc).__name__}: {exc}\n\n{code}"
        )
