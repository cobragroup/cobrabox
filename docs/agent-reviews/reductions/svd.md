# Feature Review: SVD

**File**: `src/cobrabox/features/reductions/svd.py`
**Date**: 2026-03-18
**Verdict**: NEEDS WORK

## Summary

SVD is a well-structured feature with comprehensive docstring and correct base class usage. Ruff checks pass cleanly. However, it misses two docstring sections (`Raises:` and `References:`), and could use type narrowing for the `output` parameter with `Literal`. At 94% coverage, the tests are strong but miss one parameter option (`output="U"`).

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Correct `@dataclass` pattern with `BaseFeature[Data]` inheritance, matching filename, and proper `__call__` signature. No loose module-level helpers. The class name uses `SVD`, which is acceptable as a domain-standard acronym (singular value decomposition is universally recognized).

Minor note: The `__call__` return type is annotated as `xr.DataArray` (not `xr.DataArray | Data`), which technically omits `Data` from the union. This is not flagged by ruff, but the base class `BaseFeature.__call__` accepts returning either type. Given the complexity of this feature returning different shapes based on parameters, explicitly allowing `Data` in the return annotation would align with the base class contract: `xr.DataArray | Data`.

## Docstring

Comprehensive docstring with excellent extended description, detailed `Args:` covering all 7 fields, and clear `Returns:` explaining both output modes and the structured `attrs["svd"]` storage. Good variety in `Example:` showing different use cases (fMRI, EEG, time-frequency, trial-wise).

Missing sections:

- **`Raises:`** — should list `ValueError` conditions that are explicitly raised in `__call__` (lines 81-86)
- **`References:`** — SVD has a clear mathematical formulation and literature basis; adding a reference (e.g. Golub & Reinsch 1970, or any standard numerical analysis text) would strengthen the docstring

## Typing

All fields annotated. The `output: str = "V"` field accepts a fixed set of strings but uses a plain `str` type with runtime validation (line 85: `if self.output not in {"V", "U"}:`). Consider using `Literal` for stronger static type checking:

```python
from typing import Literal, get_args

OutputMode = Literal["V", "U"]

@dataclass
class SVD(BaseFeature[Data]):
    ...
    output: OutputMode = "V"

    def __call__(self, data: Data) -> xr.DataArray:
        ...
        if self.output not in get_args(OutputMode):
            raise ValueError(f"output must be in {get_args(OutputMode)}, got {self.output!r}")
```

This keeps the validation single-source while enabling IDE autocomplete.

## Safety & Style

Input validation is thorough (lines 81-86) checking dim existence, `n_components > 0`, and valid `output` value. No mutation of input `data` — works on `data.data` and returns new arrays. No `print()` statements. Line length ≤ 100 per ruff enforcement.

## Action List

1. [MEDIUM] Add `Raises:` section to docstring listing the three `ValueError` conditions (lines 81-86)
2. [MEDIUM] Add `References:` section with a standard SVD citation (e.g., Golub & Van Loan 2013 or Strang 2016)
3. [LOW] Consider using `Literal["V", "U"]` type for the `output` field instead of plain `str` for stronger typing
4. [LOW] Optionally update `__call__` return type to `xr.DataArray | Data` to match base class contract
