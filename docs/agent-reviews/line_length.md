# Feature Review: line_length

**File**: `src/cobrabox/features/line_length.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

`LineLength` is structurally sound: it has the correct imports, decorators, base class, return type, input validation for the `time` dimension, and no mutation of input data. The primary issues are in the docstring: `data` is documented in `Args:` when it should not be (only dataclass fields belong there, and `LineLength` has no fields), the `Returns:` section is missing the dimension names and value semantics of the output array, and the `Example:` code references a non-existent `wdata` variable that would confuse a new user. These are all medium-severity docstring deficiencies.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- **Line 1**: `from __future__ import annotations` is present and first. Correct.
- **Lines 3–8**: Import order is future → stdlib (`dataclasses`) → third-party (`xarray`) → internal. Correct.
- **Lines 11–12**: `@dataclass` and `BaseFeature` inheritance are present. Correct.
- **Class name**: `LineLength` is PascalCase and matches `line_length.py`. Correct.
- **Line 30**: `__call__` signature is `(self, data: Data) -> xr.DataArray`. Correct for a `BaseFeature`.
- `LineLength` has no dataclass fields, which is valid. `data` is not a field. Correct.
- No reimplementation of `.apply()`. Correct.

## Docstring

- **Lines 13–28**: Google-style docstring is present with a one-line summary and extended description. Good.
- **Lines 18–20 (`Args:`)**: `data` is listed in `Args:`. Per the review criteria, `Args:` must document only the dataclass fields, not `data`. Since `LineLength` has no fields, the `Args:` section should be omitted entirely.
- **Lines 22–23 (`Returns:`)**: The `Returns:` section says "xarray DataArray with 'time' dimension removed (or 'window_index' preserved)". This is partially correct but lacks the output shape description, the remaining dimension names (i.e. `space`, and optionally `window_index`), and the value semantics (sum of absolute first differences).
- **Lines 25–27 (`Example:`)**: The example applies `SlidingWindow` to produce `wdata` without showing that `data` must first be created. A reader cannot run this example without additional context. A minimal self-contained example using `.apply(data)` on a plain `Data` object would be clearer and more consistent with the convention used in other features.

## Typing

- No dataclass fields to type. Correct.
- `__call__` return type is `xr.DataArray`. Correct.
- No bare `Any`. Correct.

## Safety & Style

- No `print()` statements. Correct.
- **Line 33–34**: Validates that `time` is in `data.data.dims` and raises `ValueError`. Correct.
- No mutation of input `data`. `xr_data.diff(...)` and `.sum(...)` return new arrays. Correct.
- All lines are within 100 characters. Correct.
- No `__post_init__` needed (no fields with constraints). Correct.

## Action List

1. [MEDIUM] Remove the `Args:` section entirely, or replace it with a note that this feature takes no configuration parameters. `data` must not appear in `Args:`.
2. [MEDIUM] Expand `Returns:` to specify the output dimensions (e.g. `(space,)` or `(window_index, space)`), the dtype, and that values represent the sum of absolute first differences along the time axis.
3. [LOW] Rewrite the `Example:` to be a minimal, self-contained snippet that constructs a `Data` object and calls `LineLength().apply(data)` directly, without depending on `SlidingWindow` or an unexplained `wdata` variable.
