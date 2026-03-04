# Feature Review: sliding_window

**File**: `src/cobrabox/features/sliding_window.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

`SlidingWindow` is a well-implemented `SplitterFeature` that lazily yields per-window `Data` slices over the `time` dimension. Dimension validation, edge-case checking (zero or negative window count), and lazy generation are all handled correctly. The primary deficiency is an incomplete docstring: the `Returns:` section is absent and the `Example:` block, while illustrating chord construction, does not show the generator output or demonstrate standalone usage. There is also a minor redundancy in the window-start calculation: `n_windows` is computed by floor-division and then immediately used to slice `window_starts` after `np.arange` already produces the same set — the two computations are slightly inconsistent in edge cases and the slice is superfluous. Additionally, field-level validation (`__post_init__`) for `window_size` and `step_size` being positive integers is absent.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- Line 1: `from __future__ import annotations` is correctly the first import. (PASS)
- Lines 12-13: Class is `@dataclass` and inherits `SplitterFeature`. (PASS)
- Class name `SlidingWindow` is PascalCase and matches filename `sliding_window.py`. (PASS)
- Line 33: `__call__` signature is `(self, data: Data) -> Iterator[Data]`, which exactly matches the `SplitterFeature` abstract interface. (PASS)
- `data` is not a dataclass field. (PASS)
- No `apply()` reimplementation. (PASS)
- Lines 30-31: `window_size` and `step_size` are proper dataclass fields with defaults. (PASS)
- Import order: future → stdlib (`collections.abc`, `dataclasses`) → third-party (`numpy`) → internal (`..base_feature`, `..data`). (PASS)
- No unused imports (both `field` and `Iterator` are used), no `function_wrapper`. (PASS)
- Line 50: yields `data._copy_with_new_data(new_data=window_data, operation_name="SlidingWindow")`, correctly using `_copy_with_new_data` to produce immutable `Data` slices with history tracking. (PASS)

## Docstring

- One-line summary present: "Yield one Data per sliding window over the time dimension." (PASS)
- Extended description present (line 16): notes laziness to avoid materialising all windows. (PASS)
- `Args:` section: present and covers both fields (`window_size`, `step_size`) with clear descriptions. `data` is correctly not documented. (PASS)
- `Returns:` section: entirely absent. The criteria require documenting what is yielded — specifically the shape (slice of input along `time` of length `window_size`), the preserved dimensions, and the added `SlidingWindow` history entry. (FAIL)
- `Example:` section: present (lines 23-27) but shows only `Chord` construction. It does not demonstrate standalone generator usage (e.g., `list(SlidingWindow(100, 50)(data))`) and references `Chord`, `LineLength`, and `MeanAggregate` without imports, making it non-runnable as a doctest.

## Typing

- Fields `window_size: int` and `step_size: int` are typed. (PASS)
- `__call__` return type `-> Iterator[Data]` is explicit. (PASS)
- No bare `Any`. (PASS)

## Safety & Style

- No `print()` statements. (PASS)
- `time` dimension check at line 36-37: raises `ValueError` if missing. (PASS)
- Zero/negative window count check at lines 41-42: raises `ValueError` with an informative message. (PASS)
- No `__post_init__` for field validation: `window_size` and `step_size` have no guard against zero or negative values. A `window_size=0` or `step_size=0` would only be caught indirectly (division by zero in `np.arange` or a downstream error), rather than raising a clear `ValueError` at construction time. (MEDIUM)
- Input `data` is not mutated; `xr_data.isel(...)` returns a new DataArray. (PASS)
- All lines are within 100 characters. (PASS)
- Lines 44-45: minor logic redundancy. `n_windows` is computed via floor-division on line 40, and then `window_starts` is sliced to `n_windows` on line 45. However, `np.arange(0, n_time - self.window_size + 1, self.step_size)` already produces exactly the correct starts without needing the slice — the `[:n_windows]` slice is superfluous and can mask the fact that the two formulas may disagree in integer edge cases. (LOW)

## Action List

1. [HIGH] Add a `Returns:` section to the docstring. It should describe that the generator yields `Data` objects whose `time` dimension has length `window_size`, all other dimensions are preserved, and each yielded object has `"SlidingWindow"` appended to its history.
2. [MEDIUM] Add `__post_init__` validation to enforce that `window_size >= 1` and `step_size >= 1`, raising `ValueError` with descriptive messages at construction time rather than failing obscurely at call time.
3. [MEDIUM] Expand the `Example:` to include a runnable standalone usage showing a `SlidingWindow` generator being consumed (e.g., iterating or listing windows), in addition to the chord-syntax illustration. Add necessary imports to make it a valid doctest.
4. [LOW] Remove the redundant `[:n_windows]` slice on line 45. `np.arange(0, n_time - self.window_size + 1, self.step_size)` is sufficient and consistent with the guard already applied on lines 41-42. Removing the slice eliminates the latent discrepancy between the floor-division formula and the `np.arange` formula.
