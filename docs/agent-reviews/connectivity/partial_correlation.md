# Feature Review: partial_correlation

**File**: `src/cobrabox/features/connectivity/partial_correlation.py`
**Date**: 2025-03-24
**Verdict**: NEEDS WORK

## Summary

The `PartialCorrelation` and `PartialCorrelationMatrix` features implement partial correlation correctly using the precision matrix method. Both classes properly inherit from `BaseFeature[SignalData]` and set `output_type = Data` since they remove the time dimension. Ruff passes cleanly. The previous issue with fake singleton dimensions has been fixed — `PartialCorrelation` now correctly returns a 0-dimensional DataArray. However, there are structural issues: a module-level helper function violates the "no loose helpers" rule, `__post_init__` validation is missing for constructor parameters, and the docstrings lack a `References` section for this established statistical method.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Both classes correctly use the dataclass pattern:

- ✅ `from __future__ import annotations` present (line 1)
- ✅ `@dataclass` decorator on both classes (lines 53, 125)
- ✅ `BaseFeature[SignalData]` inheritance appropriate for time-series analysis
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly set (lines 78, 155) — these features return correlation coefficients without time dimension
- ✅ `__call__` signatures correct: `(self, data: SignalData) -> xr.DataArray`
- ✅ Class names match filename (`partial_correlation.py` → `PartialCorrelation`, `PartialCorrelationMatrix`)
- ❌ **Loose helper function** (lines 13–50): `_compute_partial_correlation` is a module-level private function only used by these classes. Per criteria, helpers should be `@staticmethod` methods inside the class they serve.

## Docstring

Both classes have comprehensive Google-style docstrings with:

- ✅ One-line summary
- ✅ Extended description explaining behavior
- ✅ `Args:` section with all fields documented
- ✅ `Returns:` section describing output shape
- ✅ `Raises:` section listing exceptions
- ✅ `Example:` section showing `.apply()` usage
- ❌ **Missing `References:` section**: Partial correlation is an established statistical method (originally developed by Yule, 1907). The docstring should cite the mathematical foundation.

Suggested reference:

```python
References:
    Yule, G. U. (1907). On the Theory of Correlation for any Number of
    Variables, Treated by a New System of Notation. *Proceedings of the
    Royal Society A*, 79(529), 182-193.
```

## Typing

- ✅ All fields typed: `coord_x: str | int`, `control_vars: list[str] | list[int]`, etc.
- ✅ `__call__` return type: `xr.DataArray`
- ✅ `ClassVar` used correctly for `output_type`
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation present in `__call__` for dimension existence and coordinate presence
- ✅ `Data` immutability respected — works on `data.data` and returns new arrays
- ✅ **Previous issue FIXED**: `PartialCorrelation` now correctly returns a 0-dimensional DataArray (line 122: `return xr.DataArray(result)`)
- ❌ **Missing `__post_init__` validation**: Parameters like `control_vars` (must have at least one element) are only validated in `__call__`. Per criteria, dataclass fields with constraints should be validated in `__post_init__`:

```python
def __post_init__(self) -> None:
    if not self.control_vars:
        raise ValueError("control_vars must have at least one coordinate")
    if len(set(self.control_vars)) != len(self.control_vars):
        raise ValueError("control_vars contains duplicates")
```

- ⚠️ **Potential issue**: `PartialCorrelationMatrix` allows `coords` to include elements from `control_vars` — should document or validate this edge case.

## Action List

1. **[Severity: HIGH]** Move `_compute_partial_correlation` function (lines 13–50) inside `PartialCorrelation` class as a `@staticmethod` method, and update the call sites.

2. **[Severity: HIGH]** Add `__post_init__` validation to both classes to validate `control_vars` is non-empty at construction time (currently only validated in `__call__`).

3. **[Severity: MEDIUM]** Add `References:` section to both class docstrings citing Yule (1907) or a modern partial correlation reference.

4. **[Severity: LOW]** Consider adding validation to `PartialCorrelationMatrix` to warn when `coords` and `control_vars` overlap (currently silently computes but result may be unexpected).
