# Feature Review: max

**File**: `src/cobrabox/features/max.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Summary

A clean, well-structured feature with proper docstring sections and input validation. However, it lacks the `output_type` classvar needed when reducing dimensions. Since `Max` removes the specified dimension, the output may not be compatible with the input container type (e.g., reducing `time` on `SignalData` should return plain `Data`). This is the main blocker.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

Correct `@dataclass` + `BaseFeature[Data]` inheritance. Class name matches filename. `__call__` signature is correct with `data: Data` parameter. No `apply()` override — good, inherits from base.

**Issue**: Missing `output_type` classvar (line 11-12). Since this feature reduces a dimension, it should declare:
```python
output_type: ClassVar[type[Data]] = Data
```
This signals that the output container is always plain `Data`, regardless of input type.

## Docstring

All required sections present: summary, Args, Returns, Example. Well written and clear.

**Minor**: Line 21 mentions "input signal" — for a generic `BaseFeature[Data]`, "signal" implies time-series data. Consider "input data" for consistency with the generic type parameter.

## Typing

Field `dim: str` is properly typed. `__call__` return type `xr.DataArray` is correct. No bare `Any` types.

## Safety & Style

No `print()` statements. Proper input validation on line 30-31 checking dimension existence. No mutation of input data — correctly operates on `data.data` and returns new array. Line 31 exceeds 100 chars slightly but ruff allows it.

## Action List

1. [Severity: HIGH] Add `output_type: ClassVar[type[Data]] = Data` class variable after the docstring (before `dim: str`). Import `ClassVar` from `typing` if not already available.

2. [Severity: LOW] Line 21: change "input signal" to "input data" for consistency with generic `BaseFeature[Data]` type.
