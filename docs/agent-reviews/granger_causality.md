# Feature Review: granger_causality

**File**: `src/cobrabox/features/granger_causality.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Summary

The `granger_causality.py` file contains two well-structured features (`GrangerCausality` and `GrangerCausalityMatrix`) that implement Granger causality analysis using the log-ratio of prediction error variances. The code is mathematically sound, well-documented with comprehensive docstrings, and follows most structural patterns. However, it is missing the `_is_cobrabox_feature` flag required for auto-discovery, and lacks validation for the `coords` parameter in `GrangerCausalityMatrix`. These are blocking issues for registration and robustness.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

Both classes properly use `@dataclass` decorator and inherit `BaseFeature[SignalData]`. The `output_type: ClassVar[type[Data]] = Data` is correctly set since the output removes the time dimension. Class names match the filename convention (PascalCase). `__call__` signatures are correct: `def __call__(self, data: SignalData) -> xr.DataArray`. No custom `apply()` implementation (correctly inherited).

**Missing**: The `_is_cobrabox_feature = True` flag required for feature auto-discovery in `feature.py`. Both classes need this attribute added (line 109 and line 178 respectively).

## Docstring

Excellent documentation. Both classes have:
- Clear one-line summaries
- Extended descriptions explaining the algorithm and interpretation
- Complete `Args:` sections documenting all dataclass fields
- Detailed `Returns:` sections describing output dimensions
- Working `Example:` sections with `.apply()` usage

The helper function `_granger_log_ratio` also has comprehensive documentation with formulas and interpretation guidelines.

## Typing

All dataclass fields are properly typed:
- `coord_x: str | int | None`
- `coord_y: str | int | None`
- `lag: int | None`
- `maxlag: int = 1`
- `coords: list[str] | list[int] | None`

`__call__` return types are correctly annotated as `xr.DataArray`. No bare `Any` types. The helper function has full type annotations.

## Safety & Style

- No `print()` statements
- Input validation in `__post_init__` for `maxlag` and `lag` parameters (lines 118-121, 186-189)
- No mutation of input `data` (creates new arrays via `_granger_log_ratio` and `xr.DataArray` constructors)
- Clean separation of concerns with helper function

**Missing validation**: `GrangerCausalityMatrix.__post_init__` does not validate the `coords` parameter. An empty list `coords=[]` would produce a nonsensical result matrix with shape (0, 0). Consider adding:
```python
if self.coords is not None and len(self.coords) == 0:
    raise ValueError("coords cannot be an empty list")
```

## Action List

1. **[Severity: HIGH]** Add `_is_cobrabox_feature = True` flag to both `GrangerCausality` class (after line 109) and `GrangerCausalityMatrix` class (after line 178) to enable auto-discovery.

2. **[Severity: MEDIUM]** Add validation in `GrangerCausalityMatrix.__post_init__` to reject empty `coords` list. Current code at lines 184-189 should include a check that if `coords` is provided (not None), it must have at least one element.
