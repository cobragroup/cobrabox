# Feature Review: mutual_information

**File**: `src/cobrabox/features/mutual_information.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

The `MutualInformation` feature is generally well-structured with good docstring coverage and proper type annotations. However, it has a critical mutability issue where `_n_bins` is stored as a dataclass field but modified during `__call__`, making the feature non-thread-safe and stateful. The docstring is also missing `Raises:` and `References:` sections.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

The class structure is correct:

- Uses `@dataclass` decorator with `BaseFeature[SignalData]` inheritance (line 13-14)
- `output_type: ClassVar[type[Data] | None] = Data` is correctly set (line 65) since the output removes the time dimension
- Class name `MutualInformation` is descriptive and matches filename
- `__call__` signature correctly takes `data: SignalData` and returns `xr.DataArray` (line 78)
- No redundant `apply()` method (correctly inherited)

**Issue on line 122**: The `transpose()` call has no effect. `xarray.DataArray.transpose()` returns a new array, so this line should be `x = x.transpose(..., self.other_dim, self.dim)` or removed if not needed.

## Docstring

Good coverage with one-line summary, extended description, Args, Returns, and Example sections. However, missing required sections:

- **Missing `Raises:` section**: The feature raises `ValueError` in `__post_init__` (lines 69-76) and `__call__` (lines 80, 88-90, 92). Each should be documented.
- **Missing `References:` section**: Mutual information is a well-established metric. A citation (e.g., Shannon 1948 or Cover & Thomas) should be included.

## Typing

All fields have type annotations:

- `dim: str` (line 58)
- `other_dim: str | None` (line 59)
- `bins: int | None` (line 60)
- `equiprobable_bins: bool` (line 61)
- `log_base: float` (line 62)

**Issue on line 63**: `_n_bins: int = 0` should not be a dataclass field at all (see Safety section).

Return type of `__call__` is correctly `xr.DataArray` (line 78).

## Safety & Style

### Critical: Mutable state in dataclass

**Line 63**: `_n_bins: int = 0` is defined as a dataclass field, but it is modified in `__call__` (lines 95, 97). This makes the feature:

1. Non-thread-safe (concurrent calls will corrupt state)
2. Stateful (subsequent calls with different data sizes will have wrong bin count)

**Fix**: Remove `_n_bins` from dataclass fields and make it a local variable in `__call__`:

```python
# Remove this line entirely:
# _n_bins: int = 0

# In __call__, use a local variable:
n_bins = int(data.data[self.dim].size ** (1 / 3)) if self.bins is None else self.bins
```

Update all references from `self._n_bins` to the local `n_bins` variable.

### Input validation

Validation is thorough:

- `__post_init__` validates `bins`, `dim`, and `other_dim` (lines 67-76)
- `__call__` validates dimension existence (lines 79-92)

No issues here.

### Line 122: Unused transpose

```python
x.transpose(..., self.other_dim, self.dim)  # Has no effect
```

Should be assigned: `x = x.transpose(..., self.other_dim, self.dim)`

## Action List

1. **[Severity: HIGH]** Remove `_n_bins` from dataclass fields (line 63). Make it a local variable in `__call__` to ensure thread-safety and immutability.

2. **[Severity: MEDIUM]** Fix line 122: Assign the result of `transpose()` to a variable, or remove if the operation is unnecessary.

3. **[Severity: MEDIUM]** Add `Raises:` section to docstring documenting all `ValueError` exceptions raised in `__post_init__` and `__call__`.

4. **[Severity: LOW]** Add `References:` section with citation for mutual information (e.g., Shannon 1948 or Cover & Thomas, Elements of Information Theory).
