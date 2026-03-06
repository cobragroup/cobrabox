# Feature Review: fourier_transform_surrogates

**File**: `src/cobrabox/features/fourier_transform_surrogates.py`
**Date**: 2025-03-06
**Verdict**: NEEDS WORK

## Summary

The `FourierTransformSurrogates` feature is a `SplitterFeature` that generates surrogate time series by randomizing Fourier phases while preserving the power spectrum. The implementation has good algorithmic logic but has several issues: missing `__future__` import, docstring format inconsistencies, a critical bug where `transpose()` result is not assigned, and metadata is not preserved in the output. These must be fixed before the feature is production-ready.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- ✅ Correct `@dataclass` decorator
- ✅ Inherits `SplitterFeature[SignalData]` (appropriate for time-series splitting)
- ✅ Class name matches filename (`FourierTransformSurrogates`)
- ❌ Missing `from __future__ import annotations` (line 1) — should be before other imports
- ❌ Line 43: `data.data.transpose(..., "time")` has no effect — result is not assigned

## Docstring

- ❌ Uses "Parameters:" instead of "Args:" (Google style requires "Args:")
- ❌ Uses "Output:" instead of "Yields:" (SplitterFeature yields, not returns)
- ❌ Missing "Example:" section showing typical usage
- ✅ Has one-line summary
- ✅ Describes parameters with types

## Typing

- ✅ All fields typed (`n_surrogates: int`, `multivariate: bool`, etc.)
- ✅ `__call__` has correct return type `Iterator[SignalData]`
- ⚠️ Could add explicit type hint for `__post_init__` return (has `-> None`)

## Safety & Style

- ❌ Line 38-39: Uses `assert` for validation instead of `ValueError`. Assertions can be disabled with `-O` flag. Should use:

  ```python
  if not isinstance(self.n_surrogates, int):
      raise ValueError("The number of surrogates must be an integer.")
  if self.n_surrogates < 0:
      raise ValueError("The number of surrogates cannot be negative.")
  ```

- ❌ Line 43: `data.data.transpose(..., "time")` — this is a no-op because the result is not assigned. Should be:

  ```python
  transposed = data.data.transpose(..., "time")
  tmp = np.reshape(transposed.data, [-1, transposed.shape[-1]])
  ```

- ❌ Line 53: Metadata is not preserved. `SignalData.from_numpy()` only receives `dims` but loses:
  - `subjectID`
  - `groupID`
  - `condition`
  - `sampling_rate`
  - `extra` attributes

  Should use `_copy_with_new_data` or pass all metadata explicitly.

- ✅ No print statements
- ✅ No mutation of input data (creates new arrays)

## Action List

1. **[Severity: HIGH]** Add `from __future__ import annotations` as the first import line.

2. **[Severity: HIGH]** Fix line 43: Assign the transpose result or remove if unnecessary. The current line has no effect:

   ```python
   # Current (broken):
   data.data.transpose(..., "time")
   tmp = np.reshape(data.data.data, [-1, data.data.shape[-1]])

   # Fixed:
   transposed = data.data.transpose(..., "time")
   tmp = np.reshape(transposed.data, [-1, transposed.shape[-1]])
   ```

3. **[Severity: HIGH]** Preserve metadata in `_surrogate` method (line 53). The output loses subjectID, groupID, condition, sampling_rate, and extra. Use `_copy_with_new_data` pattern or pass all metadata to `from_numpy`:

   ```python
   return data._copy_with_new_data(
       xr.DataArray(xs, dims=data.data.dims, coords=data.data.coords, attrs=data.data.attrs)
   )
   ```

4. **[Severity: MEDIUM]** Replace `assert` statements with `ValueError` in `__post_init__` (lines 38-39).

5. **[Severity: MEDIUM]** Fix docstring format:
   - Change "Parameters:" to "Args:"
   - Change "Output:" to "Yields:"
   - Add "Example:" section showing usage with `.apply()` or in a Chord

6. **[Severity: LOW]** Consider adding type annotation for `xs` variable or using explicit intermediate variables for clarity in the FFT operations.
