# Feature Review: dummy

**File**: `src/cobrabox/features/dummy.py`
**Date**: 2025-03-05
**Verdict**: SKIPPED (Intentional Negative Reference)

## Summary

This file is intentionally a negative reference and should not be "fixed." The docstring explicitly states: "This feature exists as a negative reference showing what a poorly written feature looks like. Do not use this as a template for new features."

## Note

Per the review criteria, `dummy.py` is excluded from quality audits as it serves as an intentional example of what NOT to do. The file contains known issues that are documented in its own docstring.

Known intentional issues:

- Missing type parameter on `BaseFeature` (should be `BaseFeature[SignalData]`)
- Wrong `__call__` return type (`Data` instead of `xr.DataArray | Data`)
- Missing `__post_init__` validation
- Drops metadata (subjectID, groupID, condition, extra)

## Action List

None — this file is intentionally non-compliant as a negative reference.
