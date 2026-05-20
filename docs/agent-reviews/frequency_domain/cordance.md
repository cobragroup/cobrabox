# Feature Review: cordance

**File**: `src/cobrabox/features/frequency_domain/cordance.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

The `Cordance` feature is a well-implemented, high-quality feature that follows all cobrabox conventions. It correctly inherits from `BaseFeature[SignalData]`, includes comprehensive docstring with algorithm description, proper type annotations, and robust input validation. The feature implements the Leuchter 1994 cordance algorithm combining absolute and relative bandpower measures. Only minor issue: missing `Raises:` section in docstring.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

- ✅ Line 1: `from __future__ import annotations` present as first import
- ✅ Line 14: `@dataclass` decorator applied
- ✅ Line 15: Correctly inherits `BaseFeature[SignalData]` (requires time dimension)
- ✅ Line 92: `output_type: ClassVar[type[Data]] = Data` correctly set (removes time dimension)
- ✅ Class name `Cordance` matches filename `cordance.py`
- ✅ No redundant `_is_cobrabox_feature` marker (inherited from base)
- ✅ Line 110: `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray:`
- ✅ Does NOT implement `apply()` (correctly inherited)
- ✅ No loose module-level helper functions
- ✅ Imports ordered correctly: __future__ → stdlib → third-party → internal

## Docstring

Comprehensive Google-style docstring with most required sections:

- ✅ One-line summary present (line 16)
- ✅ Extended algorithm description with step-by-step breakdown (lines 21-38)
- ✅ `Args:` section documents all 5 dataclass fields (lines 40-68)
- ✅ `Returns:` section describes output shape and dims (lines 75-78)
- ✅ `References:` section with full citations including US Patent (lines 80-89)
- ✅ `Example:` section with 3 usage examples (lines 70-73)
- ❌ **Missing `Raises:` section** — feature raises `ValueError` in `__post_init__` (lines 101-108) and `__call__` (lines 111-121) but these are not documented

## Typing

- ✅ All 5 dataclass fields fully typed (lines 94-98)
- ✅ `__call__` return type: `-> xr.DataArray` (line 110)
- ✅ No bare `Any` types
- ✅ `Literal["cordance", "concordance", "discordance"]` used for constrained string field (line 97)

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation in `__post_init__` (lines 101-108):
  - Validates `nperseg >= 2`
  - Validates `threshold in (0, 1)`
  - Validates `output` is one of allowed values
- ✅ Input validation in `__call__` (lines 111-121):
  - Validates `space` dimension exists
  - Validates at least 2 spatial channels for normalization
- ✅ No mutation of input `data` — all operations return new arrays
- ✅ Handles edge case of zero total bandpower with `nan_on_zero` flag

## Action List

1. [Severity: LOW] Add `Raises:` section to docstring documenting `ValueError` conditions raised in `__post_init__` and `__call__`.
