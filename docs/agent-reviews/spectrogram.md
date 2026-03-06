# Feature Review: spectrogram

**File**: `src/cobrabox/features/spectrogram.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

A well-implemented feature that computes power spectrograms using scipy. The code is clean, properly typed, and follows all project conventions. Comprehensive docstring with all required sections (Args, Returns, Raises, Example). Validation logic is thorough, checking all parameter constraints before computation.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present.

Line 16-17: Correctly decorated with `@dataclass` and inherits `BaseFeature[SignalData]` (appropriate for time-series feature).

Line 64: `__call__` signature is `def __call__(self, data: SignalData) -> xr.DataArray` — matches base class contract.

Class name `Spectrogram` matches filename `spectrogram.py`.

No `apply()` override (correctly inherits from `BaseFeature`).

Imports are clean and in correct order (stdlib → third-party → internal).

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary** (line 18): Clear and descriptive.
- **Extended description** (lines 20-24): Explains algorithm and dimension handling.
- **Args** (lines 26-41): Documents all 4 dataclass fields with types and constraints.
- **Returns** (lines 43-46): Describes output dimensions and coordinate meanings.
- **Raises** (lines 48-50): Lists ValueError conditions.
- **Example** (lines 52-56): Working code snippet using `.apply()`.

## Typing

All fields properly typed:

- `nperseg: int | None = None`
- `noverlap: int | None = None`
- `window: str = "hann"`
- `scaling: str = "log"`

`__call__` return type is `xr.DataArray` — acceptable per criteria (can be `xr.DataArray | Data`).

No bare `Any` types.

## Safety & Style

No `print()` statements.

Input validation present (lines 67-80):

- Validates `scaling` against `_VALID_SCALINGS`
- Validates `nperseg >= 2`
- Validates `nperseg <= n_time`
- Validates `noverlap < nperseg`

No mutation of input `data` — operates on `data.data` and returns new `xr.DataArray`.

## Action List

None.
