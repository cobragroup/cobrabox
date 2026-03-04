# Feature Review: autocorr

**File**: `src/cobrabox/features/autocorr.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

The feature is structurally sound — correct `@dataclass` / `BaseFeature[Data]` pattern,
clean ruff output, all fields typed, no print statements, no input mutation. Two issues
prevent a PASS: (1) the docstring is missing a `Returns:` section describing the output
shape and dimensions, and (2) `fs` has no validation guard against non-positive values,
which would silently produce a meaningless or zero lag.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural criteria met. `from __future__ import annotations` is present on line 1.
`@dataclass` and `BaseFeature[Data]` are correct — the feature is dimension-agnostic
(accepts any `dim`), so `Data` is the right type parameter. `__call__` signature
(`data: Data) -> xr.DataArray`) matches the base class contract. No `apply()` override.
`__post_init__` correctly validates the `lag_steps`/`lag_ms` mutual exclusion at
construction time. Imports are clean and in standard order.

## Docstring

One-line summary is clear. Extended description with MATLAB context is useful. `Args:`
section covers all four fields with adequate descriptions. `Example:` section is present
and uses the correct `.apply()` call style. **Missing**: a `Returns:` section. The
feature's output shape (the requested `dim` is reduced to a scalar per remaining
dimension element) is non-obvious and should be documented explicitly.

## Typing

All dataclass fields have type annotations (`str`, `float`, `int | None`, `float | None`).
`__call__` return type is `xr.DataArray`. `_acf_numpy` static method is annotated
(`np.ndarray, int) -> float`. No bare `Any`. No issues.

## Safety & Style

No print statements. No input mutation. `dim` is validated in `__call__` with a clear
`ValueError` (line 68). Lag range is validated against the dimension length (line 79–80).
The `lag_steps`/`lag_ms` conflict is caught at `__post_init__` time (line 39–40).
**Gap**: `fs` is not validated — a zero or negative value will silently compute `lag = 0`
(from `round(0 * 5 / 1000.0)`) which then hits the range check, but the error message
(`"lag must be between 1 and N"`) gives no hint that `fs` is the culprit. A direct guard
in `__post_init__` makes the failure message actionable.

## Action List

1. [MEDIUM] Add a `Returns:` section to the class docstring describing the output: the
   requested dimension is consumed as a core dim via `apply_ufunc`, so the result is a
   DataArray with all input dimensions except `dim`. Example text: "xarray DataArray with
   the `dim` dimension removed. Shape is the input shape minus the size of `dim`."

2. [LOW] Add `fs` validation in `__post_init__` (after line 40): raise `ValueError` if
   `self.fs <= 0` so a misconfigured sampling rate produces an immediately actionable
   message rather than a downstream lag-range error.
