# Feature Review: mean_aggregate

**File**: `src/cobrabox/features/mean_aggregate.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

`MeanAggregate` is a compact and functionally correct `AggregatorFeature` that averages a stream of per-window `Data` objects by stacking on a temporary `window` dimension and calling `.mean()`. It correctly builds and propagates history manually (as required by the `AggregatorFeature` contract), raises on an empty stream, and avoids mutating its inputs. However, the docstring is substantially incomplete — it is missing `Args:`, `Returns:`, and a meaningful `Example:` block — and the `__call__` parameter is named `stream` rather than `windows`, which is a minor but real inconsistency with the abstract base's own parameter name and with the CLAUDE.md convention. Additionally, the `Data` constructor call on line 35 omits `sampling_rate`, which means the sampling rate from the original data is silently dropped.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- Line 1: `from __future__ import annotations` is correctly the first import. (PASS)
- Lines 12-13: Class is `@dataclass` and inherits `AggregatorFeature`. (PASS)
- Class name `MeanAggregate` is PascalCase and matches the filename `mean_aggregate.py`. (PASS)
- Line 27: `__call__` signature is `(self, data: Data, stream: Iterator[Data]) -> Data`, consistent with `AggregatorFeature.__call__` in `base_feature.py`. The abstract method uses the parameter name `stream` (same as here), so that is fine. However, CLAUDE.md (Architecture section, `AggregatorFeature` bullet) specifies the signature as `(self, data: Data, windows: Iterator[Data]) -> Data`. The parameter name `stream` vs. `windows` is an inconsistency with the documented convention, though not with the actual abstract base.
- `data` is not a dataclass field. (PASS)
- No `apply()` reimplementation. (PASS)
- Line 35-42: `MeanAggregate` constructs `Data` directly and builds history manually — correct for an `AggregatorFeature`. (PASS)
- Import order: future → stdlib (`collections.abc`) → third-party (`xarray`) → internal (`..base_feature`, `..data`). (PASS)
- No unused imports, no `function_wrapper`. (PASS)
- Line 35: The `Data(...)` constructor call omits `sampling_rate`. `Data.__init__` accepts `sampling_rate` as a keyword argument. Without it, `Data` will attempt to infer it from the averaged array's time coordinates. After `xr.concat` + `.mean(dim="window")`, the time coordinates of the averaged result come from the per-window slices and may or may not be inferrable. This is a latent bug that silently discards the original `sampling_rate` from `data.sampling_rate`. `_copy_with_new_data` (used by `BaseFeature.apply`) preserves sampling rate correctly; `MeanAggregate` side-steps that helper and does not replicate the preservation.

## Docstring

- One-line summary present: "Aggregate a stream of per-window Data by averaging across windows." (PASS)
- Extended description present (lines 16-17). (PASS)
- `Args:` section: entirely absent. `MeanAggregate` has no dataclass fields, so there are no field args to document. However, per the criteria, `data` must NOT be documented but the class's own fields (none here) must be. Since there are no fields, the omission of `Args:` is technically acceptable — but it would be clearer to explicitly state "No configuration parameters." or omit the section intentionally with a note.
- `Returns:` section: absent. The criteria require documenting shape, dims, and values of the return. Missing. (FAIL)
- `Example:` section: present (lines 20-24), but it shows construction of a `Chord`, not the result of calling `.apply()` or even invoking `MeanAggregate` inside a full pipeline. It references `Chord`, `SlidingWindow`, and `LineLength` without importing them in the docstring context, making the example non-executable as a doctest. More importantly, the example does not show the aggregator actually being used on data and does not illustrate the output.

## Typing

- `MeanAggregate` has no dataclass fields, so no field typing is required. (PASS)
- `__call__` return type is explicit (`-> Data`). (PASS)
- Parameters `data: Data` and `stream: Iterator[Data]` are typed. (PASS)
- No bare `Any`. (PASS)

## Safety & Style

- No `print()` statements. (PASS)
- Empty-stream guard: line 29-30 raises `ValueError("MeanAggregate received an empty stream")`. (PASS)
- No `__post_init__` needed (no numeric fields). (PASS)
- Input `data` is not mutated. (PASS)
- All lines are within 100 characters. (PASS)
- No dimension validation on `data` itself (e.g., confirming `time` and `space` are present) — but since `MeanAggregate` only reads metadata from `data` (not its array values), this is low-severity.
- `sampling_rate` omission in `Data(...)` constructor (line 35): this silently drops the original sampling rate from the result. (MEDIUM — latent bug)

## Action List

1. [HIGH] Add a `Returns:` section to the docstring describing the output `Data` shape, dimensions, and values (mean across all windows).
2. [MEDIUM] Pass `sampling_rate=data.sampling_rate` to the `Data(...)` constructor on line 35 to preserve the original sampling rate in the aggregated result, matching the behaviour of `_copy_with_new_data`.
3. [MEDIUM] Expand the `Example:` to show a runnable usage that includes `.apply(data)` and illustrates what the output looks like (or at minimum add all necessary imports so it is a valid doctest).
4. [LOW] Align the parameter name on line 27 from `stream` to `windows` to match the CLAUDE.md documented convention for `AggregatorFeature.__call__` signatures.
