# Test Review: concat_aggregate

**Feature**: `src/cobrabox/features/concat_aggregate.py`
**Test file**: `tests/test_feature_concat_aggregate.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Coverage

```text
concat_aggregate.py: 100% (16 statements, 0 missing)
```

Coverage is exemplary — all 11 tests pass and every line is exercised.

## Summary

Solid test suite with good breadth: happy path, shape, window coordinates, single-window,
empty-stream error, metadata, history, immutability, `Data` instance type, Chord
integration, and namespace accessibility are all covered. Two gaps against the criteria:
(1) the basic test does not assert values are NaN-free, and (2) the metadata test only
exercises the `sampling_rate is None` branch (time dim removed by `LineLength`); the
path where `ConcatAggregate` preserves `sampling_rate` when time survives is untested.
Helper function returns `cb.Data` instead of `cb.SignalData` — minor style point.

## Keep

- `test_concat_aggregate_basic` — verifies window dim exists and count is correct ✅
- `test_concat_aggregate_shape` — concrete shape assertion `(5, 4)` ✅
- `test_concat_aggregate_integer_coordinates` — uses `np.testing.assert_array_equal` ✅
- `test_concat_aggregate_single_window` — edge case ✅
- `test_concat_aggregate_empty_stream_raises` — `match="empty stream"` ✅
- `test_concat_aggregate_preserves_original_metadata` — subjectID/groupID/condition ✅
- `test_concat_aggregate_propagates_window_history` — checks `history[-1]` ✅
- `test_concat_aggregate_does_not_mutate_original` — checks history and shape for both
  original and each window ✅

- `test_concat_aggregate_returns_data_instance` — `isinstance(result, cb.Data)` ✅
- `test_concat_aggregate_via_chord` — full end-to-end with pipe syntax ✅
- `test_concat_aggregate_accessible_via_cb_feature` — namespace identity check ✅

## Fix

### `test_concat_aggregate_basic`

Missing NaN-free assertion — required by the happy-path criteria.

```python
def test_concat_aggregate_basic() -> None:
    """ConcatAggregate stacks windows along a new 'window' dimension."""
    original_data = _make_data(n_time=5, n_space=2)
    windows = _make_windows(n_windows=3)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(windows))

    assert isinstance(result, cb.Data)
    assert "window" in result.data.dims
    assert result.data.sizes["window"] == 3
    assert not np.any(np.isnan(result.data.values))  # add this line
```

### `test_concat_aggregate_preserves_original_metadata`

Only the `sampling_rate is None` branch (time dim gone after `LineLength`) is tested.
Add a second case using a pipeline that keeps the time dimension so that the
`sampling_rate` pass-through path is also exercised.

```python
def test_concat_aggregate_preserves_sampling_rate_with_time_dim() -> None:
    """ConcatAggregate preserves sampling_rate when windows still have a time dimension."""
    import xarray as xr

    original_data = _make_data(n_time=5, n_space=2, sampling_rate=250.0)
    # Build windows that still have a time dimension (no reduction step)
    raw_windows = []
    for i in range(3):
        arr = np.ones((5, 2)) * i
        w = cb.SignalData.from_numpy(
            arr,
            dims=["time", "space"],
            sampling_rate=250.0,
            subjectID="s1",
        )
        raw_windows.append(w)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(raw_windows))

    # Result has (window, time, space) — Data sees time dim and preserves sampling_rate
    assert "time" in result.data.dims
    assert result.sampling_rate == pytest.approx(250.0)
```

## Add

No additional scenarios required — all mandatory categories from the criteria are covered.

## Action List

1. [Severity: MEDIUM] `test_concat_aggregate_basic` is missing `assert not np.any(np.isnan(...))` — add per happy-path criteria (line 56 in test file).
2. [Severity: MEDIUM] `sampling_rate` preservation when time dimension survives is untested — add `test_concat_aggregate_preserves_sampling_rate_with_time_dim` (new test).
3. [Severity: LOW] `_make_data` helper returns `cb.Data` — test-patterns convention recommends `cb.SignalData.from_numpy` for the original data helper (not blocking, since the aggregator accepts any `Data`).
