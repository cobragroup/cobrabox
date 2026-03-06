# Test Review: wavelet_transform

**Feature**: `src/cobrabox/features/wavelet_transform.py`
**Test file**: `tests/test_feature_wavelet_transform.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```
wavelet_transform.py: 100% (99 statements, 0 missing)
```

Coverage is at 100% — no gaps.

## Summary

The test file is comprehensive and well-structured. It covers output structure, numerical
correctness (verified against pywt directly), NaN-padding behaviour, extra dimensions,
metadata propagation, parameter validation, and API accessibility for both classes. Two
gaps were found: the DWT metadata test omits an assertion that `sampling_rate` becomes
`None` after `output_type = Data` strips the time axis (a required scenario per criteria),
and the CWT metadata test compares `sampling_rate` with `==` rather than `pytest.approx`.

## Keep

- `test_dwt_output_dims` — correct isinstance and dims check.
- `test_dwt_output_type_is_data_not_signal_data` — correctly uses `type(out) is cb.Data`.
- `test_dwt_level_labels_correct_order` — explicit coordinate list comparison.
- `test_dwt_n_levels_equals_level_plus_one` — parametric loop over multiple levels.
- `test_dwt_space_coords_preserved` — uses named string coords.
- `test_dwt_coef_index_length_matches_finest_detail` — cross-checks against pywt directly.
- `test_dwt_approx_matches_pywt` — channel-by-channel allclose against pywt.
- `test_dwt_detail_coefficients_match_pywt` — covers all detail levels, maps pywt order correctly.
- `test_dwt_shorter_levels_are_nan_padded` — checks both the finite prefix and the NaN tail.
- `test_dwt_finest_detail_has_no_nan` — confirms finest detail is always fully populated.
- `test_dwt_default_level_uses_max` — validates the `level=None` branch.
- `test_dwt_preserves_window_dim` — extra-dim coverage.
- `test_dwt_does_not_mutate_input` — immutability check.
- `test_dwt_higher_level_gives_more_nan_in_approx` — good sanity check on NaN extent.
- `test_dwt_different_wavelets_produce_different_results` — validates wavelet dispatch via pywt.
- `test_dwt_raises_on_invalid_wavelet`, `test_dwt_raises_on_level_zero`, `test_dwt_raises_when_level_exceeds_max` — all use `match=`.
- `test_dwt_accessible_via_feature_module` — discovery check.
- All CWT tests — solid coverage of structure, coords, numerics, extra dims, metadata, and errors.

## Fix

### `test_dwt_preserves_metadata`

**Issue**: Missing `assert out.sampling_rate is None`. With `output_type = Data` the time
axis is consumed; the criteria requires asserting that `sampling_rate` becomes `None`
when a feature removes the time dimension.

```python
def test_dwt_preserves_metadata(self) -> None:
    """DWT propagates subjectID, groupID, condition, sets sampling_rate to None, and appends to history."""
    data = _make_data()
    out = cb.feature.DiscreteWaveletTransform(level=2).apply(data)
    assert out.subjectID == "sub-01"
    assert out.groupID == "ctrl"
    assert out.condition == "rest"
    assert out.sampling_rate is None  # time axis consumed → no meaningful sampling rate
    assert out.history[-1] == "DiscreteWaveletTransform"
```

### `test_cwt_preserves_metadata`

**Issue**: `assert out.sampling_rate == SR` compares a float with `==`. Criteria prefers
`pytest.approx` for float comparisons.

```python
def test_cwt_preserves_metadata(self) -> None:
    """CWT propagates subjectID, groupID, condition, sampling_rate and appends history."""
    data = _make_data()
    out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
    assert out.subjectID == "sub-01"
    assert out.groupID == "ctrl"
    assert out.condition == "rest"
    assert out.sampling_rate == pytest.approx(SR)
    assert out.history[-1] == "ContinuousWaveletTransform"
```

## Add

No missing scenarios.

## Action List

1. [Severity: MEDIUM] Add `assert out.sampling_rate is None` to `test_dwt_preserves_metadata` (line 216) — required by the `output_type = Data` metadata scenario.
2. [Severity: LOW] Replace `assert out.sampling_rate == SR` with `assert out.sampling_rate == pytest.approx(SR)` in `test_cwt_preserves_metadata` (line 482).
