# Test Review: spectrogram

**Feature**: `src/cobrabox/features/spectrogram.py`
**Test file**: `tests/test_feature_spectrogram.py`
**Date**: 2025-03-04
**Verdict**: NEEDS WORK

## Summary

The spectrogram test suite is extensive and well-structured with 26 tests covering output structure, numerical correctness, all four scaling modes, parameter behavior, extra dimensions, and error cases. The tests follow naming conventions and include docstrings. However, two required scenarios are missing: (1) validation that the feature raises when the `time` dimension is absent, and (2) a no-mutation test verifying the input Data object remains unchanged after application.

## Keep

Tests that are correct and complete — no changes needed:

- `test_spectrogram_output_dims` — verifies output dims are (space, frequency, time)
- `test_spectrogram_space_dim_preserved` — checks space coordinates are preserved
- `test_spectrogram_frequency_coords_are_nonneg_and_bounded` — validates Nyquist constraint
- `test_spectrogram_time_coords_are_positive` — checks time axis values
- `test_spectrogram_output_shape_matches_scipy` — validates shape against scipy reference
- `test_spectrogram_log_scaling_matches_scipy` — numerical correctness for log scaling
- `test_spectrogram_density_scaling_matches_scipy` — numerical correctness for density scaling
- `test_spectrogram_spectrum_scaling_matches_scipy` — numerical correctness for spectrum scaling
- `test_spectrogram_magnitude_scaling_matches_scipy` — numerical correctness for magnitude scaling
- `test_spectrogram_log_no_neg_inf` — validates -inf clamping for near-zero signals
- `test_spectrogram_density_values_nonneg` — verifies PSD non-negativity
- `test_spectrogram_pure_tone_has_peak_at_correct_freq` — end-to-end frequency accuracy test
- `test_spectrogram_preserves_metadata` — comprehensive metadata preservation check
- `test_spectrogram_preserves_extra_dim` — validates extra dimensions (window_index) handling
- `test_spectrogram_custom_nperseg_changes_freq_resolution` — parameter effect test
- `test_spectrogram_noverlap_changes_time_resolution` — parameter effect test
- `test_spectrogram_different_windows_produce_different_results` — window parameter test
- `test_spectrogram_raises_on_invalid_scaling` — error case for invalid scaling parameter
- `test_spectrogram_raises_when_nperseg_exceeds_n_time` — runtime validation test
- `test_spectrogram_raises_when_nperseg_is_less_than_two` — runtime validation test
- `test_spectrogram_raises_when_noverlap_gte_nperseg` — runtime validation test
- `test_spectrogram_accessible_via_feature_module` — API accessibility test

## Fix

None — all existing tests are correct.

## Add

Missing scenarios — new tests to add:

### `test_spectrogram_missing_time_raises`

```python
def test_spectrogram_missing_time_raises() -> None:
    """Spectrogram raises ValueError when 'time' dimension is missing."""
    import xarray as xr
    # Build Data without time dimension using __new__ bypass
    bad_xr = xr.DataArray(np.random.randn(100, 10), dims=["epoch", "space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError, match="time"):
        cb.feature.Spectrogram().apply(raw)
```

### `test_spectrogram_does_not_mutate_input`

```python
def test_spectrogram_does_not_mutate_input() -> None:
    """Spectrogram.apply() leaves the input Data object unchanged."""
    rng = np.random.default_rng(42)
    data = _make_data(rng.standard_normal((256, 3)))
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Spectrogram().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: HIGH] Add missing test: `test_spectrogram_missing_time_raises` — validates that Spectrogram raises ValueError when input lacks 'time' dimension (`tests/test_feature_spectrogram.py`, after line 330)
2. [Severity: HIGH] Add missing test: `test_spectrogram_does_not_mutate_input` — verifies input Data object is not mutated (`tests/test_feature_spectrogram.py`, after line 340)
