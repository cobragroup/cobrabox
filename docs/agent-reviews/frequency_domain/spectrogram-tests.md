# Test Review: spectrogram

**Feature**: `src/cobrabox/features/spectrogram.py`
**Test file**: `tests/test_feature_spectrogram.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
spectrogram.py: 100% (55 statements, 0 missing)
```

## Summary

Comprehensive test suite with 22 tests covering all major functionality. The tests verify output structure, numerical correctness against scipy reference implementations, all four scaling modes (log, density, spectrum, magnitude), extra dimension handling, parameter behavior, error handling, and metadata propagation. Coverage is at 100%.

## Keep

Tests that are correct and complete:

- `test_spectrogram_output_dims` — verifies output dimensions are (space, frequency, time)
- `test_spectrogram_space_dim_preserved` — confirms space coordinates are unchanged
- `test_spectrogram_frequency_coords_are_nonneg_and_bounded` — validates frequency axis bounds
- `test_spectrogram_time_coords_are_positive` — checks time axis values are positive window centres
- `test_spectrogram_output_shape_matches_scipy` — confirms shape matches scipy spectrogram output
- `test_spectrogram_log_scaling_matches_scipy` — numerical correctness for log scaling
- `test_spectrogram_density_scaling_matches_scipy` — numerical correctness for density scaling
- `test_spectrogram_spectrum_scaling_matches_scipy` — numerical correctness for spectrum scaling
- `test_spectrogram_magnitude_scaling_matches_scipy` — numerical correctness for magnitude scaling
- `test_spectrogram_log_no_neg_inf` — verifies log scaling clamps near-zero values
- `test_spectrogram_density_values_nonneg` — confirms density values are non-negative
- `test_spectrogram_pure_tone_has_peak_at_correct_freq` — signal processing sanity check with sine wave
- `test_spectrogram_preserves_metadata` — validates subjectID, groupID, condition, sampling_rate, extra, history
- `test_spectrogram_preserves_extra_dim` — extra dimensions (e.g., window_index) are handled correctly
- `test_spectrogram_custom_nperseg_changes_freq_resolution` — parameter effect on frequency bins
- `test_spectrogram_noverlap_changes_time_resolution` — parameter effect on time bins
- `test_spectrogram_different_windows_produce_different_results` — window parameter changes output
- `test_spectrogram_raises_on_invalid_scaling` — error handling for invalid scaling parameter
- `test_spectrogram_raises_when_nperseg_exceeds_n_time` — runtime validation of nperseg
- `test_spectrogram_raises_when_nperseg_is_less_than_two` — validation of minimum nperseg
- `test_spectrogram_raises_when_noverlap_gte_nperseg` — validation of noverlap constraint
- `test_spectrogram_accessible_via_feature_module` — API accessibility check

## Fix

None.

## Add

Missing scenarios — new tests to add:

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

1. [Severity: LOW] Add `test_spectrogram_does_not_mutate_input` to verify input Data is not modified (tests/test_feature_spectrogram.py:340)
