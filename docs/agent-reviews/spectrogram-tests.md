# Test Review: spectrogram

**Feature**: `src/cobrabox/features/spectrogram.py`
**Test file**: `tests/test_feature_spectrogram.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
spectrogram.py: 100% (55 statements, 0 missing)
```

## Summary

Comprehensive test suite with 22 tests covering all major functionality:

- Output structure and dimension handling (5 tests)
- Numerical correctness against scipy reference (6 tests)
- All four scaling modes verified (log, density, spectrum, magnitude)
- Metadata preservation (1 test)
- Extra dimensions handling (1 test)
- Parameter behavior (3 tests)
- Error handling for invalid parameters (4 tests)
- API accessibility (1 test)

The tests use appropriate numerical tolerances (`rtol=1e-10`) and verify physical correctness (pure tone peak detection within 2 Hz).

## Keep

Tests that are correct and complete — no changes needed:

- `test_spectrogram_output_dims` — verifies output has correct dims
- `test_spectrogram_space_dim_preserved` — checks coordinate preservation
- `test_spectrogram_frequency_coords_are_nonneg_and_bounded` — physical correctness
- `test_spectrogram_time_coords_are_positive` — physical correctness
- `test_spectrogram_output_shape_matches_scipy` — shape validation against reference
- `test_spectrogram_log_scaling_matches_scipy` — numerical correctness
- `test_spectrogram_density_scaling_matches_scipy` — numerical correctness
- `test_spectrogram_spectrum_scaling_matches_scipy` — numerical correctness
- `test_spectrogram_magnitude_scaling_matches_scipy` — numerical correctness
- `test_spectrogram_log_no_neg_inf` — edge case (zero signals)
- `test_spectrogram_density_values_nonneg` — physical constraint
- `test_spectrogram_pure_tone_has_peak_at_correct_freq` — physical correctness
- `test_spectrogram_preserves_metadata` — comprehensive metadata + history test
- `test_spectrogram_preserves_extra_dim` — extra dimension handling
- `test_spectrogram_custom_nperseg_changes_freq_resolution` — parameter effect
- `test_spectrogram_noverlap_changes_time_resolution` — parameter effect
- `test_spectrogram_different_windows_produce_different_results` — parameter effect
- `test_spectrogram_raises_on_invalid_scaling` — parameter validation
- `test_spectrogram_raises_when_nperseg_exceeds_n_time` — parameter validation
- `test_spectrogram_raises_when_nperseg_is_less_than_two` — parameter validation
- `test_spectrogram_raises_when_noverlap_gte_nperseg` — parameter validation
- `test_spectrogram_accessible_via_feature_module` — API check

## Fix

None.

## Add

One optional enhancement — not required for PASS:

### `test_spectrogram_missing_time_raises`

```python
def test_spectrogram_missing_time_raises() -> None:
    """Spectrogram raises error when 'time' dimension is missing."""
    import xarray as xr
    # Build data without time dimension
    bad_xr = xr.DataArray(np.random.randn(10, 3), dims=["epoch", "space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    # Feature will raise KeyError when accessing xr_data.sizes["time"]
    with pytest.raises((KeyError, ValueError)):
        cb.feature.Spectrogram().apply(raw)
```

**Note**: This scenario is implicitly covered — the feature will naturally raise `KeyError` when accessing `xr_data.sizes["time"]` on line 70 if the dimension is missing. Adding an explicit test would document this behavior but is not required for coverage.

## Action List

None.
