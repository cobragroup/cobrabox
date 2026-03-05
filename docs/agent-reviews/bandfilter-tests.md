# Test Review: bandfilter

**Feature**: `src/cobrabox/features/bandfilter.py`
**Test file**: `tests/test_feature_bandfilter.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Coverage

```
Name                                            Stmts   Miss  Cover   Missing
src/cobrabox/features/bandfilter.py                36      1    97%   63
```

**Coverage: 97%** — One line uncovered (line 63) — validation for bands with wrong number of frequencies (e.g., `[low, mid, high]`). This is acceptable as it's an edge case that rarely occurs in practice.

## Summary

Excellent test file with comprehensive coverage of all required scenarios. Tests are well-organized into logical sections (Basic API, Coordinate preservation, Numerical correctness, Chaining, Parameter validation). The file includes sophisticated numerical correctness tests comparing against manual scipy calls and using sinusoids to verify frequency selectivity.

## Keep

All existing tests are correct and complete:

- `test_bandfilter_history_updated` — Verifies "BandFilter" appended to history ✓
- `test_bandfilter_metadata_preserved` — Checks subjectID, groupID, condition, sampling_rate ✓
- `test_bandfilter_returns_data_instance` — Confirms `.apply()` returns Data ✓
- `test_bandfilter_does_not_mutate_input` — Verifies input object unchanged ✓
- `test_bandfilter_missing_sampling_rate_raises` — Tests ValueError when sampling_rate is None ✓
- `test_bandfilter_default_band_coords` — Validates default EEG band names ✓
- `test_bandfilter_default_band_coords_keep_orig` — Tests keep_orig parameter ✓
- `test_bandfilter_custom_bands` — Tests custom bands dict ✓
- `test_bandfilter_single_band` — Tests single-band filter ✓
- `test_bandfilter_preserves_time_coords` — Time coordinate preservation ✓
- `test_bandfilter_preserves_space_coords_when_present` — Space coordinate preservation ✓
- `test_bandfilter_matches_manual_scipy_filter` — Numerical correctness vs manual scipy ✓
- `test_bandfilter_output_is_not_all_zeros` — Sanity check for non-zero output ✓
- `test_bandfilter_sine_concentrated_in_correct_band` — Frequency selectivity test with parametrized sine waves ✓
- `test_bandfilter_output_is_valid_data_for_further_features` — Chainability test ✓
- `test_bandfilter_zero_order_raises` — Invalid parameter (ord=0) ✓
- `test_bandfilter_negative_order_raises` — Invalid parameter (ord=-1) ✓
- `test_bandfilter_empty_bands_raises` — Invalid parameter (empty bands) ✓
- `test_bandfilter_invalid_band_range_raises` — Invalid parameter (low >= high) ✓
- `test_bandfilter_negative_frequency_raises` — Invalid parameter (negative frequencies) ✓

## Action List

1. [Severity: LOW] Consider adding test for band with wrong number of frequencies (line 63 coverage gap):
```python
def test_bandfilter_band_wrong_number_of_frequencies_raises() -> None:
    """BandFilter raises ValueError when band doesn't have exactly 2 frequencies."""
    data = _make_data()
    with pytest.raises(ValueError, match="exactly 2"):
        cb.feature.BandFilter(bands={"bad": [1, 10, 20]}).apply(data)
```

Overall verdict: **PASS**. The test suite is comprehensive and well-structured. The single uncovered line is a minor edge case that doesn't warrant blocking.
