# Test Review: spikes_calc

**Feature**: `src/cobrabox/features/spikes_calc.py`
**Test file**: `tests/test_feature_spikes_calc.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
spikes_calc.py: 100% (21 statements, 0 missing)
```

## Summary

Excellent test coverage for SpikesCalc. The tests comprehensively cover the IQR-based outlier detection, including clean data, data with outliers, metadata preservation, scalar output format, empty data handling, and boundary value testing.

## Keep

- `test_spikes_calc_clean_data_no_outliers` — Tests clean normal distribution returns ~0 spikes
- `test_spikes_calc_with_outliers` — Tests detection of extreme values beyond IQR bounds
- `test_spikes_calc_preserves_metadata` — Tests all metadata preservation (subjectID, groupID, condition, extra)
- `test_spikes_calc_returns_scalar` — Tests output is scalar with shape ()
- `test_spikes_calc_multivariate_data` — Tests works on multichannel data
- `test_spikes_calc_empty_data_raises` — Tests ValueError for empty input
- `test_spikes_calc_sampling_rate_none` — Tests sampling_rate is None (time dimension removed)
- `test_spikes_calc_does_not_mutate_input` — Tests input Data is unchanged
- `test_spikes_calc_boundary_values` — Tests values exactly at IQR bounds are not spikes

## Action List

None.
