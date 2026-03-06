# Test Review: emd

**Feature**: `src/cobrabox/features/emd.py`
**Test file**: `tests/test_feature_emd.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
EMD: 100% (61 statements, 0 missing)
```

## Summary

Excellent test coverage for the EMD feature. All 61 statements are covered. The test file contains 29 comprehensive tests covering happy path, parameter validation, multi-dimensional data handling, n_imfs tracking, and feature chaining. Tests are well-organized into logical sections with clear docstrings.

## Keep

Tests that are correct and complete:

- `test_emd_history_updated` — verifies "EMD" appended to history
- `test_emd_metadata_preserved` — checks all metadata fields preserved
- `test_emd_returns_data_instance` — verifies Data instance returned
- `test_emd_does_not_mutate_input` — ensures input not modified
- `test_emd_adds_imf_dimension` — verifies imf dimension added
- `test_emd_imf_coords_labelled_correctly` — checks coordinate labels
- `test_emd_residual_always_included` — confirms residual present
- `test_emd_keep_orig_includes_original` — tests keep_orig=True
- `test_emd_keep_orig_false_excludes_original` — tests keep_orig=False
- `test_emd_keep_orig_original_matches_input` — validates original signal matches
- `test_emd_preserves_time_coords` — time coordinates preserved
- `test_emd_preserves_space_coords` — space coordinates preserved
- `test_emd_output_not_all_zeros` — output contains non-zero values
- `test_emd_imfs_sum_to_original` — IMFs reconstruct original signal
- `test_emd_extracts_oscillatory_components` — validates decomposition
- `test_emd_different_methods_work` — parametrized test for all 3 sift methods
- `test_emd_3d_data` — handles 3D data correctly
- `test_emd_1d_data` — handles 1D data correctly
- `test_emd_1d_data_n_imfs_in_attrs` — n_imfs dict for 1D data
- `test_emd_1d_data_n_imfs_correct_with_keep_orig` — n_imfs count with keep_orig
- `test_emd_multidim_n_imfs_in_attrs` — n_imfs dict for multi-dimensional
- `test_emd_multidim_n_imfs_keys_match_coords` — n_imfs keys match coordinates
- `test_emd_3d_data_n_imfs_dict` — n_imfs for 3D with "/" separator
- `test_emd_output_is_valid_for_further_features` — chaining works
- `test_emd_invalid_method_raises` — invalid method rejected
- `test_emd_zero_max_imfs_raises` — zero max_imfs rejected
- `test_emd_negative_max_imfs_raises` — negative max_imfs rejected

## Action List

None.
