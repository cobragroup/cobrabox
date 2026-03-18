# Test Review: SVD

**Feature**: `src/cobrabox/features/reductions/svd.py`
**Test file**: `tests/features/reductions/test_feature_svd.py`
**Date**: 2026-03-18
**Verdict**: NEEDS WORK

## Coverage

```text
SVD: 94% (68 statements, 4 missing: 86, 97, 103, 159)
```

Coverage is close to threshold at 94%, missing 4 lines:

- Line 86: `if output != "U"` — condition for storing `U` in attrs when `output="U"`
- Line 97: `if self.output != "U"` — same condition
- Line 103: `if mean_da is not None` — centering condition
- Line 159: `if std_da is not None` — z-score condition

Lines 103 and 159 are partially covered by existing tests (center=True, zscore=True), but the uncovered state `center=False, zscore=False` is never tested. The most significant gap is **no test for `output="U"` mode**, which covers lines 86 and 97.

## Summary

Strong test coverage with 10 tests covering basic usage, multi-dimensional unstacking, masking, centering/z-score properties, reconstruction, validation, metadata preservation, and immutability. Tests are well-named (`test_feature_svd_*` pattern), have clear docstrings, include `match=` in `pytest.raises`, and use `pytest.approx` correctly. Coverage at 94% is good but falls short of the 95% threshold due to missing test for `output="U"` and the `center=False, zscore=False` configuration.

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_svd_returns_dataarray_and_updates_history` — verifies basic V output, history, and attrs structure
- `test_feature_svd_unstacks_V_for_multidim_input` — confirms V unstacking for 4D input
- `test_feature_svd_mask_reduces_feature_count` — validates mask functionality and feature reduction
- `test_feature_svd_center_makes_feature_means_zero` — mathematically verifies centering zeroing means
- `test_feature_svd_zscore_makes_means_zero_and_stds_one` — mathematically verifies z-score normalization
- `test_feature_svd_reconstruction_matches_centered_matrix_full_rank_case` — validates SVD reconstruction accuracy
- `test_feature_svd_raises_for_unknown_dimension` — proper error test with `match=`
- `test_feature_svd_n_components_non_positive_raises` — proper parameter validation error test
- `test_feature_svd_metadata_preserved` — confirms metadata preservation and `sampling_rate=None` for Data return
- `test_feature_svd_does_not_mutate_input` — verifies immutability of input

## Fix

All existing tests are correct — no fixes needed.

## Add

Missing scenarios — new tests to add:

### `test_feature_svd_output_u_mode`

Test the `output="U"` mode which returns U (sample scores/timecourses) instead of V (feature patterns), and stores V in attrs. This covers lines 86 and 97 that are currently uncovered.

```python
def test_feature_svd_output_u_mode() -> None:
    """SVD with output='U' returns U and stores V in attrs."""
    arr = np.random.default_rng(42).normal(size=(30, 5)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "channel"], sampling_rate=100.0)

    out = cb.feature.SVD(dim="time", n_components=3, center=True, output="U").apply(data)

    assert out.data.name == "U"
    assert out.data.dims == ("time", "component")
    assert out.data.shape == (30, 3)

    svd = out.data.attrs["svd"]
    assert svd["U"] is None  # U is primary output, not stored in attrs
    assert svd["Vh"] is not None
    assert svd["Vh"].shape == (3, 5)
    assert svd["S"].shape == (3,)
```

### `test_feature_svd_no_centering_no_zscore`

Test the `center=False, zscore=False` configuration to cover the code paths where `mean_da` and `std_da` remain `None`. This covers lines 103 and 159.

```python
def test_feature_svd_no_centering_no_zscore() -> None:
    """SVD with center=False, zscore=False skips normalization and stores None in attrs."""
    arr = np.random.default_rng(42).normal(size=(25, 4)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SVD(
        dim="time",
        n_components=3,
        center=False,
        zscore=False,
        return_unstacked_V=False,
    ).apply(data)

    svd = out.data.attrs["svd"]
    assert svd["mean"] is None
    assert svd["std"] is None
    assert svd["masked"] is False
    assert svd["center"] is False
    assert svd["zscore"] is False
```

## Action List

1. [HIGH] Add `test_feature_svd_output_u_mode` to test `output="U"` parameter option (lines 86, 97)
2. [HIGH] Add `test_feature_svd_no_centering_no_zscore` to test `center=False, zscore=False` (lines 103, 159) — will bring coverage to ≥95%
