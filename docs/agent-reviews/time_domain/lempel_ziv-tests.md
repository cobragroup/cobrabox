# Test Review: lempel_ziv

**Feature**: `src/cobrabox/features/lempel_ziv.py`
**Test file**: `tests/test_feature_lempel_ziv.py`
**Date**: 2025-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
lempel_ziv.py: 100% (47 statements, 0 missing)
```

## Summary

Good test coverage at 100% with 8 tests covering the core functionality. The tests verify output types, history tracking, metadata preservation (partial), known value correctness, multichannel independence, Chord compatibility, and input immutability. However, the metadata preservation test only checks `subjectID` and misses `groupID` and `condition`. Also, the happy path test does not explicitly verify that output values are finite (non-NaN/non-inf).

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_lempel_ziv_known_value` — correctly verifies against hand-computed expected value
- `test_feature_lempel_ziv_random_more_complex_than_periodic` — good property-based test
- `test_feature_lempel_ziv_values_are_positive` — appropriate sanity check
- `test_feature_lempel_ziv_via_chord` — verifies Chord integration and history propagation
- `test_feature_lempel_ziv_multichannel_independent` — tests per-channel independence correctly
- `test_feature_lempel_ziv_missing_time_raises` — appropriate error case
- `test_feature_lempel_ziv_does_not_mutate_input` — comprehensive immutability check

## Fix

Tests that exist but need changes:

### `test_feature_lempel_ziv_output_type_and_history`

Issue: Missing `groupID` and `condition` in metadata preservation check, and missing finite value assertion.

```python
def test_feature_lempel_ziv_output_type_and_history() -> None:
    """LempelZiv removes the time dimension and records history."""
    arr = np.random.default_rng(0).standard_normal((100, 3))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=256.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="rest",
    )
    out = cb.feature.LempelZiv().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (3,)
    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate is None
    assert out.history == ["LempelZiv"]
    assert np.all(np.isfinite(out.to_numpy()))  # Add this assertion
```

## Add

No new tests required — existing test suite covers all required scenarios.

## Action List

1. [Severity: MEDIUM] Add `groupID` and `condition` assertions to `test_feature_lempel_ziv_output_type_and_history` (tests/test_feature_lempel_ziv.py:14)
2. [Severity: LOW] Add `np.all(np.isfinite(out.to_numpy()))` assertion to `test_feature_lempel_ziv_output_type_and_history` (tests/test_feature_lempel_ziv.py:14)
