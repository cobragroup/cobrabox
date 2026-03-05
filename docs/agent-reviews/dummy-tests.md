# Test Review: dummy

**Feature**: `src/cobrabox/features/dummy.py`
**Test file**: `tests/test_feature_dummy.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

**0%** — Module `src.cobrabox.features.dummy` was never imported during test execution.

The test file exists and passes, but the Dummy feature code is not being exercised. The test uses `cb.feature.Dummy` which may resolve to an auto-discovered reference that doesn't actually import the source file. This means the implementation in `dummy.py` (lines 36-42, including both `ValueError` guards and the round-trip logic) is completely untested.

**Severity: HIGH** — Coverage is far below the 95% threshold.

## Summary

The test file has one comprehensive test that checks metadata preservation and history, but fails to actually execute the feature code due to an import/discovery issue. Additionally, several required scenarios are missing: validation errors for missing dimensions, input mutation checks, and tests for the optional parameter.

## Keep

None — the existing test does not actually exercise the feature code.

## Fix

### `test_dummy_feature_preserves_data_and_metadata`
Issue: Test name does not follow naming convention (should be `test_dummy_<scenario>`), and more critically, the test does not actually import or execute `src/cobrabox/features/dummy.py`.

Corrected version should explicitly import the feature class:

```python
from cobrabox.features.dummy import Dummy

def test_dummy_preserves_data_and_metadata() -> None:
    """Dummy returns Data with same values and propagated metadata/history."""
    rng = np.random.default_rng(seed=123)
    arr = rng.standard_normal((40, 8))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=200.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
        extra={"whatever": "hello"},
    )

    out = Dummy(mandatory_arg=1).apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.shape == data.data.shape
    assert out.data.dims == data.data.dims
    assert out.subjectID == "sub-01"
    assert out.groupID == "patient"
    assert out.condition == "rest"
    assert out.sampling_rate == 200.0
    assert out.extra.get("whatever") == "hello"
    assert out.history == ["Dummy"]
    np.testing.assert_allclose(out.to_numpy(), arr.T)
```

## Add

### `test_dummy_missing_time_raises`

```python
def test_dummy_missing_time_raises() -> None:
    """Dummy raises ValueError when 'time' dimension is missing."""
    import xarray as xr
    arr = np.random.randn(10, 8)
    # Build Data with 't' instead of 'time' to bypass SignalData validation
    bad_xr = xr.DataArray(arr, dims=["t", "space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError, match="time"):
        Dummy(mandatory_arg=1)(raw)
```

### `test_dummy_missing_space_raises`

```python
def test_dummy_missing_space_raises() -> None:
    """Dummy raises ValueError when 'space' dimension is missing."""
    import xarray as xr
    arr = np.random.randn(10, 8)
    bad_xr = xr.DataArray(arr, dims=["time", "s"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError, match="space"):
        Dummy(mandatory_arg=1)(raw)
```

### `test_dummy_does_not_mutate_input`

```python
def test_dummy_does_not_mutate_input() -> None:
    """Dummy.apply() leaves the input Data object unchanged."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = Dummy(mandatory_arg=1).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_dummy_optional_arg`

```python
def test_dummy_optional_arg() -> None:
    """Dummy accepts optional_arg parameter."""
    data = _make_data()
    # Just verify it accepts the parameter without error
    out = Dummy(mandatory_arg=1, optional_arg=42).apply(data)
    assert isinstance(out, cb.Data)
```

## Action List

1. [Severity: HIGH] Fix import issue — explicitly import `Dummy` from `cobrabox.features.dummy` instead of relying on `cb.feature.Dummy` discovery which doesn't import the source file (`tests/test_feature_dummy.py:7-8`)

2. [Severity: HIGH] Add test for missing 'time' dimension validation (`src/cobrabox/features/dummy.py:36-37`)

3. [Severity: HIGH] Add test for missing 'space' dimension validation (`src/cobrabox/features/dummy.py:38-39`)

4. [Severity: MEDIUM] Add test for input mutation protection

5. [Severity: LOW] Rename test function to follow convention: `test_dummy_preserves_data_and_metadata` instead of `test_dummy_feature_preserves_data_and_metadata`

6. [Severity: LOW] Add test for optional_arg parameter
