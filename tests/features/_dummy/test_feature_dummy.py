"""Tests for the Dummy feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.features._dummy import Dummy


def _make_data() -> cb.SignalData:
    """Create test data for Dummy feature."""
    rng = np.random.default_rng(seed=123)
    arr = rng.standard_normal((40, 8))
    return cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=200.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
        extra={"whatever": "hello"},
    )


def test_dummy_preserves_data_and_metadata() -> None:
    """Dummy returns Data with same values and propagated metadata/history."""
    data = _make_data()
    arr = data.to_numpy()

    out = Dummy(mandatory_arg=1).apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.shape == data.data.shape
    assert out.data.dims == data.data.dims
    # Dummy feature preserves sampling_rate but not other metadata (it's a negative reference)
    assert out.sampling_rate == 200.0
    assert out.history == ["Dummy"]
    # Values should be preserved (shape matches input)
    np.testing.assert_allclose(out.to_numpy(), arr)


def test_dummy_missing_time_raises() -> None:
    """Dummy raises ValueError when 'time' dimension is missing."""
    import xarray as xr

    arr = np.random.default_rng(42).standard_normal((10, 8))
    # Build Data with 't' instead of 'time' to bypass SignalData validation
    bad_xr = xr.DataArray(arr, dims=["t", "space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError, match="time"):
        Dummy(mandatory_arg=1)(raw)


def test_dummy_missing_space_raises() -> None:
    """Dummy raises ValueError when 'space' dimension is missing."""
    import xarray as xr

    arr = np.random.default_rng(42).standard_normal((10, 8))
    bad_xr = xr.DataArray(arr, dims=["time", "s"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError, match="space"):
        Dummy(mandatory_arg=1)(raw)


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


def test_dummy_optional_arg() -> None:
    """Dummy accepts optional_arg parameter."""
    data = _make_data()
    # Just verify it accepts the parameter without error
    out = Dummy(mandatory_arg=1, optional_arg=42).apply(data)
    assert isinstance(out, cb.Data)
