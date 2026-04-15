"""Tests for general Data container (no dimension requirements)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

RNG = np.random.default_rng(seed=42)


def test_data_from_numpy_basic() -> None:
    """Data.from_numpy creates a Data object with arbitrary dimensions."""
    a = RNG.standard_normal((100, 10))
    ds = cb.Data.from_numpy(a, dims=["time", "space"], sampling_rate=100.0)
    # Data preserves original dimension order
    assert ds.data.shape == (100, 10)
    assert ds.data.dims == ("time", "space")
    assert ds.sampling_rate == pytest.approx(100.0)
    np.testing.assert_array_almost_equal(ds.to_numpy(), a)


def test_data_from_numpy_1d() -> None:
    """Data.from_numpy accepts 1-D arrays."""
    a = RNG.standard_normal(20)
    ds = cb.Data.from_numpy(a, dims=["time"], sampling_rate=10.0)
    assert ds.data.shape == (20,)
    assert ds.data.dims == ("time",)


def test_data_from_numpy_no_time() -> None:
    """Data.from_numpy works without time dimension."""
    a = RNG.standard_normal((5, 3))
    ds = cb.Data.from_numpy(a, dims=["x", "y"])
    assert ds.data.shape == (5, 3)
    assert ds.data.dims == ("x", "y")
    assert ds.sampling_rate is None  # No time dimension


def test_data_from_xarray_basic() -> None:
    """Data.from_xarray wraps a DataArray with arbitrary dimensions."""
    ar = xr.DataArray(
        RNG.standard_normal((30, 6)),
        dims=["time", "space"],
        coords={"time": np.arange(30) / 100.0, "space": [f"ch{i}" for i in range(6)]},
    )
    ds = cb.Data.from_xarray(ar)
    # Data preserves original dimension order
    assert ds.data.shape == (30, 6)
    assert ds.data.dims == ("time", "space")
    assert ds.sampling_rate == pytest.approx(100.0)


def test_data_from_xarray_no_time() -> None:
    """Data.from_xarray works without time dimension."""
    ar = xr.DataArray(RNG.standard_normal((5, 3)), dims=["x", "y"])
    ds = cb.Data.from_xarray(ar)
    assert ds.data.shape == (5, 3)
    assert ds.data.dims == ("x", "y")
    assert ds.sampling_rate is None


def test_data_sampling_rate_none_without_time() -> None:
    """Data without time dimension has sampling_rate=None."""
    ar = xr.DataArray(np.ones((3, 2)), dims=["foo", "bar"])
    ds = cb.Data(ar)
    assert ds.sampling_rate is None


def test_data_invalid_sampling_rate() -> None:
    """Data rejects non-positive sampling_rate when provided."""
    with pytest.raises(ValueError, match="must be positive"):
        cb.Data.from_numpy(RNG.standard_normal((10, 5)), dims=["time", "space"], sampling_rate=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        cb.Data.from_numpy(RNG.standard_normal((10, 5)), dims=["time", "space"], sampling_rate=-1.0)


def test_data_from_numpy_dims_validation() -> None:
    """Data.from_numpy validates dims length matches array ndim."""
    with pytest.raises(ValueError, match="dims length must match"):
        cb.Data.from_numpy(RNG.standard_normal((10, 5, 2)), dims=["time", "space"])


def test_data_dtype_is_float64() -> None:
    """Data stores float64 for int/float inputs, preserves complex."""
    # Int and float types should be cast to float64
    for dtype in [np.float32, np.int16, np.int32, np.float16]:
        a = np.ones((10, 4), dtype=dtype)
        ds = cb.Data.from_numpy(a, dims=["time", "space"])
        assert ds.data.dtype == np.float64, f"expected float64 for input dtype {dtype}"


def test_data_dtype_complex_preserved() -> None:
    """Data preserves complex dtype for complex inputs."""
    # Test with complex128
    a_complex128 = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
    ds = cb.Data.from_numpy(a_complex128, dims=["time", "space"])
    assert ds.data.dtype == np.complex128
    np.testing.assert_array_equal(ds.to_numpy(), a_complex128)

    # Test with complex64 (should be preserved as complex128 due to cast)
    a_complex64 = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    ds = cb.Data.from_numpy(a_complex64, dims=["time", "space"])
    assert ds.data.dtype == np.complex128
    np.testing.assert_array_almost_equal(ds.to_numpy(), a_complex64.astype(np.complex128))


def test_data_from_xarray_complex_preserved() -> None:
    """Data.from_xarray preserves complex dtype."""
    ar = xr.DataArray(np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128), dims=["time"])
    ds = cb.Data.from_xarray(ar)
    assert ds.data.dtype == np.complex128
    np.testing.assert_array_equal(ds.to_numpy(), ar.values)


def test_data_immutability() -> None:
    """Setting any attribute on a Data instance raises AttributeError."""
    ds = cb.Data.from_numpy(RNG.standard_normal((5, 2)), dims=["time", "space"])
    with pytest.raises(AttributeError, match="Cannot modify attribute"):
        ds.foo = "bar"


def test_data_to_pandas() -> None:
    """Data.to_pandas() returns a pandas DataFrame."""
    import pandas as pd

    ar = xr.DataArray(np.ones((4, 3)), dims=["time", "space"], name="signal")
    ds = cb.Data.from_xarray(ar)
    df = ds.to_pandas()
    assert isinstance(df, pd.DataFrame)


def test_data_repr() -> None:
    """Data.__repr__ returns expected format with shape, dims, sr, subject."""
    ds = cb.Data.from_numpy(
        np.ones((10, 5)), dims=["time", "space"], sampling_rate=100.0, subjectID="sub-01"
    )
    r = repr(ds)
    assert "shape=(10, 5)" in r
    assert "dims=['time', 'space']" in r
    assert "sr=100.0" in r
    assert "subject='sub-01'" in r


def test_data_repr_no_sampling_rate() -> None:
    """Data.__repr__ omits sampling_rate when None."""
    ds = cb.Data.from_numpy(np.ones((5, 3)), dims=["x", "y"])
    r = repr(ds)
    assert "shape=(5, 3)" in r
    assert "sr=" not in r


def test_data_str() -> None:
    """Data.__str__ returns multi-line format with all metadata."""
    ds = cb.Data.from_numpy(
        np.ones((10, 5)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="group-a",
        condition="rest",
    )
    s = str(ds)
    assert "subjectID : sub-01" in s
    assert "groupID   : group-a" in s
    assert "condition : rest" in s
    assert "sr        : 100.0 Hz" in s
    assert "history   : []" in s


def test_infer_sampling_rate_no_time_dim() -> None:
    """_infer_sampling_rate returns None when no time dimension."""
    ar = xr.DataArray(np.ones((3, 2)), dims=["x", "y"])
    ds = cb.Data(ar)
    assert ds.sampling_rate is None


def test_infer_sampling_rate_single_time_point() -> None:
    """_infer_sampling_rate returns None with only one time point."""
    ar = xr.DataArray(np.ones((1, 2)), dims=["time", "space"])
    ds = cb.Data(ar)
    assert ds.sampling_rate is None


def test_eeg_construction_does_not_raise() -> None:
    """EEG.__init__ must not raise when setting ref_channel after super().__init__().

    Regression test: the parent freezes the object at the end of Data.__init__;
    EEG previously tried to assign self.ref_channel = ref_channel afterwards,
    which raised AttributeError because the object was already frozen.
    """
    arr = RNG.standard_normal((100, 4))
    ar = xr.DataArray(arr, dims=["time", "space"])
    eeg = cb.EEG(ar, sampling_rate=256.0, subjectID="sub-01")
    assert isinstance(eeg, cb.EEG)
    assert eeg.ref_channel is None


def test_eeg_ref_channel_stored() -> None:
    """EEG stores ref_channel correctly after construction."""
    arr = RNG.standard_normal((50, 3))
    ar = xr.DataArray(arr, dims=["time", "space"], coords={"space": ["Fp1", "Fp2", "Cz"]})
    eeg = cb.EEG(ar, ref_channel="average")
    assert eeg.ref_channel == "average"

    eeg2 = cb.EEG(ar, ref_channel="Cz")
    assert eeg2.ref_channel == "Cz"


def test_eeg_ref_channel_immutable() -> None:
    """EEG is still immutable after construction — ref_channel cannot be reassigned."""
    arr = RNG.standard_normal((50, 2))
    ar = xr.DataArray(arr, dims=["time", "space"])
    eeg = cb.EEG(ar)
    with pytest.raises(AttributeError, match="Cannot modify attribute"):
        eeg.ref_channel = "average"


# ---------------------------------------------------------------------------
# copy.copy / copy.deepcopy
# ---------------------------------------------------------------------------


def test_copy_data() -> None:
    """copy.copy() on a Data instance produces a distinct but equal object."""
    import copy

    arr = RNG.standard_normal((3, 4))
    d = cb.Data.from_numpy(arr, ["x", "y"], subjectID="S1", condition="rest")

    d2 = copy.copy(d)
    assert d2 is not d
    assert d2.subjectID == d.subjectID
    assert d2.condition == d.condition
    assert d2.history == d.history
    np.testing.assert_array_equal(d2.data.values, d.data.values)


def test_copy_signal_data() -> None:
    """copy.copy() on a SignalData instance works and preserves metadata."""
    import copy

    arr = RNG.standard_normal((10, 3))
    s = cb.SignalData.from_numpy(arr, ["time", "space"], sampling_rate=256.0, subjectID="S2")

    s2 = copy.copy(s)
    assert s2 is not s
    assert isinstance(s2, cb.SignalData)
    assert s2.subjectID == s.subjectID
    assert s2.sampling_rate == s.sampling_rate
    np.testing.assert_array_equal(s2.data.values, s.data.values)


def test_copy_eeg() -> None:
    """copy.copy() on an EEG instance preserves ref_channel and all metadata."""
    import copy

    arr = RNG.standard_normal((50, 2))
    ar = xr.DataArray(arr, dims=["time", "space"])
    eeg = cb.EEG(ar, ref_channel="average")

    eeg2 = copy.copy(eeg)
    assert eeg2 is not eeg
    assert isinstance(eeg2, cb.EEG)
    assert eeg2.ref_channel == eeg.ref_channel


def test_deepcopy_signal_data() -> None:
    """copy.deepcopy() on a SignalData produces an independent copy."""
    import copy

    arr = RNG.standard_normal((10, 3))
    s = cb.SignalData.from_numpy(arr, ["time", "space"], sampling_rate=128.0)

    s2 = copy.deepcopy(s)
    assert s2 is not s
    assert isinstance(s2, cb.SignalData)
    assert s2.sampling_rate == s.sampling_rate
    np.testing.assert_array_equal(s2.data.values, s.data.values)


def test_copy_preserves_immutability() -> None:
    """A copied Data instance is still immutable."""
    import copy

    arr = RNG.standard_normal((3, 4))
    d = cb.Data.from_numpy(arr, ["x", "y"])
    d2 = copy.copy(d)

    with pytest.raises(AttributeError, match="Cannot modify attribute"):
        d2.subjectID = "hacked"  # type: ignore[misc]
