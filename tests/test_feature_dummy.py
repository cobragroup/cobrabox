"""Tests for the Dummy feature behavior."""

from __future__ import annotations

import numpy as np

import cobrabox as cb


def test_dummy_feature_preserves_data_and_metadata() -> None:
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

    out = cb.feature.Dummy(mandatory_arg=1).apply(data)

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
