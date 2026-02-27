"""Tests for the sliding_window feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb

pytestmark = pytest.mark.fast


def test_feature_sliding_window_shapes_values_and_metadata() -> None:
    """sliding_window creates expected windows and preserves metadata/history."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
    )

    out = cb.feature.sliding_window(data, window_size=4, step_size=2)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("window_index", "time", "space")
    assert out.data.shape == (4, 4, 2)
    np.testing.assert_allclose(out.data.isel(window_index=0).values, arr[0:4, :])
    np.testing.assert_allclose(out.data.isel(window_index=1).values, arr[2:6, :])

    assert out.subjectID == "sub-01"
    assert out.groupID == "patient"
    assert out.condition == "rest"
    assert out.sampling_rate == 100.0
    assert out.history == ["sliding_window"]
