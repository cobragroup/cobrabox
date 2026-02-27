"""Tests for the line_length feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb

pytestmark = pytest.mark.fast


def test_feature_line_length_expected_values_and_history() -> None:
    """line_length computes absolute temporal differences and wraps to Data."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    # Expected line length per channel:
    # ch0: |1-0| + |3-1| + |6-3| = 1 + 2 + 3 = 6
    # ch1: |3-1| + |2-3| + |2-2| = 2 + 1 + 0 = 3
    expected = np.array([[6.0, 3.0]])

    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-02")
    out = cb.feature.line_length(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("time", "space")
    assert out.data.shape == (1, 2)
    np.testing.assert_allclose(out.to_numpy(), expected)
    assert out.subjectID == "sub-02"
    assert out.sampling_rate == 200.0
    assert out.history == ["line_length"]
