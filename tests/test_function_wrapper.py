"""Tests for the @feature decorator in function_wrapper.py."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.function_wrapper import feature


def test_feature_decorator_raises_for_non_dataarray_return() -> None:
    """@feature raises TypeError when the wrapped function returns an unexpected type."""

    @feature
    def bad_feature(data: cb.Data) -> int:  # type: ignore[return-value]
        return 42  # type: ignore[return-value]

    ds = cb.from_numpy(np.ones((5, 2)), dims=["time", "space"])
    with pytest.raises(TypeError, match=r"must return xarray\.DataArray or Data"):
        bad_feature(ds)
