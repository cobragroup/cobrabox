"""pytest suite for the Ampent feature.

Covers:
  * uniform distribution (entropy = log2(k))
  * constant signal (entropy = 0)
  * multi-channel average entropy
  * non-integer band width sanity check
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import cobrabox as cb


def make_data(arr: np.ndarray) -> cb.Data:
    """Wrap a (rows, samples) NumPy array in cb.Data."""
    return cb.Data(xr.DataArray(arr, dims=("time", "space")))


def test_uniform_distribution() -> None:
    # 4 distinct integer values, each occurring twice → entropy = log2(4)
    arr = np.array([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=float)
    result = cb.feature.Ampent(band_width=1)(make_data(arr))

    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result.item(), np.log2(4), rtol=1e-12)


def test_constant_signal() -> None:
    arr = np.full((1, 10), fill_value=7.0)
    result = cb.feature.Ampent(band_width=0.5)(make_data(arr))

    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result.item(), 0.0, atol=1e-14)


def test_two_channels_average() -> None:
    # chan0: uniform 4-bin → entropy 2.0; chan1: constant → entropy 0.0; mean = 1.0
    chan0 = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float)
    chan1 = np.full(8, 5.0)
    arr = np.stack([chan0, chan1], axis=0)  # shape (2, 8)
    result = cb.feature.Ampent(band_width=1)(make_data(arr))

    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result.item(), 1.0, rtol=1e-12)


def test_float_bandwidth() -> None:
    rng = np.random.default_rng(123)
    arr = rng.normal(loc=0.0, scale=1.0, size=(3, 1000))
    result = cb.feature.Ampent(band_width=0.25)(make_data(arr))

    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 1)
    assert result.item() > 0.0
