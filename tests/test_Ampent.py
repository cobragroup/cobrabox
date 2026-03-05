# -------------------------------------------------
#  tests/test_ampent_real.py
# -------------------------------------------------
"""pytest suite for the Ampent feature using the real cobrabox package.

The tests cover:
  * a uniform distribution case (entropy = log2(k))
  * a constant value case (entropy = 0)
  * a multi channel average entropy case
  * a non integer band width sanity check
"""
from __future__ import annotations  # Fixed asterisks to underscores

import numpy as np
import xarray as xr

# -------------------------------------------------
#  Imports from the _actual_ cobrabox distribution
# -------------------------------------------------
import cobrabox as cb  # the real cobrabox package
from cobrabox.function_wrapper import feature  # the real decorator


# ----------------------------------------------------------------------
# Helper: convert a plain NumPy array → cobrabox.Data (expects dims time, space)
# ----------------------------------------------------------------------
def make_cobrabox_data(arr: np.ndarray) -> cb.Data:  # Fixed syntax and NameError
    """Wrap a (time, space) NumPy array in the cobrabox.Data class."""
    xr_arr = xr.DataArray(arr, dims=("time", "space"))
    return cb.Data(xr_arr)  # Added cb. prefix

# ... (The rest of your test functions can remain exactly as they are) ...


# ----------------------------------------------------------------------
#  Test 1 Uniform distribution → known entropy = log2(k)
# ----------------------------------------------------------------------
def test_uniform_distribution() -> None:
    # 4 distinct integer values, each occurring twice → k = 4
    data = np.array([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=float)
    band_width = 1

    result = cb.feature.Ampent(make_cobrabox_data(data), band_width)

    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 1)  # (time, space) = (1,1)

    expected = np.log2(4)  # entropy of a uniform 4‑bin histogram
    np.testing.assert_allclose(result.item(), expected, rtol=1e-12)


# ----------------------------------------------------------------------
#  Test 2 Constant signal → entropy = 0
# ----------------------------------------------------------------------
def test_constant_signal() -> None:
    data = np.full((1, 10), fill_value=7.0)  # ten identical samples
    band_width = 0.5

    result = cb.feature.Ampent(make_cobrabox_data(data), band_width)

    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result.item(), 0.0, atol=1e-14)


# ----------------------------------------------------------------------
#  Test 3 Two channels, average entropy = (2 + 0) / 2 = 1
# ----------------------------------------------------------------------
def test_two_channels_average() -> None:
    chan0 = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float)
    chan1 = np.full(8, 5.0)  # constant channel
    data = np.stack([chan0, chan1], axis=0)  # shape (2, 8)

    band_width = 1

    result = cb.feature.Ampent(make_cobrabox_data(data), band_width)

    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 1)

    expected = 1.0  # (2.0 + 0.0) / 2
    np.testing.assert_allclose(result.item(), expected, rtol=1e-12)


# ----------------------------------------------------------------------
#  Test 4 Noninteger band width (the function must not crash)
# ----------------------------------------------------------------------
def test_float_bandwidth() -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(loc=0.0, scale=1.0, size=(3, 1000))

    band_width = 0.25  # float value

    result = cb.feature.Ampent(make_cobrabox_data(data), band_width)

    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 1)

    # Entropy of a realvalued Gaussian is finite and > 0.
    assert result.item() > 0.0
