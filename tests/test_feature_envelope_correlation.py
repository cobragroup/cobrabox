"""Tests for the envelope_correlation feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _make_data(
    n_time: int = 512,
    n_space: int = 4,
    *,
    sampling_rate: float = 256.0,
    subjectID: str = "sub-01",
    seed: int = 0,
) -> cb.Data:
    """Helper: random broadband data."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_time, n_space))
    return cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=sampling_rate, subjectID=subjectID
    )


# ---------------------------------------------------------------------------
# Dims, shape and coordinates
# ---------------------------------------------------------------------------


def test_envelope_correlation_output_dims_and_shape() -> None:
    """EnvelopeCorrelation returns (space, space_to) Data."""
    data = _make_data()
    out = cb.feature.EnvelopeCorrelation().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "space_to")
    assert out.data.shape == (4, 4)


def test_envelope_correlation_space_coords_preserved() -> None:
    """space and space_to carry the original channel coordinates."""
    import xarray as xr

    arr_xr = xr.DataArray(
        np.random.default_rng(1).standard_normal((512, 3)),
        dims=["time", "space"],
        coords={"space": ["Fz", "Cz", "Pz"], "time": np.arange(512) / 256.0},
    )
    data = cb.from_xarray(arr_xr)
    out = cb.feature.EnvelopeCorrelation().apply(data)

    np.testing.assert_array_equal(out.data.coords["space"].values, ["Fz", "Cz", "Pz"])
    np.testing.assert_array_equal(out.data.coords["space_to"].values, ["Fz", "Cz", "Pz"])


# ---------------------------------------------------------------------------
# Numerical properties
# ---------------------------------------------------------------------------


def test_envelope_correlation_matrix_is_symmetric() -> None:
    """AEC matrix must be symmetric: C[i, j] == C[j, i]."""
    data = _make_data()
    out = cb.feature.EnvelopeCorrelation().apply(data)

    mat = out.data.values
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


def test_envelope_correlation_diagonal_is_finite() -> None:
    """Diagonal entries (self-correlation) must be finite values."""
    data = _make_data()
    out = cb.feature.EnvelopeCorrelation().apply(data)

    mat = out.data.values
    assert np.all(np.isfinite(np.diag(mat)))


def test_envelope_correlation_absolute_true_gives_nonnegative() -> None:
    """absolute=True must produce non-negative off-diagonal values."""
    data = _make_data()
    out = cb.feature.EnvelopeCorrelation(absolute=True).apply(data)

    mat = out.data.values
    np.testing.assert_array_less(-1e-12, mat)


def test_envelope_correlation_values_in_valid_range() -> None:
    """All correlation values lie in [-1, 1]."""
    data = _make_data()
    out = cb.feature.EnvelopeCorrelation().apply(data)

    mat = out.data.values
    assert np.all(mat >= -1.0 - 1e-10)
    assert np.all(mat <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_envelope_correlation_orthogonalize_false_same_shape() -> None:
    """orthogonalize=False produces the same output shape."""
    data = _make_data()
    out = cb.feature.EnvelopeCorrelation(orthogonalize=False).apply(data)

    assert out.data.shape == (4, 4)
    assert out.data.dims == ("space", "space_to")


def test_envelope_correlation_orthogonalize_changes_values() -> None:
    """orthogonalize='pairwise' and False produce different correlation values."""
    data = _make_data()
    out_orth = cb.feature.EnvelopeCorrelation(orthogonalize="pairwise").apply(data)
    out_none = cb.feature.EnvelopeCorrelation(orthogonalize=False).apply(data)

    assert not np.allclose(out_orth.data.values, out_none.data.values)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_envelope_correlation_history_appended() -> None:
    """'EnvelopeCorrelation' is the last entry in history."""
    data = _make_data()
    out = cb.feature.EnvelopeCorrelation().apply(data)

    assert out.history[-1] == "EnvelopeCorrelation"


def test_envelope_correlation_metadata_preserved() -> None:
    """subjectID and sampling_rate are carried through."""
    data = _make_data(subjectID="sub-99")
    out = cb.feature.EnvelopeCorrelation().apply(data)

    assert out.subjectID == "sub-99"
    assert out.sampling_rate == 256.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_envelope_correlation_raises_for_single_channel() -> None:
    """ValueError raised when data has fewer than 2 spatial channels."""
    data = _make_data(n_space=1)

    with pytest.raises(ValueError, match="at least 2 spatial channels"):
        cb.feature.EnvelopeCorrelation().apply(data)


def test_envelope_correlation_raises_for_extra_dims() -> None:
    """ValueError raised when data has dimensions beyond space and time."""
    import xarray as xr

    arr_xr = xr.DataArray(
        np.random.default_rng(2).standard_normal((3, 512, 4)),
        dims=["run_index", "time", "space"],
        coords={
            "run_index": np.arange(3),
            "time": np.arange(512) / 256.0,
            "space": [f"ch{k}" for k in range(4)],
        },
    )
    data = cb.from_xarray(arr_xr)

    with pytest.raises(ValueError, match="extra dims"):
        cb.feature.EnvelopeCorrelation().apply(data)
