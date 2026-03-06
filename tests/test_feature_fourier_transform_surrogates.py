"""Tests for the FourierTransformSurrogates splitter feature."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.features.fourier_transform_surrogates import FourierTransformSurrogates


def test_surrogate_shape_and_dims_preserved_2D(rng: np.random.Generator) -> None:
    """Surrogate preserves 2D shape and dimensions while altering values."""
    # create a simple time x space signal
    arr = rng.standard_normal((50, 2))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    feat = FourierTransformSurrogates(n_surrogates=1, random_state=42, return_data=False)
    out = next(feat(data))

    assert isinstance(out, cb.SignalData)
    assert out.data.shape == data.data.shape
    assert out.data.dims == data.data.dims
    # surrogates should differ from the original
    assert not np.allclose(out.to_numpy(), data.to_numpy())


def test_surrogate_shape_and_dims_preserved_4D(rng: np.random.Generator) -> None:
    """Surrogate preserves 4D shape and dimensions."""
    # create a multi-dimensional time x space x ... signal
    arr = rng.standard_normal((50, 2, 3, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "chan", "x", "y"], sampling_rate=100.0)

    feat = FourierTransformSurrogates(n_surrogates=1, random_state=42, return_data=False)
    out = next(feat(data))

    assert isinstance(out, cb.SignalData)
    assert out.data.shape == data.data.shape
    assert out.data.dims == data.data.dims
    # surrogates should differ from the original
    assert not np.allclose(out.to_numpy(), data.to_numpy())


def test_return_data_flag_controls_original_yield(rng: np.random.Generator) -> None:
    """return_data flag controls whether original data is yielded first."""
    arr = rng.standard_normal((10, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"])

    feat_true = FourierTransformSurrogates(n_surrogates=2, return_data=True, random_state=0)
    seq_true = list(feat_true(data))
    assert len(seq_true) == 3  # original + 2 surrogates
    assert np.array_equal(seq_true[0].to_numpy(), data.to_numpy())

    feat_false = FourierTransformSurrogates(n_surrogates=2, return_data=False, random_state=0)
    seq_false = list(feat_false(data))
    assert len(seq_false) == 2
    assert not np.array_equal(seq_false[0].to_numpy(), data.to_numpy())


def test_n_surrogates_zero_behaviour(rng: np.random.Generator) -> None:
    """Zero surrogates yields only original data or empty list."""
    arr = rng.standard_normal((5, 1))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"])

    feat = FourierTransformSurrogates(n_surrogates=0, return_data=True)
    assert list(feat(data)) == [data]

    feat2 = FourierTransformSurrogates(n_surrogates=0, return_data=False)
    assert list(feat2(data)) == []


def test_random_state_reproducibility(rng: np.random.Generator) -> None:
    """Same random_state produces identical surrogates across different instances."""
    arr = rng.standard_normal((20, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"])

    # Two separate instances with same seed should produce identical first surrogates
    feat1 = FourierTransformSurrogates(n_surrogates=1, random_state=123, return_data=False)
    feat2 = FourierTransformSurrogates(n_surrogates=1, random_state=123, return_data=False)

    first = next(feat1(data)).to_numpy()
    second = next(feat2(data)).to_numpy()
    assert np.array_equal(first, second)


def test_multivariate_flag_changes_channel_relations(rng: np.random.Generator) -> None:
    """multivariate=True preserves correlations, multivariate=False decorrelates."""
    r = 0.6
    D = 6
    L = 10000
    S = np.ones([D, D]) * r
    S[np.diag_indices(D)] = 1
    arr = rng.multivariate_normal(np.zeros(D), S, size=L).T
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"])
    data_corr = np.corrcoef(arr)
    data_rho = np.mean(data_corr[np.triu_indices(D, 1)])
    feat_multi = FourierTransformSurrogates(
        n_surrogates=1, random_state=0, multivariate=True, return_data=False
    )
    out_multi = next(feat_multi(data)).to_numpy()
    multi_corr = np.corrcoef(out_multi)
    multi_rho = np.mean(multi_corr[np.triu_indices(D, 1)])
    # the correlation should be similar when multivariate=True
    assert np.abs(multi_rho - data_rho) / data_rho < 0.05

    feat_ind = FourierTransformSurrogates(
        n_surrogates=1, random_state=0, multivariate=False, return_data=False
    )
    out_ind = next(feat_ind(data)).to_numpy()
    ind_corr = np.corrcoef(out_ind)
    ind_rho = np.mean(ind_corr[np.triu_indices(D, 1)])
    # the correlation should be close to zero when multivariate=False
    assert np.abs(ind_rho) < 0.05


def test_identical_channels_remain_identical_multivariate(rng: np.random.Generator) -> None:
    """Identical channels remain identical with multivariate=True."""
    # Create data with identical channels - channels are rows in space x time layout
    arr = rng.standard_normal((3, 100))
    arr[1, :] = arr[0, :]  # Make second channel identical to first
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"])

    feat = FourierTransformSurrogates(
        n_surrogates=1, random_state=42, multivariate=True, return_data=False
    )
    out = next(feat(data)).to_numpy()

    # Identical channels should remain identical after surrogating
    assert np.allclose(out[0, :], out[1, :])
    # Non-identical channels should differ
    assert not np.allclose(out[0, :], out[2, :])


def test_fourier_transform_surrogates_metadata_preserved(rng: np.random.Generator) -> None:
    """Surrogate preserves metadata (subjectID, groupID, condition, sampling_rate)."""
    arr = rng.standard_normal((50, 3))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s42",
        groupID="control",
        condition="rest",
    )
    feat = FourierTransformSurrogates(n_surrogates=1, random_state=42, return_data=False)
    out = next(feat(data))

    assert out.subjectID == "s42"
    assert out.groupID == "control"
    assert out.condition == "rest"
    assert out.sampling_rate == pytest.approx(100.0)


def test_fourier_transform_surrogates_does_not_mutate_input(rng: np.random.Generator) -> None:
    """Feature does not modify the input Data object."""
    arr = rng.standard_normal((20, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"])
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    feat = FourierTransformSurrogates(n_surrogates=1, random_state=42, return_data=False)
    _ = next(feat(data))

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_fourier_transform_surrogates_invalid_n_surrogates() -> None:
    """Negative n_surrogates raises ValueError."""
    with pytest.raises(ValueError, match="negative"):
        FourierTransformSurrogates(n_surrogates=-1, random_state=42)


def test_fourier_transform_surrogates_is_lazy_generator(rng: np.random.Generator) -> None:
    """Feature returns a lazy generator, not a materialized list."""
    arr = rng.standard_normal((10, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"])
    feat = FourierTransformSurrogates(n_surrogates=100, random_state=42, return_data=False)
    gen = feat(data)

    assert inspect.isgenerator(gen)
