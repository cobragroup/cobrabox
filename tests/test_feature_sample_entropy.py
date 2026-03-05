"""Tests for cb.feature.SampEn."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb
from cobrabox import SignalData
from cobrabox.features.sample_entropy import SampEn


def _naive_sampen(ts: np.ndarray, m: int, r: float, log_base: float = 2) -> float:
    """Reference implementation matching the algorithm in ``SampEn``."""
    n = len(ts)

    def _count(seq_len: int) -> int:
        cnt = 0
        for i in range(n - seq_len):
            tmpl = ts[i : i + seq_len]
            for j in range(i + 1, n - seq_len + 1):
                if np.max(np.abs(tmpl - ts[j : j + seq_len])) < r:
                    cnt += 1
        return cnt

    b = _count(m)
    a = _count(m + 1)

    if b == 0 or a == 0:
        return np.nan
    # Use change-of-base: log_b(x) = ln(x) / ln(b)
    return -np.log(a / b) / np.log(log_base)


def test_sampen_known_value() -> None:
    """Validate SampEn against a manually computed reference (log_base 2 default)."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    m = 2
    r = 0.5  # tolerance smaller than distance between distinct values
    expected = _naive_sampen(ts, m, r, log_base=2)

    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)
    result = SampEn(m=m, r=r).apply(data)

    val = result.data.values.item()
    assert np.isfinite(val)
    assert np.allclose(val, expected, atol=1e-12)


def test_sampen_natural_log() -> None:
    """Validate SampEn with natural logarithm (log_base=e)."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    m = 2
    r = 0.5
    expected = _naive_sampen(ts, m, r, log_base=np.e)

    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)
    result = SampEn(m=m, r=r, log_base=np.e).apply(data)

    val = result.data.values.item()
    assert np.isfinite(val)
    assert np.allclose(val, expected, atol=1e-12)


def test_sampen_base_10() -> None:
    """Validate SampEn with base-10 logarithm."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    m = 2
    r = 0.5
    expected = _naive_sampen(ts, m, r, log_base=10)

    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)
    result = SampEn(m=m, r=r, log_base=10).apply(data)

    val = result.data.values.item()
    assert np.isfinite(val)
    assert np.allclose(val, expected, atol=1e-12)


def test_sampen_raises_on_short_series() -> None:
    """SampEn requires a series longer than the embedding dimension."""
    ts = np.arange(2, dtype=float)  # length == 2
    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)

    with pytest.raises(ValueError, match="must be greater than embedding dimension"):
        SampEn(m=2).apply(data)


def test_sampen_raises_on_invalid_log_base() -> None:
    """SampEn should reject invalid logarithm bases."""
    with pytest.raises(ValueError, match="Logarithm base"):
        SampEn(m=2, log_base=0)

    with pytest.raises(ValueError, match="Logarithm base"):
        SampEn(m=2, log_base=1)

    with pytest.raises(ValueError, match="Logarithm base"):
        SampEn(m=2, log_base=-2)


def test_sampen_multi_dim_preserves_other_dims() -> None:
    """Sample entropy should be computed per non-time dimension and time collapsed."""
    # Two channels: one constant, one alternating pattern.
    const = np.zeros(12)
    alt = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    arr = np.stack([const, alt])  # shape (2, 12)
    data = SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=1.0)

    feat = SampEn(m=2, r=0.5)
    result = feat.apply(data)

    # Result should have only the 'space' dimension.
    assert set(result.data.dims) == {"space"}

    # Both channels should return finite values with r=0.5.
    # Constant channel: all templates identical, so all match -> finite entropy.
    expected_const = _naive_sampen(const, m=2, r=0.5, log_base=2)
    assert np.allclose(result.data.sel(space=0).item(), expected_const, atol=1e-12)

    # Alternating channel should match the known scalar value.
    expected_alt = _naive_sampen(alt, m=2, r=0.5, log_base=2)
    assert np.allclose(result.data.sel(space=1).item(), expected_alt, atol=1e-12)


def test_sampen_invalid_m() -> None:
    """SampEn raises ValueError for m < 1."""
    with pytest.raises(ValueError, match="Embedding dimension m must be >= 1"):
        SampEn(m=0)
    with pytest.raises(ValueError, match="Embedding dimension m must be >= 1"):
        SampEn(m=-1)


def test_sampen_no_matches_returns_nan() -> None:
    """SampEn returns NaN when no template matches are found."""
    # Create data with unique values where no templates will match
    # Each template will be distinct enough that with a small tolerance,
    # no two templates will be considered similar
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)

    # Use a tolerance so small that no matches will be found
    # Templates are [0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0] for m=2
    # Max distance between any two is at least 0.5, so r=0.001 should result in no matches
    result = SampEn(m=2, r=0.001).apply(data)

    assert np.isnan(result.data.values.item())


def test_sampen_history_updated() -> None:
    """SampEn appends 'SampEn' to history."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)

    result = SampEn(m=2, r=0.5).apply(data)

    assert result.history[-1] == "SampEn"


def test_sampen_metadata_preserved() -> None:
    """SampEn preserves subjectID, groupID, condition."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    data = SignalData.from_numpy(
        ts, dims=["time"], sampling_rate=1.0, subjectID="s42", groupID="control", condition="task"
    )

    result = SampEn(m=2, r=0.5).apply(data)

    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    # sampling_rate should be None since time dimension is removed
    assert result.sampling_rate is None


def test_sampen_does_not_mutate_input() -> None:
    """SampEn.apply() leaves the input Data object unchanged."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.data.values.copy()

    _ = SampEn(m=2, r=0.5).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.data.values, original_values)


def test_sampen_returns_data_instance() -> None:
    """SampEn.apply() always returns a Data instance."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)

    result = SampEn(m=2, r=0.5).apply(data)

    assert isinstance(result, cb.Data)


def test_sampen_default_r_uses_std() -> None:
    """SampEn uses 0.2 * std as default tolerance when r=None."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)

    # Default r (None) should compute 0.2 * std
    result_default = SampEn(m=2).apply(data)

    # Explicit r = 0.2 * std should match
    explicit_r = float(0.2 * np.std(ts, ddof=0))
    result_explicit = SampEn(m=2, r=explicit_r).apply(data)

    val_default = result_default.data.values.item()
    val_explicit = result_explicit.data.values.item()
    assert np.allclose(val_default, val_explicit, atol=1e-12)
