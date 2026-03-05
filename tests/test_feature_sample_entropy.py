import numpy as np
import pytest

from cobrabox import SignalData
from cobrabox.features.sample_entropy import SampEn


def _naive_sampen(ts: np.ndarray, m: int, r: float, base: float = 2) -> float:
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
    return -np.log(a / b) / np.log(base)


def test_sampen_known_value():
    """Validate SampEn against a manually computed reference (base 2 default)."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    m = 2
    r = 0.5  # tolerance smaller than distance between distinct values
    expected = _naive_sampen(ts, m, r, base=2)

    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)
    result = SampEn(m=m, r=r)(data)

    assert np.isfinite(result.item())
    assert np.allclose(result.item(), expected, atol=1e-12)


def test_sampen_natural_log():
    """Validate SampEn with natural logarithm (base=e)."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    m = 2
    r = 0.5
    expected = _naive_sampen(ts, m, r, base=np.e)

    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)
    result = SampEn(m=m, r=r, base=np.e)(data)

    assert np.isfinite(result.item())
    assert np.allclose(result.item(), expected, atol=1e-12)


def test_sampen_base_10():
    """Validate SampEn with base-10 logarithm."""
    ts = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    m = 2
    r = 0.5
    expected = _naive_sampen(ts, m, r, base=10)

    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)
    result = SampEn(m=m, r=r, base=10)(data)

    assert np.isfinite(result.item())
    assert np.allclose(result.item(), expected, atol=1e-12)


def test_sampen_raises_on_short_series():
    """SampEn requires a series longer than the embedding dimension."""
    ts = np.arange(2, dtype=float)  # length == 2
    data = SignalData.from_numpy(ts, dims=["time"], sampling_rate=1.0)

    with pytest.raises(ValueError, match="must be greater than embedding dimension"):
        SampEn(m=2)(data)


def test_sampen_raises_on_invalid_base():
    """SampEn should reject invalid logarithm bases."""
    with pytest.raises(ValueError, match="Logarithm base"):
        SampEn(m=2, base=0)

    with pytest.raises(ValueError, match="Logarithm base"):
        SampEn(m=2, base=1)

    with pytest.raises(ValueError, match="Logarithm base"):
        SampEn(m=2, base=-2)


def test_sampen_multi_dim_preserves_other_dims():
    """Sample entropy should be computed per non-time dimension and time collapsed."""
    # Two channels: one constant, one alternating pattern.
    const = np.zeros(12)
    alt = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    arr = np.stack([const, alt])  # shape (2, 12)
    data = SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=1.0)

    feat = SampEn(m=2, r=0.5)
    result = feat(data)

    # Result should have only the 'space' dimension.
    assert set(result.dims) == {"space"}

    # Both channels should return finite values with r=0.5.
    # Constant channel: all templates identical, so all match -> finite entropy.
    expected_const = _naive_sampen(const, m=2, r=0.5, base=2)
    assert np.allclose(result.sel(space=0).item(), expected_const, atol=1e-12)

    # Alternating channel should match the known scalar value.
    expected_alt = _naive_sampen(alt, m=2, r=0.5, base=2)
    assert np.allclose(result.sel(space=1).item(), expected_alt, atol=1e-12)
