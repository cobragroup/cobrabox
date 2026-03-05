from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class SampEn(BaseFeature[SignalData]):
    """Sample Entropy (SampEn) feature.

    Sample entropy quantifies the regularity of a time-series. It is the
    negative natural logarithm of the conditional probability that two
    sequences similar for *m* points remain similar when one more point
    is included. Lower values indicate more regular (predictable)
    signals, while higher values indicate greater complexity.

    The implementation follows the classic definition (Richman & Moorman,
    2000) and works on any ``SignalData`` that contains a ``time`` dimension.
    The result is a **scalar** ``xarray.DataArray`` (the ``time`` dimension is
    removed). The feature can be used directly or as part of a pipeline.

    Args:
        m: Embedding dimension (length of compared sequences). Must be >= 1.
        r: Tolerance for matching sequences. If ``None`` a default of
           ``0.2 * std(signal)`` is used, where ``std`` is the standard
           deviation of the time series.

    Returns:
        A scalar ``xarray.DataArray`` containing the sample entropy.

    Example:
        >>> entropy = cb.feature.SampEn(m=2).apply(data)
    """

    # Output is a scalar DataArray, i.e. a ``Data`` container without a time dim.
    output_type: ClassVar[type[Data] | None] = Data

    m: int = 2
    r: float | None = None

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError(f"Embedding dimension m must be >= 1, got {self.m}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        # Extract the raw time-series as a 1-D NumPy array.
        xr_data = data.data
        if "time" not in xr_data.dims:
            raise ValueError("Sample Entropy requires a 'time' dimension.")
        # Flatten all non-time dimensions and compute entropy on the concatenated series.
        ts: np.ndarray = xr_data.values.astype(float).reshape(-1)

        n = len(ts)
        if n <= self.m:
            raise ValueError(
                f"Time series length ({n}) must be greater than embedding dimension m ({self.m})."
            )

        # Default tolerance as a fraction of the standard deviation.
        r = 0.2 * np.std(ts, ddof=0) if self.r is None else self.r

        def _count_matches(seq_len: int) -> int:
            """Count the number of template pairs of length ``seq_len`` within tolerance."""
            count = 0
            max_start = n - seq_len
            for i in range(max_start):
                template = ts[i : i + seq_len]
                for j in range(i + 1, max_start + 1):
                    if np.max(np.abs(template - ts[j : j + seq_len])) < r:
                        count += 1
            return count

        matches_m = _count_matches(self.m)
        matches_m1 = _count_matches(self.m + 1)

        if matches_m == 0 or matches_m1 == 0:
            sampen_val = np.nan
        else:
            sampen_val = -np.log(matches_m1 / matches_m)

        # Return as a scalar DataArray (no dimensions) preserving original attrs.
        return xr.DataArray(sampen_val, attrs=data.data.attrs)
