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
    negative logarithm of the conditional probability that two sequences
    similar for *m* points remain similar when one more point is included.
    Lower values indicate more regular (predictable) signals, while higher
    values indicate greater complexity.

    By default, the binary logarithm (base 2) is used, diverging from the
    original definition which uses the natural logarithm. This can be
    configured via the ``log_base`` parameter.

    The feature works on any ``SignalData`` that contains a ``time`` dimension.
    The ``time`` dimension is collapsed and all other dimensions are preserved.

    Args:
        m: Embedding dimension (length of compared sequences). Must be >= 1.
        r: Tolerance for matching sequences. If ``None`` a default of
           ``0.2 * std(signal)`` is used, where ``std`` is the standard
           deviation of the time series.
        log_base: Base of the logarithm used in the entropy calculation.
           Defaults to 2 (binary logarithm). Use ``np.e`` for the natural
           logarithm (original definition) or 10 for base-10 logarithm.

    Returns:
        An ``xarray.DataArray`` containing the sample entropy with the
        ``time`` dimension collapsed.

    Example:
        >>> entropy = cb.feature.SampEn(m=2).apply(data)  # base-2 (default)
        >>> entropy_nat = cb.feature.SampEn(m=2, log_base=np.e).apply(data)  # natural log
    """

    # Output is a DataArray without the time dim, i.e. a ``Data`` container.
    output_type: ClassVar[type[Data] | None] = Data

    m: int = 2
    r: float | None = None
    log_base: float = 2

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError(f"Embedding dimension m must be >= 1, got {self.m}")
        if self.log_base <= 0 or self.log_base == 1:
            raise ValueError(f"Logarithm base must be > 0 and != 1, got {self.log_base}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        # Extract the raw time-series as an xarray DataArray.
        xr_data = data.data
        if "time" not in xr_data.dims:
            raise ValueError("Sample Entropy requires a 'time' dimension.")

        # Pre-compute the natural logarithm of the chosen base for the change-of-base formula.
        ln_base = np.log(self.log_base)

        # Helper that computes SampEn on a 1-D NumPy array (single time series).
        def _sampen_one(ts: np.ndarray) -> float:
            n = len(ts)
            if n <= self.m:
                raise ValueError(
                    f"Time series length ({n}) must be greater than embedding dimension m ({self.m})."
                )
            # Tolerance: use provided r or default 0.2 * std of this slice.
            r_local = 0.2 * np.std(ts, ddof=0) if self.r is None else self.r

            def _count(seq_len: int) -> int:
                cnt = 0
                max_start = n - seq_len
                for i in range(max_start):
                    template = ts[i : i + seq_len]
                    for j in range(i + 1, max_start + 1):
                        if np.max(np.abs(template - ts[j : j + seq_len])) < r_local:
                            cnt += 1
                return cnt

            matches_m = _count(self.m)
            matches_m1 = _count(self.m + 1)
            if matches_m == 0 or matches_m1 == 0:
                return np.nan
            # Change-of-base: log_b(x) = ln(x) / ln(b)
            return -np.log(matches_m1 / matches_m) / ln_base

        # Apply the helper across the ``time`` dimension, collapsing it.
        # ``vectorize=True`` broadcasts the function over all other dimensions.
        result = xr.apply_ufunc(
            _sampen_one,
            xr_data,
            input_core_dims=[["time"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Preserve original attributes (excluding the time coordinate which is gone).
        return xr.DataArray(result, attrs=data.data.attrs)
