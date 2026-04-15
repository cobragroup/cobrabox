from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numba
import numpy as np
import xarray as xr

# numba.prange is identical to range at the Python level but signals numba to
# parallelise the loop.  Type checkers don't know prange is iterable, so we
# expose range to them and the real prange to the runtime.
if TYPE_CHECKING:
    from builtins import range as prange
else:
    from numba import prange

from ...base_feature import BaseFeature
from ...data import Data, SignalData


@numba.njit(parallel=True, cache=True)
def _sampen_count_jit(ts: np.ndarray, m: int, r: float) -> tuple[int, int]:
    """Count template matches for length m and m+1 in a single O(n²) pass.

    Uses one entry per outer index (``i``) to accumulate counts safely under
    ``prange`` parallelism, then sums at the end.  Early termination on the
    inner dimension loop keeps the constant factor small when ``r`` is tight.
    """
    n = len(ts)
    mm_arr = np.zeros(n - m, dtype=np.int64)
    mm1_arr = np.zeros(n - m, dtype=np.int64)
    for i in prange(n - m):  # outer loop is parallel
        mm = np.int64(0)
        mm1 = np.int64(0)
        for j in range(i + 1, n - m + 1):
            # Chebyshev distance for the m-length window; bail as soon as r is exceeded.
            cheb = 0.0
            for k in range(m):
                d = ts[i + k] - ts[j + k]
                if d < 0.0:
                    d = -d
                if d > cheb:
                    cheb = d
                if cheb >= r:
                    break
            if cheb < r:
                mm += 1
                # matches_m1 requires j <= n-m-1 (so ts[j+m] is in bounds).
                if j < n - m:
                    d_extra = ts[i + m] - ts[j + m]
                    if d_extra < 0.0:
                        d_extra = -d_extra
                    if d_extra < r:
                        mm1 += 1
        mm_arr[i] = mm
        mm1_arr[i] = mm1
    return mm_arr.sum(), mm1_arr.sum()


@dataclass
class SampleEntropy(BaseFeature[SignalData]):
    """Sample Entropy feature.

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

    Raises:
        ValueError: If embedding dimension m is less than 1.
        ValueError: If logarithm base is invalid (<= 0 or == 1).
        ValueError: If time series length is not greater than embedding dimension m.

    Example:
        >>> entropy = cb.feature.SampleEntropy(m=2).apply(data)  # base-2 (default)
        >>> entropy_nat = cb.feature.SampleEntropy(m=2, log_base=np.e).apply(data)  # natural log
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

        # Pre-compute the natural logarithm of the chosen base for the change-of-base formula.
        ln_base = np.log(self.log_base)

        # Helper that computes SampEn on a 1-D NumPy array (single time series).
        def _sampen_one(ts: np.ndarray) -> float:
            n = len(ts)
            if n <= self.m:
                raise ValueError(
                    f"Time series length ({n}) must be greater than "
                    f"embedding dimension m ({self.m})."
                )
            # Tolerance: use provided r or default 0.2 * std of this slice.
            r_local = 0.2 * np.std(ts, ddof=0) if self.r is None else self.r

            matches_m, matches_m1 = _sampen_count_jit(
                ts.astype(np.float64, copy=False), self.m, r_local
            )
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
