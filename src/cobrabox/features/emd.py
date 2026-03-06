from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import emd as emd_pkg
import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import SignalData

_SIFT_METHODS = ("sift", "mask_sift", "iterated_mask_sift")


@dataclass
class EMD(BaseFeature[SignalData]):
    """Decompose a signal into Intrinsic Mode Functions (IMFs) using Empirical Mode Decomposition.

    Applies EMD along the ``time`` axis and stacks the resulting IMFs along
    a new ``imf`` dimension with coordinates ``imf0``, ``imf1``, etc.
    The last component is always the residual (labelled ``residual``).

    The IMFs sum to the original signal: ``result.sum(dim='imf') == original``.

    When processing multi-channel data, different channels may produce different
    numbers of IMFs. Missing IMFs are filled with ``NaN`` (not zero) so that
    operations like ``mean(dim='imf')`` give correct results without being
    diluted by fake zeros.

    Args:
        max_imfs: Maximum number of IMFs to compute (not including residual).
            If ``None``, the algorithm determines the number adaptively.
            Defaults to ``None``.
        method: Sift method to use. One of:

            * ``'sift'`` — standard EMD (default)
            * ``'mask_sift'`` — mask sift for improved mode separation
            * ``'iterated_mask_sift'`` — iterated mask sift

        keep_orig: Whether to keep the original signal as an IMF named ``original``.
            Defaults to ``False``.

    Raises:
        ValueError: If ``method`` is not a valid sift method.
        ValueError: If ``max_imfs`` is not positive.

    Returns:
        xarray.DataArray: The IMFs stacked along a new ``imf`` dimension.
            The ``imf`` coordinate contains labels ``imf0``, ``imf1``, ..., ``residual``
            (and ``original`` if ``keep_orig=True``).
            Shape is the same as input with an additional ``imf`` dimension.

    Example:
        >>> result = cb.feature.EMD().apply(data)
        >>> result = cb.feature.EMD(max_imfs=5, method="mask_sift").apply(data)
        >>> result = cb.feature.EMD(keep_orig=True).apply(data)
    """

    max_imfs: int | None = None
    method: Literal["sift", "mask_sift", "iterated_mask_sift"] = "sift"
    keep_orig: bool = False

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.method not in _SIFT_METHODS:
            raise ValueError(
                f"Invalid method {self.method!r}. "
                f"Must be one of: {', '.join(repr(m) for m in _SIFT_METHODS)}."
            )
        if self.max_imfs is not None and self.max_imfs <= 0:
            raise ValueError(f"max_imfs must be positive, got {self.max_imfs}")

    def __call__(self, data: SignalData) -> xr.DataArray:
        sift_func = getattr(emd_pkg.sift, self.method)
        sift_kwargs: dict[str, object] = {}
        if self.max_imfs is not None:
            sift_kwargs["max_imfs"] = self.max_imfs

        def _apply_emd(x: xr.DataArray) -> xr.DataArray:
            """Apply EMD to a 1D time-series and return DataArray with imf dimension."""
            imfs = sift_func(x.values, **sift_kwargs)
            # Check if IMFs sum to original — if not, compute residual explicitly
            imf_sum = imfs.sum(axis=1)
            if not np.allclose(imf_sum, x.values):
                residual = x.values - imf_sum
                imfs = np.column_stack([imfs, residual])

            # Build labels: optionally include original, then imf0..imfN, then residual
            n_imfs = imfs.shape[1]
            imf_labels = [f"imf{i}" for i in range(n_imfs - 1)] + ["residual"]

            if self.keep_orig:
                # Prepend original signal
                imfs = np.column_stack([x.values, imfs])
                imf_labels = ["original", *imf_labels]

            return xr.DataArray(
                imfs, dims=["time", "imf"], coords={"time": x.coords["time"], "imf": imf_labels}
            )

        # Apply EMD along time axis, vectorizing over all other dimensions
        xr_data = data.data
        non_time_dims = [d for d in xr_data.dims if d != "time"]

        if not non_time_dims:
            # Single 1D time-series
            return _apply_emd(xr_data)

        # Stack all non-time dimensions into a single 'stacked' dimension
        stacked = xr_data.stack(stacked=non_time_dims)

        # Apply EMD to each 1D slice
        imf_arrays = []
        for i in range(stacked.sizes["stacked"]):
            slice_1d = stacked.isel(stacked=i)
            imf_result = _apply_emd(slice_1d)
            imf_arrays.append(imf_result)

        # Combine results: each has dims (time, imf), we need to concat along a new dim
        # then unstack to recover original non-time dims
        # Use join='outer' to handle varying IMF counts, filling missing IMFs with NaN
        # NaN is preferred over 0.0 because it correctly indicates "no IMF here" and
        # operations like mean() will give correct results (not diluted by fake zeros)
        combined = xr.concat(imf_arrays, dim="stacked", join="outer", fill_value=np.nan)
        combined = combined.assign_coords(stacked=stacked.coords["stacked"])
        return combined.unstack("stacked")
