from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data

_METHODS = ("minmax", "zscore", "robust", "l1", "l2", "max_abs")


@dataclass
class Normalize(BaseFeature[Data]):
    """Normalize data values along a dimension.

    Rescales values using one of several common normalization strategies.
    Each method operates independently per slice along the specified dimension.

    Args:
        method: Normalization strategy:

            - ``"minmax"`` — scale to [0, 1]: ``(x - min) / (max - min)``.
              Constant slices become all zeros.
            - ``"zscore"`` — subtract mean, divide by std.
              Constant slices become all zeros.
            - ``"robust"`` — like zscore but uses median and IQR:
              ``(x - median) / IQR``. Resilient to outliers.
              Constant slices become all zeros.
            - ``"l1"`` — divide by sum of absolute values so values sum to 1.
              All-zero slices are left as zeros.
            - ``"l2"`` — divide by Euclidean norm (unit vector).
              All-zero slices are left as zeros.
            - ``"max_abs"`` — divide by max absolute value, scales to [-1, 1].
              All-zero slices are left as zeros.

        dim: Dimension to normalize along. Each slice perpendicular to
            this dimension is normalized independently. If ``None``,
            normalizes over all values (flattened). Defaults to ``None``.

    Returns:
        :class:`~cobrabox.Data` with the same shape and dimensions as the input.

    Raises:
        ValueError: If ``method`` is not one of the six supported strategies.
        ValueError: If ``dim`` is specified but not found in the data.

    Example:
        >>> result = cb.Normalize(method="minmax").apply(data)
        >>> result = cb.Normalize(method="zscore", dim="time").apply(data)
        >>> result = cb.Normalize(method="robust", dim="space").apply(data)
    """

    _tags: ClassVar[list[str]] = ["normalization", "preprocessing", "scaling", "io:preserves-shape"]

    method: Literal["minmax", "zscore", "robust", "l1", "l2", "max_abs"]
    dim: str | None = None

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(
                f"Unknown normalization method {self.method!r}. "
                f"Supported methods: {list(_METHODS)}."
            )

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim is not None and self.dim not in data.data.dims:
            raise ValueError(f"dim {self.dim!r} not found in data dimensions {data.data.dims}")

        arr = data.data
        method = self.method

        if method == "minmax":
            lo = arr.min(dim=self.dim)
            hi = arr.max(dim=self.dim)
            span = hi - lo
            return xr.where(span > 0, (arr - lo) / span, 0.0).astype(arr.dtype)

        if method == "zscore":
            mu = arr.mean(dim=self.dim)
            sigma = arr.std(dim=self.dim)
            return xr.where(sigma > 0, (arr - mu) / sigma, 0.0).astype(arr.dtype)

        if method == "robust":
            median = arr.median(dim=self.dim)
            q75 = arr.quantile(0.75, dim=self.dim)
            q25 = arr.quantile(0.25, dim=self.dim)
            iqr = q75 - q25
            result = xr.where(iqr > 0, (arr - median) / iqr, 0.0).astype(arr.dtype)
            if "quantile" in result.coords:
                result = result.drop_vars("quantile")
            return result

        if method == "l1":
            norm = abs(arr).sum(dim=self.dim)
            return xr.where(norm > 0, arr / norm, 0.0).astype(arr.dtype)

        if method == "l2":
            norm = np.sqrt((arr**2).sum(dim=self.dim))
            return xr.where(norm > 0, arr / norm, 0.0).astype(arr.dtype)

        # max_abs
        norm = abs(arr).max(dim=self.dim)
        return xr.where(norm > 0, arr / norm, 0.0).astype(arr.dtype)


@functional(Normalize)
def normalize(
    data: Data,
    method: Literal["minmax", "zscore", "robust", "l1", "l2", "max_abs"],
    dim: str | None = None,
) -> Data:
    """Normalize data values along a dimension.

    Rescales values using one of several common normalization strategies.
    Each method operates independently per slice along the specified dimension.

    Args:
        data: The input data to process, as a :class:`~cobrabox.Data`.
        method: Normalization strategy:

            - ``"minmax"`` — scale to [0, 1]: ``(x - min) / (max - min)``.
              Constant slices become all zeros.
            - ``"zscore"`` — subtract mean, divide by std.
              Constant slices become all zeros.
            - ``"robust"`` — like zscore but uses median and IQR:
              ``(x - median) / IQR``. Resilient to outliers.
              Constant slices become all zeros.
            - ``"l1"`` — divide by sum of absolute values so values sum to 1.
              All-zero slices are left as zeros.
            - ``"l2"`` — divide by Euclidean norm (unit vector).
              All-zero slices are left as zeros.
            - ``"max_abs"`` — divide by max absolute value, scales to [-1, 1].
              All-zero slices are left as zeros.

        dim: Dimension to normalize along. Each slice perpendicular to
            this dimension is normalized independently. If ``None``,
            normalizes over all values (flattened). Defaults to ``None``.

    Returns:
        :class:`~cobrabox.Data` with the same shape and dimensions as the input.

    Raises:
        ValueError: If ``method`` is not one of the six supported strategies.
        ValueError: If ``dim`` is specified but not found in the data.

    Example:
        >>> result = cb.normalize(data, method="minmax")
        >>> result = cb.normalize(data, method="zscore", dim="time")
        >>> result = cb.normalize(data, method="robust", dim="space")
    """
    return Normalize(method=method, dim=dim).apply(data)
