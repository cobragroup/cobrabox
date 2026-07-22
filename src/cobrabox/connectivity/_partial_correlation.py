from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData


def _partial_correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Return the full partial correlation matrix given an ``(n_vars, n_samples)`` array.

    Computed via precision-matrix inversion of the full correlation matrix:
    ``rho_xy.z = -P[x, y] / sqrt(P[x, x] * P[y, y])`` for each pair.
    """
    corr = np.corrcoef(values)
    try:
        precision = np.linalg.inv(corr)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Correlation matrix is singular (non-invertible). "
            "This happens when channels are perfectly correlated or when "
            f"there are too few samples. Original error: {e}"
        ) from e
    diag = np.sqrt(np.diag(precision))
    denom = np.outer(diag, diag)
    denom = np.where(denom == 0.0, 1.0, denom)
    result = -precision / denom
    np.fill_diagonal(result, 1.0)
    return result


@dataclass
class PartialCorrelation(BaseFeature[SignalData]):
    """Compute the partial-correlation matrix across channels.

    Each off-diagonal entry ``result[i, j]`` is the correlation between
    channels ``i`` and ``j`` after partialling out the influence of every
    other channel in the matrix (and any explicit ``control_vars``). Computed
    via inversion of the full correlation matrix — conditioning therefore
    requires all channels, so ``coords`` only filters the *output*; the
    precision-matrix inversion is still O(K³) in the full channel count.

    Args:
        coords: Channels to include in the output. ``None`` (default) returns
            the full ``(K, K)`` matrix; pass a list of coordinate names to
            restrict the output (compute cost is unchanged).
        control_vars: Additional coordinates to control for that are not part
            of the output matrix. When ``None`` and ``coords`` is set, no
            extra controls are added (only the channels in ``coords`` enter
            the precision matrix).

    Returns:
        xarray DataArray with dims ``(space_to, space_from)`` and shape
        ``(len(coords), len(coords))``. Symmetric, with ``1`` on the diagonal.

    Raises:
        ValueError: If the correlation matrix is singular (perfect collinearity
            or too few samples), if ``coords`` is an empty list, or if any
            named coordinate is missing from the space dimension.

    Example:
        >>> # Full matrix
        >>> pc = cb.feature.PartialCorrelation().apply(data)
        >>> # Two-channel subset, control on an extra channel
        >>> pc = cb.feature.PartialCorrelation(
        ...     coords=["F3", "F4"], control_vars=["Cz"]
        ... ).apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "conditional-independence",
        "precision-matrix",
        "undirected",
        "resting-state",
        "eeg",
        "fmri",
        "io:matrix",
        "req:multichannel",
    ]

    output_type: ClassVar[type[Data]] = Data

    coords: list[str] | list[int] | None = None
    control_vars: list[str] | list[int] | None = None

    def __post_init__(self) -> None:
        if self.coords is not None and len(self.coords) == 0:
            raise ValueError("coords cannot be an empty list")

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        if "space" not in xr_data.dims:
            raise ValueError(f"dimension 'space' not found in data dimensions {xr_data.dims}")
        if "time" not in xr_data.dims:
            raise ValueError("data must have 'time' dimension")

        space_coords = list(xr_data.coords["space"].values)
        output_coords = list(self.coords) if self.coords is not None else list(space_coords)
        control_list = list(self.control_vars) if self.control_vars is not None else []

        for c in output_coords + control_list:
            if c not in space_coords:
                raise ValueError(f"coordinate {c!r} not found in space dimension: {space_coords}")

        # Channels entering the precision matrix: output_coords plus control_vars,
        # de-duplicated, preserving the user's order.
        seen: set = set()
        all_coords: list = []
        for c in [*output_coords, *control_list]:
            if c not in seen:
                seen.add(c)
                all_coords.append(c)

        selected = xr_data.sel(space=all_coords).transpose("space", "time")
        full_pc = _partial_correlation_matrix(selected.values)

        # Restrict to the requested output coordinates
        idx = [all_coords.index(c) for c in output_coords]
        result = full_pc[np.ix_(idx, idx)]

        return xr.DataArray(
            result,
            dims=("space_to", "space_from"),
            coords={"space_to": output_coords, "space_from": output_coords},
        )
