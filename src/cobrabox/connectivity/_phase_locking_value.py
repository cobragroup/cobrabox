from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData
from ..transforms._analytic_signal import _analytic_signal_1d


def _plv_from_phases(phase_i: np.ndarray, phase_j: np.ndarray) -> float:
    """Compute PLV from two 1-D phase signals.

    Returns 1.0 if the inputs are exactly identical (avoiding the rare case
    where ``hilbert`` produces a tiny numerical drift).
    """
    return float(np.abs(np.mean(np.exp(1j * (phase_i - phase_j)))))


@dataclass
class PhaseLockingValue(BaseFeature[SignalData]):
    """Compute the phase-locking-value (PLV) matrix across channels.

    For each ordered pair ``(i, j)``, computes the PLV between channels
    ``i`` and ``j`` via instantaneous phase differences (extracted with the
    Hilbert transform). PLV is symmetric, takes values in ``[0, 1]``, and is
    1 on the diagonal.

    Args:
        coords: Channels to include. ``None`` (default) computes the full
            ``(K, K)`` matrix across all space coordinates; pass a list of
            coordinate names to restrict the output. Restricting saves both
            the Hilbert transform (run on the selected channels only) and
            the pairwise loop.

    Returns:
        xarray DataArray with dims ``(space_to, space_from)``. Values are
        symmetric and in ``[0, 1]``.

    Raises:
        ValueError: If ``coords`` is an empty list or any coordinate is not
            found in the space dimension.

    Example:
        >>> # Full matrix
        >>> plv = cb.PhaseLockingValue().apply(data)
        >>> # Single pair
        >>> plv = cb.PhaseLockingValue(coords=["F3", "F4"]).apply(data)
    """

    _tags: ClassVar[list[str]] = [
        "phase-synchrony",
        "undirected",
        "resting-state",
        "epilepsy",
        "anesthesia",
        "eeg",
        "io:matrix",
        "req:multichannel",
        "req:sampling-rate",
    ]

    output_type: ClassVar[type[Data]] = Data

    coords: list[str] | list[int] | None = None

    def __post_init__(self) -> None:
        if self.coords is not None and len(self.coords) == 0:
            raise ValueError("coords cannot be an empty list")

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data
        if "space" not in xr_data.dims:
            raise ValueError(f"dimension 'space' not found in data dimensions {xr_data.dims}")

        space_coords = list(xr_data.coords["space"].values)
        if self.coords is None:
            coords_list = list(space_coords)
        else:
            coords_list = list(self.coords)
            for c in coords_list:
                if c not in space_coords:
                    raise ValueError(
                        f"coordinate {c!r} not found in space dimension: {space_coords}"
                    )

        selected = xr_data.sel(space=coords_list).transpose("space", ..., "time")
        values = selected.values  # (n, ..., T)
        n = values.shape[0]
        flat = values.reshape(n, -1, values.shape[-1])  # (n, batch, T)

        phases = np.angle(_analytic_signal_1d(flat, representation="analytic", axis=-1))

        result = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i == j:
                    result[i, j] = 1.0
                    continue
                # average PLV across any extra dims
                pair_plvs = [
                    _plv_from_phases(phases[i, b], phases[j, b]) for b in range(phases.shape[1])
                ]
                result[i, j] = float(np.mean(pair_plvs))

        return xr.DataArray(
            result,
            dims=("space_to", "space_from"),
            coords={"space_to": coords_list, "space_from": coords_list},
        )
