from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class OutwardStrength(BaseFeature[Data]):
    """Compute per-channel outward strength from a directed connectivity matrix.

    For each channel ``j``, outward strength is the mean influence *sent*
    to all other channels (excluding self-connections)::

        outward[j] = mean over i != j of mat[i, j]

    where ``mat[i, j]`` is the influence from channel ``j`` to channel ``i``
    (convention: ``space_to`` = row, ``space_from`` = column).

    Matrix-only input. To compute from a raw signal, pipe through a directed
    connectivity feature first::

        outward = (PartialDirectedCoherence() | OutwardStrength()).apply(signal)

    Args:
        freq_band: Frequency band ``(fmin, fmax)`` in Hz to average over.
            Required when the input matrix has a ``"frequency"`` dimension;
            pass ``None`` when the input is already a single matrix.
            Default ``None``.

    Returns:
        :class:`~cobrabox.Data` with dim ``("space",)``, shape ``(n_channels,)``.

    Raises:
        ValueError: If the input has a ``"time"`` dimension (expects a matrix).
        ValueError: If the input lacks ``"space_to"`` and ``"space_from"`` dims.
        ValueError: If ``freq_band`` is incompatible with the input.

    Example:
        >>> outward = cb.OutwardStrength().apply(pdc_matrix)
        >>> outward = cb.OutwardStrength(freq_band=(30.0, 80.0)).apply(pdc_matrix)
    """

    _tags: ClassVar[list[str]] = [
        "directed",
        "strength",
        "post-processing",
        "io:vector",
        "req:asymmetric-matrix",
    ]

    freq_band: tuple[float, float] | None = field(default=None)

    output_type: ClassVar[type[Data]] = Data

    def __post_init__(self) -> None:
        if self.freq_band is not None:
            fmin, fmax = self.freq_band
            if fmin >= fmax:
                raise ValueError(f"freq_band must satisfy fmin < fmax, got ({fmin}, {fmax}).")

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data
        dims = xr_data.dims

        if "time" in dims:
            raise ValueError(
                "OutwardStrength is matrix-only. Pipe a directed connectivity "
                "feature first, e.g. (PartialDirectedCoherence() | "
                "OutwardStrength()).apply(signal)."
            )

        if "space_to" not in dims or "space_from" not in dims:
            raise ValueError(
                f"Input must have 'space_to' and 'space_from' dimensions. Got dims: {dims}."
            )

        mat = xr_data

        if "frequency" in mat.dims:
            if self.freq_band is None:
                raise ValueError(
                    "The connectivity matrix has a 'frequency' dimension but "
                    "freq_band=None. Set freq_band to (fmin, fmax) to select a band."
                )
            freqs = mat.coords["frequency"].values
            fmin, fmax = self.freq_band
            if fmin > freqs.max() or fmax < freqs.min():
                raise ValueError(
                    f"freq_band ({fmin}, {fmax}) Hz is outside the available "
                    f"frequency range [{freqs.min():.4g}, {freqs.max():.4g}] Hz."
                )
            mat = mat.sel(frequency=slice(fmin, fmax)).mean("frequency")
        elif self.freq_band is not None:
            raise ValueError(
                f"freq_band={self.freq_band} is set but the connectivity matrix "
                "has no 'frequency' dimension. Pass freq_band=None to skip band averaging."
            )

        mat_vals = mat.values.astype(float).copy()
        n_ch = mat_vals.shape[0]
        np.fill_diagonal(mat_vals, np.nan)
        outward = np.nanmean(mat_vals, axis=0)

        if "space_from" in mat.coords:
            space_vals = mat.coords["space_from"].values
        else:
            space_vals = np.arange(n_ch)

        return xr.DataArray(outward, dims=["space"], coords={"space": space_vals})


@functional(OutwardStrength)
def outward_strength(data: Data, freq_band: tuple[float, float] | None = None) -> Data:
    """Compute per-channel outward strength from a directed connectivity matrix.

    For each channel ``j``, outward strength is the mean influence *sent*
    to all other channels (excluding self-connections)::

        outward[j] = mean over i != j of mat[i, j]

    where ``mat[i, j]`` is the influence from channel ``j`` to channel ``i``
    (convention: ``space_to`` = row, ``space_from`` = column).

    Matrix-only input. To compute from a raw signal, pipe through a directed
    connectivity feature first::

        outward = (PartialDirectedCoherence() | OutwardStrength()).apply(signal)

    Args:
        data: The input data to process, as a :class:`~cobrabox.Data`.
        freq_band: Frequency band ``(fmin, fmax)`` in Hz to average over.
            Required when the input matrix has a ``"frequency"`` dimension;
            pass ``None`` when the input is already a single matrix.
            Default ``None``.

    Returns:
        :class:`~cobrabox.Data` with dim ``("space",)``, shape ``(n_channels,)``.

    Raises:
        ValueError: If the input has a ``"time"`` dimension (expects a matrix).
        ValueError: If the input lacks ``"space_to"`` and ``"space_from"`` dims.
        ValueError: If ``freq_band`` is incompatible with the input.

    Example:
        >>> outward = cb.outward_strength(pdc_matrix)
        >>> outward = cb.outward_strength(pdc_matrix, freq_band=(30.0, 80.0))
    """
    return OutwardStrength(freq_band=freq_band).apply(data)
