from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr

from .._functional import functional
from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class InwardStrength(BaseFeature[Data]):
    """Compute per-channel inward strength from a directed connectivity matrix.

    For each channel ``i``, inward strength is the mean influence *received*
    from all other channels (excluding self-connections)::

        inward[i] = mean over j != i of mat[i, j]

    where ``mat[i, j]`` is the influence from channel ``j`` to channel ``i``
    (convention: ``space_to`` = row, ``space_from`` = column).

    Matrix-only input. To compute from a raw signal, pipe through a directed
    connectivity feature first::

        inward = (PartialDirectedCoherence() | InwardStrength()).apply(signal)

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
        >>> inward = cb.InwardStrength().apply(pdc_matrix)
        >>> inward = cb.InwardStrength(freq_band=(30.0, 80.0)).apply(pdc_matrix)
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
                "InwardStrength is matrix-only. Pipe a directed connectivity "
                "feature first, e.g. (PartialDirectedCoherence() | "
                "InwardStrength()).apply(signal)."
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
            warnings.warn(
                f"freq_band={self.freq_band} is set but the connectivity matrix "
                "has no 'frequency' dimension. Ignoring freq_band.",
                UserWarning,
                stacklevel=2,
            )

        mat_vals = mat.values.astype(float).copy()
        n_ch = mat_vals.shape[0]
        np.fill_diagonal(mat_vals, np.nan)
        inward = np.nanmean(mat_vals, axis=1)

        if "space_to" in mat.coords:
            space_vals = mat.coords["space_to"].values
        else:
            space_vals = np.arange(n_ch)

        return xr.DataArray(inward, dims=["space"], coords={"space": space_vals})


@functional(InwardStrength)
def inward_strength(data: Data, freq_band: tuple[float, float] | None = None) -> Data:
    """Compute per-channel inward strength from a directed connectivity matrix.

    For each channel ``i``, inward strength is the mean influence *received*
    from all other channels (excluding self-connections)::

        inward[i] = mean over j != i of mat[i, j]

    where ``mat[i, j]`` is the influence from channel ``j`` to channel ``i``
    (convention: ``space_to`` = row, ``space_from`` = column).

    Matrix-only input. To compute from a raw signal, pipe through a directed
    connectivity feature first::

        inward = (PartialDirectedCoherence() | InwardStrength()).apply(signal)

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
        >>> inward = cb.inward_strength(pdc_matrix)
        >>> inward = cb.inward_strength(pdc_matrix, freq_band=(30.0, 80.0))
    """
    return InwardStrength(freq_band=freq_band).apply(data)
