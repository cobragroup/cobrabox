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
class ReciprocalConnectivity(BaseFeature[Data]):
    """Compute per-channel Reciprocal Connectivity (RC) from a directed connectivity matrix.

    **Reciprocal Connectivity** quantifies the net directional role of each
    channel: positive values indicate a net *sink* (more influence received
    than sent), and negative values indicate a net *source* (more influence
    sent than received).

    For each channel ``i``::

        RC[i] = in_strength[i] - out_strength[i]

    where (with ``mat[i, j]`` = flow *from j to i*):

    * ``in_strength[i]``  = mean over j of ``mat[i, j, band]``
    * ``out_strength[i]`` = mean over j of ``mat[j, i, band]``

    Matrix-only input. To compute RC from a raw signal, pipe it through a
    directed connectivity feature first::

        rc = (PartialDirectedCoherence() | ReciprocalConnectivity()).apply(signal)

    Args:
        freq_band: Frequency band ``(fmin, fmax)`` in Hz to average over.
            Required when the input matrix has a ``"frequency"`` dimension;
            pass ``None`` when the input is already a single matrix.
            Default ``(30.0, 80.0)`` Hz.
        normalize: If ``True``, z-score the matrix (excluding the diagonal)
            before computing in/out strengths.

    Returns:
        :class:`~cobrabox.Data` with dim ``("space",)``, shape ``(n_channels,)``.

    Raises:
        ValueError: If the input lacks ``"space_to"`` and ``"space_from"``
            dimensions, if the matrix is asymmetric vs symmetric requirements
            are violated, or if ``freq_band`` is incompatible with the input.

    Example:
        >>> rc = cb.ReciprocalConnectivity(freq_band=(30.0, 80.0)).apply(pdc_matrix)
        >>> rc = (cb.PartialDirectedCoherence() | rc).apply(signal_data)
    """

    _tags: ClassVar[list[str]] = [
        "directed",
        "sink-source",
        "post-processing",
        "epilepsy",
        "seizure-onset-zone",
        "eeg",
        "io:vector",
        "req:asymmetric-matrix",
    ]

    freq_band: tuple[float, float] | None = field(default=(30.0, 80.0))
    normalize: bool = False

    output_type: ClassVar[type[Data]] = Data

    def __post_init__(self) -> None:
        if self.freq_band is not None:
            fmin, fmax = self.freq_band
            if fmin >= fmax:
                raise ValueError(f"freq_band must satisfy fmin < fmax, got ({fmin}, {fmax}).")

    @staticmethod
    def _is_symmetric(mat: xr.DataArray) -> bool:
        """Symmetry across (space_to, space_from), preserving any other dims."""
        swapped = mat.transpose("space_from", "space_to", ...).rename(
            {"space_from": "space_to", "space_to": "space_from"}
        )
        return bool(np.allclose(mat.values, swapped.values, atol=1e-6))

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data
        dims = xr_data.dims

        if "time" in dims:
            raise ValueError(
                "ReciprocalConnectivity is matrix-only. Pipe a directed connectivity "
                "feature first, e.g. (PartialDirectedCoherence() | "
                "ReciprocalConnectivity()).apply(signal)."
            )

        if "space_to" not in dims or "space_from" not in dims:
            raise ValueError(
                f"Input must have 'space_to' and 'space_from' dimensions. Got dims: {dims}."
            )

        mat = xr_data

        if self._is_symmetric(mat):
            warnings.warn(
                "Input matrix appears symmetric. ReciprocalConnectivity requires "
                "a directed/asymmetric connectivity matrix (e.g. from PDC, Granger, DTF). "
                "Symmetric input will produce near-zero values.",
                UserWarning,
                stacklevel=2,
            )

        if self.freq_band is not None:
            if "frequency" not in mat.dims:
                raise ValueError(
                    f"freq_band={self.freq_band} is set but the connectivity matrix "
                    "has no 'frequency' dimension. Pass freq_band=None to skip band averaging."
                )
            freqs = mat.coords["frequency"].values
            fmin, fmax = self.freq_band
            if fmin > freqs.max() or fmax < freqs.min():
                raise ValueError(
                    f"freq_band ({fmin}, {fmax}) Hz is outside the available "
                    f"frequency range [{freqs.min():.4g}, {freqs.max():.4g}] Hz."
                )
            mat = mat.sel(frequency=slice(fmin, fmax)).mean("frequency")
        elif "frequency" in mat.dims:
            raise ValueError(
                "The connectivity matrix has a 'frequency' dimension but "
                "freq_band=None. Set freq_band to (fmin, fmax) to select a band."
            )

        mat_vals = mat.values.copy().astype(float)
        n_ch = mat_vals.shape[0]

        if self.normalize:
            mask = ~np.eye(n_ch, dtype=bool)
            off_diag = mat_vals[mask]
            mu, sigma = off_diag.mean(), off_diag.std()
            if sigma > 0:
                mat_vals = (mat_vals - mu) / sigma

        np.fill_diagonal(mat_vals, np.nan)
        in_strength = np.nanmean(mat_vals, axis=1)
        out_strength = np.nanmean(mat_vals, axis=0)
        rc = in_strength - out_strength

        if "space_to" in mat.coords:
            space_vals = mat.coords["space_to"].values
        else:
            space_vals = np.arange(n_ch)

        return xr.DataArray(rc, dims=["space"], coords={"space": space_vals})


@functional(ReciprocalConnectivity)
def reciprocal_connectivity(
    data: Data, freq_band: tuple[float, float] | None = (30.0, 80.0), normalize: bool = False
) -> Data:
    """Compute per-channel Reciprocal Connectivity (RC) from a directed connectivity matrix.

    **Reciprocal Connectivity** quantifies the net directional role of each
    channel: positive values indicate a net *sink* (more influence received
    than sent), and negative values indicate a net *source* (more influence
    sent than received).

    For each channel ``i``::

        RC[i] = in_strength[i] - out_strength[i]

    where (with ``mat[i, j]`` = flow *from j to i*):

    * ``in_strength[i]``  = mean over j of ``mat[i, j, band]``
    * ``out_strength[i]`` = mean over j of ``mat[j, i, band]``

    Matrix-only input. To compute RC from a raw signal, pipe it through a
    directed connectivity feature first::

        rc = (PartialDirectedCoherence() | ReciprocalConnectivity()).apply(signal)

    Args:
        data: The input data to process, as a :class:`~cobrabox.Data`.
        freq_band: Frequency band ``(fmin, fmax)`` in Hz to average over.
            Required when the input matrix has a ``"frequency"`` dimension;
            pass ``None`` when the input is already a single matrix.
            Default ``(30.0, 80.0)`` Hz.
        normalize: If ``True``, z-score the matrix (excluding the diagonal)
            before computing in/out strengths.

    Returns:
        :class:`~cobrabox.Data` with dim ``("space",)``, shape ``(n_channels,)``.

    Raises:
        ValueError: If the input lacks ``"space_to"`` and ``"space_from"``
            dimensions, if the matrix is asymmetric vs symmetric requirements
            are violated, or if ``freq_band`` is incompatible with the input.

    Example:
        >>> rc = cb.reciprocal_connectivity(pdc_matrix, freq_band=(30.0, 80.0))
        >>> rc = (cb.PartialDirectedCoherence() | rc).apply(signal_data)
    """
    return ReciprocalConnectivity(freq_band=freq_band, normalize=normalize).apply(data)
