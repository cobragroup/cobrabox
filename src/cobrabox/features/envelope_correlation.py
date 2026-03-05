from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr
from mne_connectivity import envelope_correlation as _mne_envelope_correlation

from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class EnvelopeCorrelation(BaseFeature[SignalData]):
    """Compute amplitude envelope correlation (AEC) between all channel pairs.

    Estimates the correlation between the amplitude envelopes of each pair of
    spatial channels. Envelopes are extracted via the analytic signal (Hilbert
    transform). When ``orthogonalize='pairwise'``, zero-lag contributions are
    removed between each pair before correlating, which reduces the influence
    of volume conduction.

    Delegates to :func:`mne_connectivity.envelope_correlation` with
    ``log=False``.

    Args:
        orthogonalize: Whether and how to orthogonalise signal pairs before
            correlating. ``'pairwise'`` (default) removes zero-lag correlations
            per pair. ``False`` skips orthogonalisation.
        absolute: If ``True``, take the absolute value of correlation
            coefficients before symmetrising. Defaults to ``False`` (unlike
            the mne-connectivity default of ``True``).

    Returns:
        xarray DataArray with dims ``(space_to, space_from)`` containing
        Pearson correlation values between amplitude envelopes.

    Raises:
        ValueError: If ``data`` has dimensions other than ``time`` and
            ``space``.
        ValueError: If ``data`` has fewer than 2 spatial channels.

    Example:
        >>> aec = cb.feature.EnvelopeCorrelation().apply(data)
        >>> aec_signed = cb.feature.EnvelopeCorrelation(absolute=False).apply(data)
        >>> aec_no_orth = cb.feature.EnvelopeCorrelation(orthogonalize=False).apply(data)
    """

    output_type: ClassVar[type[Data]] = Data

    orthogonalize: str | bool = "pairwise"
    absolute: bool = False

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data

        extra_dims = [d for d in xr_data.dims if d not in ("space", "time")]
        if extra_dims:
            raise ValueError(
                f"envelope_correlation only supports 'space' and 'time' dimensions, "
                f"got extra dims: {extra_dims}"
            )

        space_coords = xr_data.coords["space"].values if "space" in xr_data.coords else None
        n_space = xr_data.sizes["space"]

        if n_space < 2:
            raise ValueError(
                f"envelope_correlation requires at least 2 spatial channels, got {n_space}"
            )

        # values shape needs to be (space, time) for mne_connectivity
        values = xr_data.transpose("space", "time").values

        conn = _mne_envelope_correlation(
            [values],
            orthogonalize=self.orthogonalize,
            log=False,
            absolute=self.absolute,
            verbose=False,
        )

        # get_data() → (n_epochs, n_nodes, n_nodes, n_times); take the single epoch and time
        mat = conn.get_data()[0, :, :, 0]

        da = xr.DataArray(mat, dims=["space_to", "space_from"])
        coords: dict[str, np.ndarray] = {}
        if space_coords is not None:
            coords["space_to"] = space_coords
            coords["space_from"] = space_coords
        return da.assign_coords(coords)
