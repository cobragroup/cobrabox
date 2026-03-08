"""Recurrence matrix feature: pairwise similarity across time-points or windows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray as xr
from scipy.signal import hilbert
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from ..base_feature import BaseFeature
from ..data import Data, SignalData
from .envelope_correlation import EnvelopeCorrelation
from .phase_locking_value import _compute_plv

_REC_METRICS = ("cosine", "correlation", "euclidean")
_FC_METRICS = ("pearson", "spearman", "MI", "PLV", "AEC")


def _similarity_matrix(vecs: np.ndarray, metric: str) -> np.ndarray:
    """Compute pairwise similarity/distance matrix from row vectors (n, d) → (n, n)."""
    if metric == "cosine":
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / norms
        return normed @ normed.T
    elif metric == "correlation":
        return np.corrcoef(vecs)
    else:  # euclidean
        return cdist(vecs, vecs, metric="euclidean")


def _fc_matrix(window: np.ndarray, fc_metric: str) -> np.ndarray:
    """Compute FC matrix for a window of shape (n_time, n_channels).

    Args:
        window: Array of shape (n_time, n_channels).
        fc_metric: One of 'pearson', 'spearman', 'MI', 'PLV', 'AEC'.

    Returns:
        FC matrix of shape (n_channels, n_channels).
    """
    if fc_metric == "pearson":
        return np.corrcoef(window.T)

    elif fc_metric == "spearman":
        corr, _ = spearmanr(window)
        return np.atleast_2d(corr) if window.shape[1] == 2 else corr

    elif fc_metric == "MI":
        n_ch = window.shape[1]
        fc = np.zeros((n_ch, n_ch))
        bins = 10
        from sklearn.metrics import mutual_info_score
        for i in range(n_ch):
            xi = np.digitize(window[:, i], np.histogram_bin_edges(window[:, i], bins=bins))
            for j in range(i, n_ch):
                xj = np.digitize(window[:, j], np.histogram_bin_edges(window[:, j], bins=bins))
                mi = mutual_info_score(xi, xj)
                fc[i, j] = mi
                fc[j, i] = mi
        return fc

    elif fc_metric == "PLV":
        # Vectorized: extract analytic phase for all channels, compute all pairs at once
        # window.T shape: (n_channels, n_time)
        phases = np.angle(hilbert(window.T, axis=1))  # (n_channels, n_time)
        # phase difference for all pairs: (n_channels, n_channels, n_time)
        phase_diff = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]
        return np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))  # (n_channels, n_channels)

    else:  # AEC — wrap window into SignalData and use EnvelopeCorrelation feature
        # window.T shape: (n_channels, n_time); EnvelopeCorrelation expects (space, time)
        sig = SignalData.from_numpy(window.T, dims=["space", "time"], sampling_rate=1.0)
        result = EnvelopeCorrelation().apply(sig)
        return result.data.values


@dataclass
class RecurrenceMatrix(BaseFeature[Data]):
    """Compute a pairwise recurrence (self-similarity) matrix from a time-series.

    Behaviour depends on the **shape of the input** and the contents of
    ``fc_options``:

    **2-D input** ``(N, T)`` — multivariate time-series:

    - ``fc_options=[]`` *(default)*: each time-point is used as a state
      vector directly.  Output shape ``(T, T)``.
    - ``fc_options=['pearson']``: FC computed per window with default
      ``window_size=10`` and ``overlap=0.5``.
    - ``fc_options=['pearson', 50]``: FC with ``window_size=50``,
      default ``overlap=0.5``.
    - ``fc_options=['pearson', 50, 0.25]``: FC with ``window_size=50``
      and ``overlap=0.25``.
    - Supported FC metrics: ``'pearson'``, ``'spearman'``, ``'MI'``, ``'PLV'``, ``'AEC'``.
    - A ``UserWarning`` is raised if ``window_size < 5``.

    **3-D input** ``(N, N, T)`` — time-series of FC matrices:

    Already-computed FC matrices; ``fc_options`` is ignored.
    Output shape ``(T, T)``.

    Args:
        rec_metric: Pairwise similarity metric.  One of ``'cosine'``
            *(default)*, ``'correlation'``, ``'euclidean'``.
        fc_options: List controlling window-based FC computation.
            ``[]`` = state-vector mode.
            ``[fc_metric]`` = FC with defaults.  Supported: ``'pearson'``,
            ``'spearman'``, ``'MI'``, ``'PLV'``, ``'AEC'``.
            ``[fc_metric, window_size]`` = FC with given window size.
            ``[fc_metric, window_size, overlap]`` = full control.

    Returns:
        xarray DataArray of shape ``(n, n)`` with dims ``('t1', 't2')``.

    Raises:
        ValueError: If a required dimension is missing.
        ValueError: If ``window_size >= n_time``.
        ValueError: If metric values are invalid.

    Example:
        >>> # state-vector mode (default)
        >>> rec = cb.feature.RecurrenceMatrix().apply(data)

        >>> # window/FC mode — just fc_metric, rest default
        >>> rec = cb.feature.RecurrenceMatrix('cosine', ['pearson']).apply(data)

        >>> # window/FC mode — full control
        >>> rec = cb.feature.RecurrenceMatrix('cosine', ['pearson', 50, 0.25]).apply(data)

        >>> # 3-D pre-computed FC time-series
        >>> rec = cb.feature.RecurrenceMatrix('cosine').apply(fc_data)
    """

    rec_metric: str = field(default="cosine")
    fc_options: list = field(default_factory=list)

    # Derived fields unpacked from fc_options in __post_init__
    _fc_metric: str = field(default="pearson", init=False, repr=False)
    _window_size: int = field(default=1,       init=False, repr=False)
    _overlap: float   = field(default=0.5,     init=False, repr=False)

    output_type: ClassVar[type[Data]] = Data

    def __post_init__(self) -> None:
        if self.rec_metric not in _REC_METRICS:
            raise ValueError(f"rec_metric must be one of {_REC_METRICS}, got '{self.rec_metric}'")

        opts = self.fc_options
        if len(opts) > 3:
            raise ValueError("fc_options may have at most 3 elements: [fc_metric, window_size, overlap]")

        if opts:
            self._fc_metric = opts[0]
            if self._fc_metric not in _FC_METRICS:
                raise ValueError(f"fc_options[0] (fc_metric) must be one of {_FC_METRICS}, got '{self._fc_metric}'")
            self._window_size = int(opts[1]) if len(opts) >= 2 else 10
            self._overlap     = float(opts[2]) if len(opts) >= 3 else 0.5

        if self._window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self._window_size}")
        if not (0.0 <= self._overlap < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {self._overlap}")

    def __call__(self, data: Data) -> xr.DataArray:
        xr_data = data.data

        if "time" not in xr_data.dims:
            raise ValueError("data must have a 'time' dimension")

        spatial_dims = [d for d in xr_data.dims if d != "time"]
        n_spatial = len(spatial_dims)

        if n_spatial not in (1, 2):
            raise ValueError(
                f"Expected 1 or 2 spatial dimensions, got {n_spatial}: {spatial_dims}"
            )

        # Move time to last axis → shape (..., T)
        arr = xr_data.transpose(*spatial_dims, "time").values
        n_time = arr.shape[-1]

        if n_spatial == 2:
            # --- 3-D input: (N, N, T) — pre-computed FC matrices ---
            N1, N2, _ = arr.shape
            if N1 != N2:
                raise ValueError(
                    f"For 3-D input the two spatial dimensions must be equal, "
                    f"got {N1} x {N2}"
                )
            triu_idx = np.triu_indices(N1, k=1)
            flat_vecs = np.array([arr[:, :, t][triu_idx] for t in range(n_time)])
            mat = _similarity_matrix(flat_vecs, self.rec_metric)
            n = n_time

        else:
            # --- 2-D input: (N, T) — multivariate time-series ---
            X = arr.T  # (n_time, n_channels)
            n_channels = X.shape[1]

            if not self.fc_options:
                # State-vector mode: each time-point is a vector
                mat = _similarity_matrix(X, self.rec_metric)
                n = n_time

            else:
                # Window / FC mode
                ws = self._window_size
                if ws < 5:
                    warnings.warn(
                        f"window_size={ws} is very small (<5); "
                        "FC estimates may be unreliable.",
                        UserWarning,
                        stacklevel=2,
                    )
                if ws >= n_time:
                    raise ValueError(
                        f"window_size ({ws}) must be < n_time ({n_time})"
                    )
                step_size = max(1, int(ws * (1.0 - self._overlap)))
                triu_idx = np.triu_indices(n_channels, k=1)
                starts = range(0, n_time - ws + 1, step_size)
                flat_vecs = np.array([
                    _fc_matrix(X[s : s + ws], self._fc_metric)[triu_idx]
                    for s in starts
                ])
                mat = _similarity_matrix(flat_vecs, self.rec_metric)
                n = len(starts)

        return xr.DataArray(
            mat,
            dims=["t1", "t2"],
            coords={"t1": np.arange(n), "t2": np.arange(n)},
        )
