from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.signal import spectrogram as _sp_spectrogram
from scipy.signal import stft as _sp_stft

from ..base_feature import BaseFeature
from ..data import SignalData

_VALID_SCALINGS = ("log", "density", "spectrum", "magnitude")


@dataclass
class Spectrogram(BaseFeature[SignalData]):
    """Compute the power spectrogram for each spatial channel.

    Uses ``scipy.signal.spectrogram`` (Welch-style STFT) to estimate the
    short-time power spectral density (or magnitude spectrum) for every
    channel independently.  Extra dimensions (e.g. ``window_index``) are
    preserved: the spectrogram is computed along the ``time`` axis for each
    slice of those dimensions.

    Args:
        nperseg: Samples per FFT segment.  Controls frequency resolution.
            Defaults to ``min(256, n_time)``.  Must be >= 2 and <= n_time.
        noverlap: Overlap between consecutive segments in samples.  Defaults
            to ``nperseg // 2`` (50 % overlap).  Must be < nperseg.
        window: Window function name passed to scipy (e.g. ``"hann"``,
            ``"hamming"``).  Defaults to ``"hann"``.
        scaling: Output scaling.  One of:

            * ``"log"`` *(default)* — 10 · log₁₀(PSD) in dB re 1 V²/Hz.
              Values where PSD ≈ 0 are clamped to avoid -inf.
            * ``"density"`` — power spectral density in V²/Hz.
            * ``"spectrum"`` — power spectrum in V² (not normalised by
              bandwidth).
            * ``"magnitude"`` — |STFT| (absolute value of complex STFT
              coefficients, computed via ``scipy.signal.stft``).

    Returns:
        xarray DataArray with dims ``(*extra_dims, "space", "frequency",
        "time")`` where ``frequency`` is in Hz and ``time`` holds the window
        centre times in seconds (as returned by scipy).

    Raises:
        ValueError: If ``scaling`` is not one of the four valid options, if
            ``nperseg`` is invalid, or if ``noverlap >= nperseg``.

    Example:
        >>> data = cb.dataset("dummy_random")[0]
        >>> sg = cb.feature.Spectrogram().apply(data)
        >>> sg.data.dims
        ('space', 'frequency', 'time')
    """

    nperseg: int | None = None
    noverlap: int | None = None
    window: str = "hann"
    scaling: str = "log"

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data

        if self.scaling not in _VALID_SCALINGS:
            raise ValueError(f"scaling must be one of {_VALID_SCALINGS!r}, got {self.scaling!r}")

        n_time = xr_data.sizes["time"]
        seg = min(256, n_time) if self.nperseg is None else self.nperseg

        if seg < 2:
            raise ValueError("nperseg must be at least 2")
        if seg > n_time:
            raise ValueError(
                f"nperseg ({seg}) must not exceed the number of time samples ({n_time})"
            )
        if self.noverlap is not None and self.noverlap >= seg:
            raise ValueError(f"noverlap ({self.noverlap}) must be less than nperseg ({seg})")

        fs = data.sampling_rate if data.sampling_rate is not None else 1.0

        # Transpose to (*extra_dims, space, time) for uniform indexing
        extra_dims = [d for d in xr_data.dims if d not in ("space", "time")]
        xr_t = xr_data.transpose(*extra_dims, "space", "time")
        arr = xr_t.values  # (*extra_shape, n_space, n_time)

        space_coords = xr_data.coords["space"].values
        extra_shape = arr.shape[:-2]
        n_space = arr.shape[-2]

        # Flatten to (n_total, n_time) to loop uniformly, reshape back afterwards
        arr_flat = arr.reshape(-1, n_time)
        n_total = arr_flat.shape[0]

        # Determine output shape from a single test call
        scipy_scaling = "density" if self.scaling in ("log", "density") else "spectrum"
        if self.scaling == "magnitude":
            f_axis, t_axis, _ = _sp_stft(
                arr_flat[0], fs=fs, nperseg=seg, noverlap=self.noverlap, window=self.window
            )
        else:
            f_axis, t_axis, _ = _sp_spectrogram(
                arr_flat[0],
                fs=fs,
                nperseg=seg,
                noverlap=self.noverlap,
                window=self.window,
                scaling=scipy_scaling,
            )

        n_freqs = len(f_axis)
        n_t_out = len(t_axis)
        out_flat = np.empty((n_total, n_freqs, n_t_out), dtype=np.float64)

        for i in range(n_total):
            sig = arr_flat[i]
            if self.scaling == "magnitude":
                _, _, Zxx = _sp_stft(
                    sig, fs=fs, nperseg=seg, noverlap=self.noverlap, window=self.window
                )
                out_flat[i] = np.abs(Zxx)
            else:
                _, _, Sxx = _sp_spectrogram(
                    sig,
                    fs=fs,
                    nperseg=seg,
                    noverlap=self.noverlap,
                    window=self.window,
                    scaling=scipy_scaling,
                )
                if self.scaling == "log":
                    out_flat[i] = 10.0 * np.log10(np.maximum(Sxx, np.finfo(np.float64).tiny))
                else:
                    out_flat[i] = Sxx

        out = out_flat.reshape(*extra_shape, n_space, n_freqs, n_t_out)

        extra_coords = {d: xr_data.coords[d].values for d in extra_dims if d in xr_data.coords}
        return xr.DataArray(
            out,
            dims=(*extra_dims, "space", "frequency", "time"),
            coords={**extra_coords, "space": space_coords, "frequency": f_axis, "time": t_axis},
            attrs={"sampling_rate": fs},
        )
