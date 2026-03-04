"""Epileptogenicity Index (EI) per channel.

Bartolomei F, Chauvel P, Wendling F. Epileptogenicity of brain structures in human
temporal lobe epilepsy: a quantified study from intracerebral EEG.
Brain (2008), 131:1818-1830. DOI: 10.1093/brain/awn111
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import SignalData

# Frequency band boundaries (Hz) — Table 2, Bartolomei et al. (2008)
_THETA = (3.5, 7.4)
_ALPHA = (7.4, 12.4)
_BETA = (12.4, 24.0)
_GAMMA_MIN = 24.0  # upper bound = Nyquist (paper capped at 97 Hz for 256 Hz recording)

_EPS = np.finfo(float).eps


@dataclass
class EpileptogenicityIndex(BaseFeature[SignalData]):
    """Compute the Epileptogenicity Index (EI) per channel (Bartolomei et al., 2008).

    EI quantifies the epileptogenicity of each recorded brain structure by combining
    two features of the ictal SEEG signal: the *spectral* property (presence of
    high-frequency rapid discharges) and the *temporal* property (how early in the
    seizure process the discharge appears relative to other channels). Higher EI
    indicates a more epileptogenic structure.

    The algorithm operates in three stages:

    1. **Energy Ratio ER[n]** — a sliding-window periodogram is computed along each
       channel. At each window position the ratio of high- to low-frequency band
       power is computed::

           ER[n] = (E_beta + E_gamma) / (E_theta + E_alpha)

       Frequency bands (Table 2 of the paper):
       θ 3.5-7.4 Hz · alpha 7.4-12.4 Hz · beta 12.4-24 Hz · gamma 24 Hz-Nyquist

    2. **Page-Hinkley (CUSUM) detection** — applied per channel to find the detection
       time *N_d* (onset of the rapid discharge). Channels where no alarm fires are
       assigned *N_d* = last ER sample, which yields near-zero EI after normalisation.

    3. **EI formula and normalisation** — given the earliest detection time *N_0*
       (reference channel) and integration window *H* samples::

           EI_i = sum(ER[N_di : N_di + H]) / (N_di - N_0 + tau)

       Values are divided by the per-recording maximum to obtain a [0, 1] scale.

    Reference:
        Bartolomei F, Chauvel P, Wendling F. *Epileptogenicity of brain structures
        in human temporal lobe epilepsy: a quantified study from intracerebral EEG.*
        Brain (2008), 131:1818-1830. DOI: 10.1093/brain/awn111

    Args:
        window_duration: Duration of the ER sliding window in seconds. Longer windows
            give better frequency resolution but coarser temporal resolution.
            Default: 1.0 s.
        bias: Page-Hinkley bias *v*. Higher values suppress small ER fluctuations and
            reduce false detections. Default: 0.5.
        threshold: Page-Hinkley alarm threshold *λ*. Higher values reduce false alarms
            at the cost of delayed or missed detections. Default: 30.0.
        integration_window: Duration *H* (seconds) over which ER[n] is summed after
            detection to capture the rapid-discharge energy. Default: 5.0 s.
        tau: Small constant *τ* preventing division by zero when a channel is the
            first to fire (delay = 0). Default: 1.0.

    Returns:
        xarray DataArray with dim ``(space,)``, values normalised to [0, 1]. If no
        rapid discharge is detected in any channel, all values are 0.

    Raises:
        ValueError: If ``data`` does not have exactly ``time`` and ``space`` dimensions.
        ValueError: If ``data.sampling_rate`` is not set.
        ValueError: If any channel's signal is shorter than one ER window.

    Example:
        >>> ei = cb.feature.EpileptogenicityIndex().apply(seizure_data)
        >>> ei_vals = ei.data  # (space,) values in [0, 1]
    """

    window_duration: float = 1.0
    bias: float = 0.5
    threshold: float = 30.0
    integration_window: float = 5.0
    tau: float = 1.0

    def _energy_ratio(self, signal: np.ndarray, fs: float, window_samples: int) -> np.ndarray:
        """Compute ER[n] time series using a sliding-window periodogram (step = 1 sample).

        At each position n, the ratio of high- to low-frequency band power is computed
        from the one-sided periodogram of signal[n : n+window_samples]:

            ER[n] = (E_beta + E_gamma) / (E_theta + E_alpha)

        Because only the ratio matters, raw |FFT|² sums are used (normalisation cancels).
        All windows are batched via stride tricks before the FFT to avoid a Python loop.

        Args:
            signal: 1-D float64 array, shape (n_time,).
            fs: Sampling rate in Hz.
            window_samples: Window length in samples.

        Returns:
            ER time series, shape (n_time - window_samples + 1,).

        Raises:
            ValueError: If the signal is shorter than one window.
        """
        n = len(signal)
        n_steps = n - window_samples + 1
        if n_steps <= 0:
            raise ValueError(
                f"Signal length ({n} samples) is shorter than window_duration "
                f"({window_samples} samples). Reduce window_duration or provide a longer signal."
            )

        freqs = np.fft.rfftfreq(window_samples, d=1.0 / fs)
        theta_mask = (freqs >= _THETA[0]) & (freqs < _THETA[1])
        alpha_mask = (freqs >= _ALPHA[0]) & (freqs < _ALPHA[1])
        beta_mask = (freqs >= _BETA[0]) & (freqs < _BETA[1])
        gamma_mask = freqs >= _GAMMA_MIN

        # Create overlapping windows as a read-only strided view, then batch-FFT
        strides = (signal.strides[0], signal.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            signal, shape=(n_steps, window_samples), strides=strides
        )
        psd = np.abs(np.fft.rfft(windows, axis=-1)) ** 2

        numer = psd[:, beta_mask].sum(-1) + psd[:, gamma_mask].sum(-1)
        denom = psd[:, theta_mask].sum(-1) + psd[:, alpha_mask].sum(-1)
        return numer / (denom + _EPS)

    def _page_hinkley(self, er: np.ndarray, v: float, lam: float) -> int | None:
        """Page-Hinkley (CUSUM) change-point detection on an ER[n] series.

        Builds a cumulative sum U_N that drifts downward under H0 (no change) and
        rises when ER[n] exceeds its running mean by more than the bias v.  An alarm
        fires when the rise from the last local minimum exceeds the threshold lam.

        Args:
            er: Energy-ratio time series, shape (n,).
            v: Bias — controls which magnitude of ER increase is ignored (sensitivity).
            lam: Alarm threshold — higher values reduce false alarms.

        Returns:
            Detection time N_d = index of the last U_N local minimum before the alarm
            (the estimated onset of the rapid discharge), or ``None`` if no alarm fires.
        """
        er_mean = 0.0
        U = 0.0
        u_min = 0.0
        u_min_idx = 0

        for i, er_i in enumerate(er):
            er_mean = er_mean * i / (i + 1) + er_i / (i + 1)  # online running mean
            U += er_i - er_mean - v
            if U < u_min:
                u_min = U
                u_min_idx = i
            if U - u_min > lam:
                return u_min_idx

        return None

    def __call__(self, data: SignalData) -> xr.DataArray:
        if set(data.data.dims) != {"time", "space"}:
            raise ValueError(
                "epileptogenicity_index requires exactly 'time' and 'space' dimensions, "
                f"got {set(data.data.dims)}"
            )
        fs = data.sampling_rate
        if fs is None:
            raise ValueError(
                "sampling_rate must be set on data to compute epileptogenicity_index; "
                "pass sampling_rate when constructing the Data object"
            )

        window_samples = max(2, round(self.window_duration * fs))
        h_samples = max(1, round(self.integration_window * fs))

        # Transpose to (space, time) so we can iterate over channels on axis 0
        xr_data = data.data.transpose("space", "time")
        arr = xr_data.values.astype(np.float64)  # (n_channels, n_time)
        n_channels = arr.shape[0]
        space_coords = xr_data.coords["space"].values

        # Stages 1 & 2: ER time series + Page-Hinkley per channel
        er_list: list[np.ndarray] = []
        detection_times: list[int] = []
        for ch in range(n_channels):
            er = self._energy_ratio(arr[ch], fs, window_samples)
            nd = self._page_hinkley(er, self.bias, self.threshold)
            er_list.append(er)
            # Channels without a detected rapid discharge → assign last ER index
            detection_times.append(nd if nd is not None else len(er) - 1)

        # Stage 3: EI formula
        n0 = min(detection_times)
        ei_values = np.empty(n_channels)
        for i in range(n_channels):
            nd = detection_times[i]
            er_slice = er_list[i][nd : nd + h_samples]
            ei_values[i] = er_slice.sum() / (nd - n0 + self.tau)

        # Normalise to [0, 1]; guard against all-zero (no rapid discharge detected anywhere)
        max_ei = ei_values.max()
        if max_ei > 0:
            ei_values /= max_ei

        return xr.DataArray(ei_values, dims=["space"], coords={"space": space_coords})
