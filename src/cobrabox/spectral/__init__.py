"""Spectral: What's happening in frequency space?

Frequency and time-frequency views of a signal: power spectral density,
band power, spectrograms, wavelet transforms, and derived measures such as
cordance.

Every feature here comes in two forms, both also exported at the top level:
the class (``cb.BandPower()``), which composes into pipelines with ``|``, and a
one-shot function (``cb.band_power(data)``). ``cb.spectral.__all__`` lists
them, and ``help()`` on any of them gives its parameters.
"""

from ._band_power import BandPower, band_power
from ._continuous_wavelet_transform import ContinuousWaveletTransform, continuous_wavelet_transform
from ._cordance import Cordance, cordance
from ._discrete_wavelet_transform import DiscreteWaveletTransform, discrete_wavelet_transform
from ._power_spectral_density import PowerSpectralDensity, power_spectral_density
from ._spectrogram import Spectrogram, spectrogram

__all__ = [
    "BandPower",
    "ContinuousWaveletTransform",
    "Cordance",
    "DiscreteWaveletTransform",
    "PowerSpectralDensity",
    "Spectrogram",
    "band_power",
    "continuous_wavelet_transform",
    "cordance",
    "discrete_wavelet_transform",
    "power_spectral_density",
    "spectrogram",
]
