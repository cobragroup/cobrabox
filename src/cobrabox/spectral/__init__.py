# Spectral features: what's happening in frequency space.
from ._band_power import BandPower
from ._continuous_wavelet_transform import ContinuousWaveletTransform
from ._cordance import Cordance
from ._discrete_wavelet_transform import DiscreteWaveletTransform
from ._spectrogram import Spectrogram

__all__ = [
    "BandPower",
    "ContinuousWaveletTransform",
    "Cordance",
    "DiscreteWaveletTransform",
    "Spectrogram",
]

# One-shot functional wrappers (cb.line_length(d) beside cb.LineLength().apply(d)). See GH #116.
from .._functional import install as _install_functional

_install_functional(__name__)
