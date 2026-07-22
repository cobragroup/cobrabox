# Transform features: change the signal's representation without characterizing
# frequency content.
from ._analytic_signal import AnalyticSignal
from ._bandpass_filter import BandpassFilter
from ._fourier_transform import FourierTransform
from ._inverse_fourier_transform import InverseFourierTransform

__all__ = ["AnalyticSignal", "BandpassFilter", "FourierTransform", "InverseFourierTransform"]

# One-shot functional wrappers (cb.line_length(d) beside cb.LineLength().apply(d)). See GH #116.
from .._functional import install as _install_functional

_install_functional(__name__)
