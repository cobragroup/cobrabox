"""Transforms: How do I transform my signal into another representation?

Operations that change a signal's representation without reducing it to
summary measures: into and out of the frequency domain (Fourier and inverse
Fourier transforms), the analytic signal, a normalized scale, and filtered
versions of the signal. Spectral measures such as power spectral density and
band power live in ``cb.spectral``.

Every feature here comes in two forms, both also exported at the top level:
the class (``cb.Normalize()``), which composes into pipelines with ``|``, and a
one-shot function (``cb.normalize(data)``). ``cb.transforms.__all__`` lists
them, and ``help()`` on any of them gives its parameters.
"""

from ._analytic_signal import AnalyticSignal, analytic_signal
from ._fourier_transform import FourierTransform, fourier_transform
from ._inverse_fourier_transform import InverseFourierTransform, inverse_fourier_transform
from ._normalize import Normalize, normalize
from ._notch_filter import NotchFilter, notch_filter

__all__ = [
    "AnalyticSignal",
    "FourierTransform",
    "InverseFourierTransform",
    "Normalize",
    "NotchFilter",
    "analytic_signal",
    "fourier_transform",
    "inverse_fourier_transform",
    "normalize",
    "notch_filter",
]
