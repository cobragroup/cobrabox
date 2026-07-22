# Infometric features: complexity, regularity and information content.
from ._amplitude_entropy import AmplitudeEntropy
from ._fractal_dimension import FractalDimension
from ._lempel_ziv import LempelZiv
from ._nonreversibility import Nonreversibility
from ._recurrence_matrix import RecurrenceMatrix
from ._sample_entropy import SampleEntropy

__all__ = [
    "AmplitudeEntropy",
    "FractalDimension",
    "LempelZiv",
    "Nonreversibility",
    "RecurrenceMatrix",
    "SampleEntropy",
]

# One-shot functional wrappers (cb.line_length(d) beside cb.LineLength().apply(d)). See GH #116.
from .._functional import install as _install_functional

_install_functional(__name__)
