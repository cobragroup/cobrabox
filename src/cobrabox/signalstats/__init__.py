# Signal-statistics features: basic properties of a signal.
from ._amplitude_variation import AmplitudeVariation
from ._autocorrelation import Autocorrelation
from ._epileptogenicity_index import EpileptogenicityIndex
from ._line_length import LineLength
from ._max import Max
from ._mean import Mean
from ._min import Min
from ._spike_count import SpikeCount

__all__ = [
    "AmplitudeVariation",
    "Autocorrelation",
    "EpileptogenicityIndex",
    "LineLength",
    "Max",
    "Mean",
    "Min",
    "SpikeCount",
]

# One-shot functional wrappers (cb.line_length(d) beside cb.LineLength().apply(d)). See GH #116.
from .._functional import install as _install_functional

_install_functional(__name__)
