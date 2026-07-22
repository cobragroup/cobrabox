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
