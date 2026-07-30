# Signal-statistics features: basic properties of a signal.
from ._amplitude_variation import AmplitudeVariation, amplitude_variation
from ._autocorrelation import Autocorrelation, autocorrelation
from ._epileptogenicity_index import EpileptogenicityIndex, epileptogenicity_index
from ._line_length import LineLength, line_length
from ._max import Max, max
from ._mean import Mean, mean
from ._min import Min, min
from ._spike_count import SpikeCount, spike_count

__all__ = [
    "AmplitudeVariation",
    "Autocorrelation",
    "EpileptogenicityIndex",
    "LineLength",
    "Max",
    "Mean",
    "Min",
    "SpikeCount",
    "amplitude_variation",
    "autocorrelation",
    "epileptogenicity_index",
    "line_length",
    "max",
    "mean",
    "min",
    "spike_count",
]
