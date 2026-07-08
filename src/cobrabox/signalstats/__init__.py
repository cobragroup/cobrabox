# Signal-statistics features: basic properties of a signal.
from .amplitude_variation import AmplitudeVariation
from .autocorrelation import Autocorrelation
from .epileptogenicity_index import EpileptogenicityIndex
from .line_length import LineLength
from .max import Max
from .mean import Mean
from .min import Min
from .spike_count import SpikeCount

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
