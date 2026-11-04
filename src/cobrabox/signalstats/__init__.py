"""Signal Statistics: What are the basic properties of my signal?

Per-channel descriptors computed straight from the time series: amplitude
range and variation, line length, autocorrelation, and epilepsy-oriented
markers such as spike counts and the epileptogenicity index.

Every feature here comes in two forms, both also exported at the top level:
the class (``cb.LineLength()``), which composes into pipelines with ``|``, and a
one-shot function (``cb.line_length(data)``). ``cb.signalstats.__all__`` lists
them, and ``help()`` on any of them gives its parameters.
"""

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
