# Connectivity features: which regions are synchronized/interacting.
from ._coherence import Coherence
from ._correlation import Correlation
from ._covariance import Covariance
from ._directed_transfer_function import DirectedTransferFunction
from ._envelope_correlation import EnvelopeCorrelation
from ._granger_causality import GrangerCausality
from ._mutual_information import MutualInformation
from ._partial_correlation import PartialCorrelation
from ._partial_directed_coherence import PartialDirectedCoherence
from ._phase_locking_value import PhaseLockingValue
from ._reciprocal_connectivity import ReciprocalConnectivity

__all__ = [
    "Coherence",
    "Correlation",
    "Covariance",
    "DirectedTransferFunction",
    "EnvelopeCorrelation",
    "GrangerCausality",
    "MutualInformation",
    "PartialCorrelation",
    "PartialDirectedCoherence",
    "PhaseLockingValue",
    "ReciprocalConnectivity",
]

# One-shot functional wrappers (cb.line_length(d) beside cb.LineLength().apply(d)). See GH #116.
from .._functional import install as _install_functional

_install_functional(__name__)
