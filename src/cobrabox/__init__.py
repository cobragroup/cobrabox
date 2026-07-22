# Domain subpackages, plus the flat `feature` convenience namespace
from . import (
    connectivity,
    decompositions,
    feature,
    infometrics,
    serialization,
    signalstats,
    spectral,
    surrogates,
    transforms,
    windowing,
)
from .base_feature import AggregatorFeature, BaseFeature, Chord, Pipeline, SplitterFeature

# Every feature is re-exported at the root namespace, so `cb.Correlation()` is the
# canonical way to reach one. `cb.<domain>.Correlation()` and `cb.Correlation()`
# remain valid aliases. `tests/test_public_api.py` fails if these drift apart.
from .connectivity import (
    Coherence,
    Correlation,
    Covariance,
    DirectedTransferFunction,
    EnvelopeCorrelation,
    GrangerCausality,
    MutualInformation,
    PartialCorrelation,
    PartialDirectedCoherence,
    PhaseLockingValue,
    ReciprocalConnectivity,
)
from .data import EEG, FMRI, Data, SignalData
from .dataset import Dataset
from .datasets import (
    dataset_info,
    delete_dataset,
    download_dataset,
    list_datasets,
    load_dataset,
    show_datasets,
)
from .decompositions import EMD, SVD
from .downloader import DownloadCancelled, get_dataset_dir, set_dataset_dir
from .egg.gorkastyle import gorkastyle
from .infometrics import (
    AmplitudeEntropy,
    FractalDimension,
    LempelZiv,
    Nonreversibility,
    RecurrenceMatrix,
    SampleEntropy,
)
from .serialization import deserialize, load, save, serialize
from .signalstats import (
    AmplitudeVariation,
    Autocorrelation,
    EpileptogenicityIndex,
    LineLength,
    Max,
    Mean,
    Min,
    SpikeCount,
)
from .spectral import (
    BandPower,
    ContinuousWaveletTransform,
    Cordance,
    DiscreteWaveletTransform,
    Spectrogram,
)
from .surrogates import FourierTransformSurrogates
from .transforms import AnalyticSignal, BandpassFilter, FourierTransform, InverseFourierTransform
from .windowing import ConcatAggregate, MeanAggregate, SlidingWindow, SlidingWindowReduce

# Package-level aliases for class methods
from_numpy = Data.from_numpy
from_xarray = Data.from_xarray

__all__ = [
    "EEG",
    "EMD",
    "FMRI",
    "SVD",
    "AggregatorFeature",
    "AmplitudeEntropy",
    "AmplitudeVariation",
    "AnalyticSignal",
    "Autocorrelation",
    "BandPower",
    "BandpassFilter",
    "BaseFeature",
    "Chord",
    "Coherence",
    "ConcatAggregate",
    "ContinuousWaveletTransform",
    "Cordance",
    "Correlation",
    "Covariance",
    "Data",
    "Dataset",
    "DirectedTransferFunction",
    "DiscreteWaveletTransform",
    "DownloadCancelled",
    "EnvelopeCorrelation",
    "EpileptogenicityIndex",
    "FourierTransform",
    "FourierTransformSurrogates",
    "FractalDimension",
    "GrangerCausality",
    "InverseFourierTransform",
    "LempelZiv",
    "LineLength",
    "Max",
    "Mean",
    "MeanAggregate",
    "Min",
    "MutualInformation",
    "Nonreversibility",
    "PartialCorrelation",
    "PartialDirectedCoherence",
    "PhaseLockingValue",
    "Pipeline",
    "ReciprocalConnectivity",
    "RecurrenceMatrix",
    "SampleEntropy",
    "SignalData",
    "SlidingWindow",
    "SlidingWindowReduce",
    "Spectrogram",
    "SpikeCount",
    "SplitterFeature",
    "connectivity",
    "dataset_info",
    "decompositions",
    "delete_dataset",
    "deserialize",
    "download_dataset",
    "feature",
    "from_numpy",
    "from_xarray",
    "get_dataset_dir",
    "gorkastyle",
    "infometrics",
    "list_datasets",
    "load",
    "load_dataset",
    "save",
    "serialization",
    "serialize",
    "set_dataset_dir",
    "show_datasets",
    "signalstats",
    "spectral",
    "surrogates",
    "transforms",
    "windowing",
]
