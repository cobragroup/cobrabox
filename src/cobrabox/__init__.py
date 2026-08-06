from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("cobrabox")
except PackageNotFoundError:
    __version__ = "unknown"

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

# Every feature is re-exported at the root namespace in both forms: the class
# `cb.Correlation()` for composition and serialization, and the one-shot function
# `cb.correlation(d)` for a single call. `cb.<domain>.X` and `cb.X` remain
# valid aliases. `tests/test_public_api.py` fails if these drift apart.
from .connectivity import (
    Coherence,
    Correlation,
    Covariance,
    DirectDirectedTransferFunction,
    DirectedTransferFunction,
    EnvelopeCorrelation,
    GrangerCausality,
    MutualInformation,
    PartialCorrelation,
    PartialDirectedCoherence,
    PhaseLockingValue,
    ReciprocalConnectivity,
    coherence,
    correlation,
    covariance,
    direct_directed_transfer_function,
    directed_transfer_function,
    envelope_correlation,
    granger_causality,
    mutual_information,
    partial_correlation,
    partial_directed_coherence,
    phase_locking_value,
    reciprocal_connectivity,
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
from .decompositions import EMD, SVD, BandpassFilter, bandpass_filter, emd, svd
from .downloader import DownloadCancelled, LargeLoadError, get_dataset_dir, set_dataset_dir
from .egg.gorkastyle import gorkastyle
from .infometrics import (
    AmplitudeEntropy,
    FractalDimension,
    LempelZiv,
    Nonreversibility,
    RecurrenceMatrix,
    SampleEntropy,
    amplitude_entropy,
    fractal_dimension,
    lempel_ziv,
    nonreversibility,
    recurrence_matrix,
    sample_entropy,
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
    amplitude_variation,
    autocorrelation,
    epileptogenicity_index,
    line_length,
    max,
    mean,
    min,
    spike_count,
)
from .spectral import (
    BandPower,
    ContinuousWaveletTransform,
    Cordance,
    DiscreteWaveletTransform,
    PowerSpectralDensity,
    Spectrogram,
    band_power,
    continuous_wavelet_transform,
    cordance,
    discrete_wavelet_transform,
    power_spectral_density,
    spectrogram,
)
from .surrogates import FourierTransformSurrogates, fourier_transform_surrogates
from .transforms import (
    AnalyticSignal,
    FourierTransform,
    InverseFourierTransform,
    NotchFilter,
    analytic_signal,
    fourier_transform,
    inverse_fourier_transform,
    notch_filter,
)
from .windowing import (
    ConcatAggregate,
    MeanAggregate,
    SlidingWindow,
    SlidingWindowReduce,
    sliding_window,
    sliding_window_reduce,
)

# Package-level aliases for class methods
from_numpy = Data.from_numpy
from_xarray = Data.from_xarray

# Importing a name out of a submodule also binds that submodule here, so `cb.data`,
# `cb.dataset` and friends leak implementation modules into the public namespace.
# Feature modules solve this by being private (`_line_length.py`); these top-level
# modules cannot be renamed as freely, so drop the attributes explicitly. The
# modules stay in `sys.modules`, so `from cobrabox.data import Data` and
# `importlib.import_module` are unaffected — only the `cb.<module>` shortcut goes.
#
# `cb.dataset` is the one with teeth: it made `cb.dataset("dummy_chain")` fail with
# "'module' object is not callable", which reads like a broken function rather than
# a name that never existed. The loader is `cb.load_dataset`.
for _leaked in (
    "base_feature",
    "data",
    "dataset",
    "dataset_loader",
    "datasets",
    "downloader",
    "egg",
):
    globals().pop(_leaked, None)
del _leaked

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
    "DirectDirectedTransferFunction",
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
    "LargeLoadError",
    "LempelZiv",
    "LineLength",
    "Max",
    "Mean",
    "MeanAggregate",
    "Min",
    "MutualInformation",
    "Nonreversibility",
    "NotchFilter",
    "PartialCorrelation",
    "PartialDirectedCoherence",
    "PhaseLockingValue",
    "Pipeline",
    "PowerSpectralDensity",
    "ReciprocalConnectivity",
    "RecurrenceMatrix",
    "SampleEntropy",
    "SignalData",
    "SlidingWindow",
    "SlidingWindowReduce",
    "Spectrogram",
    "SpikeCount",
    "SplitterFeature",
    "__version__",
    "amplitude_entropy",
    "amplitude_variation",
    "analytic_signal",
    "autocorrelation",
    "band_power",
    "bandpass_filter",
    "coherence",
    "connectivity",
    "continuous_wavelet_transform",
    "cordance",
    "correlation",
    "covariance",
    "dataset_info",
    "decompositions",
    "delete_dataset",
    "deserialize",
    "direct_directed_transfer_function",
    "directed_transfer_function",
    "discrete_wavelet_transform",
    "download_dataset",
    "emd",
    "envelope_correlation",
    "epileptogenicity_index",
    "feature",
    "fourier_transform",
    "fourier_transform_surrogates",
    "fractal_dimension",
    "from_numpy",
    "from_xarray",
    "get_dataset_dir",
    "gorkastyle",
    "granger_causality",
    "infometrics",
    "inverse_fourier_transform",
    "lempel_ziv",
    "line_length",
    "list_datasets",
    "load",
    "load_dataset",
    "max",
    "mean",
    "min",
    "mutual_information",
    "nonreversibility",
    "notch_filter",
    "partial_correlation",
    "partial_directed_coherence",
    "phase_locking_value",
    "power_spectral_density",
    "reciprocal_connectivity",
    "recurrence_matrix",
    "sample_entropy",
    "save",
    "serialization",
    "serialize",
    "set_dataset_dir",
    "show_datasets",
    "signalstats",
    "sliding_window",
    "sliding_window_reduce",
    "spectral",
    "spectrogram",
    "spike_count",
    "surrogates",
    "svd",
    "transforms",
    "windowing",
]
