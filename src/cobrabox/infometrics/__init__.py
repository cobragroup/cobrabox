"""Infometrics: How complex/irregular is my signal?

Measures of complexity, regularity and information content: entropies,
Lempel-Ziv complexity, fractal dimension, time-reversal asymmetry, and
recurrence analysis.

Every feature here comes in two forms, both also exported at the top level:
the class (``cb.SampleEntropy()``), which composes into pipelines with ``|``, and a
one-shot function (``cb.sample_entropy(data)``). ``cb.infometrics.__all__`` lists
them, and ``help()`` on any of them gives its parameters.
"""

from ._amplitude_entropy import AmplitudeEntropy, amplitude_entropy
from ._fractal_dimension import FractalDimension, fractal_dimension
from ._lempel_ziv import LempelZiv, lempel_ziv
from ._nonreversibility import Nonreversibility, nonreversibility
from ._recurrence_matrix import RecurrenceMatrix, recurrence_matrix
from ._sample_entropy import SampleEntropy, sample_entropy

__all__ = [
    "AmplitudeEntropy",
    "FractalDimension",
    "LempelZiv",
    "Nonreversibility",
    "RecurrenceMatrix",
    "SampleEntropy",
    "amplitude_entropy",
    "fractal_dimension",
    "lempel_ziv",
    "nonreversibility",
    "recurrence_matrix",
    "sample_entropy",
]
