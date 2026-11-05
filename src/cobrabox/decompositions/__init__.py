"""Decompositions: How do I decompose my signal into components?

Methods that split a signal into constituent parts, such as empirical mode
decomposition, singular value decomposition and frequency-band
decomposition.

Every feature here comes in two forms, both also exported at the top level:
the class (``cb.SVD()``), which composes into pipelines with ``|``, and a
one-shot function (``cb.svd(data)``). ``cb.decompositions.__all__`` lists
them, and ``help()`` on any of them gives its parameters.
"""

from ._bandpass_filter import BandpassFilter, bandpass_filter
from ._emd import EMD, emd
from ._svd import SVD, svd

__all__ = ["EMD", "SVD", "BandpassFilter", "bandpass_filter", "emd", "svd"]
