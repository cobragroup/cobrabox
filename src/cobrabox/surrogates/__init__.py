"""Surrogates: How do I test statistical significance?

Generators of surrogate data that keep chosen properties of a signal (for
example its power spectrum) while destroying others. Compute a feature on
many surrogates to build the null distribution for the real value.

Every feature here comes in two forms, both also exported at the top level:
the class (``cb.FourierTransformSurrogates()``), which composes into pipelines with ``|``, and a
one-shot function (``cb.fourier_transform_surrogates(data)``). ``cb.surrogates.__all__`` lists
them, and ``help()`` on any of them gives its parameters.
"""

from ._fourier_transform_surrogates import FourierTransformSurrogates, fourier_transform_surrogates

__all__ = ["FourierTransformSurrogates", "fourier_transform_surrogates"]
