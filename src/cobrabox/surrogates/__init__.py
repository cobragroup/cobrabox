# Surrogate features: statistical null-hypothesis methods.
from ._fourier_transform_surrogates import FourierTransformSurrogates

__all__ = ["FourierTransformSurrogates"]

# One-shot functional wrappers (cb.line_length(d) beside cb.LineLength().apply(d)). See GH #116.
from .._functional import install as _install_functional

_install_functional(__name__)
