# Decomposition features: break a signal into constituent components.
from ._emd import EMD
from ._svd import SVD

__all__ = ["EMD", "SVD"]

# One-shot functional wrappers (cb.line_length(d) beside cb.LineLength().apply(d)). See GH #116.
from .._functional import install as _install_functional

_install_functional(__name__)
