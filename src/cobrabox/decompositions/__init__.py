# Decomposition features: break a signal into constituent components.
from ._emd import EMD, emd
from ._svd import SVD, svd

__all__ = ["EMD", "SVD", "emd", "svd"]
