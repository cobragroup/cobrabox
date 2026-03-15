"""Lempel-Ziv Complexity feature.

The core counting algorithm (_count) is adapted from NeuroKit2
(https://github.com/neuropsychology/NeuroKit, MIT licence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ...base_feature import BaseFeature
from ...data import Data, SignalData


@dataclass
class LempelZiv(BaseFeature[SignalData]):
    """Compute Lempel-Ziv Complexity (LZC) over the time dimension.

    The signal is binarised relative to its mean (values above the mean → 1,
    values at or below → 0). LZC counts distinct patterns in the resulting
    binary sequence and normalises by the theoretical maximum, giving a value
    in (0, 1] where higher values indicate greater irregularity.

    The counting algorithm follows Lempel & Ziv (1976) as implemented in
    NeuroKit2 (https://github.com/neuropsychology/NeuroKit, MIT licence).

    Args:
        None

    Returns:
        xarray DataArray with the ``time`` dimension removed. Shape is
        ``(space,)`` for standard input. Values are dimensionless floats in
        (0, 1].

    References:
        Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences.
        IEEE Transactions on Information Theory, 22(1), 75-81.

    Example:
        >>> result = cb.feature.LempelZiv().apply(data)
    """

    output_type: ClassVar[type[Data]] = Data  # pyright: ignore[reportIncompatibleVariableOverride]

    def __call__(self, data: SignalData) -> xr.DataArray:
        return xr.apply_ufunc(self._lzc_1d, data.data, input_core_dims=[["time"]], vectorize=True)

    @staticmethod
    def _lzc_1d(signal: np.ndarray) -> float:
        """Compute normalised LZC for a single 1-D signal (binarised by mean)."""
        symbolic = (signal > signal.mean()).astype(int)
        c, n = LempelZiv._count(symbolic)
        return float((c * math.log2(n)) / n)

    @staticmethod
    def _count(symbolic: np.ndarray) -> tuple[int, int]:
        """Count LZ76 complexity tokens for a 1-D symbolic sequence.

        Adapted from NeuroKit2 (MIT licence):
        https://github.com/neuropsychology/NeuroKit
        """
        string = "".join(symbolic.astype(int).astype(str).tolist())
        n = len(string)
        s = "0" + string
        c = 1
        j = 1
        i = 0
        k = 1
        k_max = 1
        stop = False
        while not stop:
            if s[i + k] != s[j + k]:
                k_max = max(k_max, k)
                i += 1
                if i == j:
                    c += 1
                    j += k_max
                    if j + 1 > n:
                        stop = True
                    else:
                        i = 0
                        k = 1
                        k_max = 1
                else:
                    k = 1
            else:
                k += 1
                if j + k > n:
                    c += 1
                    stop = True
        return c, n
