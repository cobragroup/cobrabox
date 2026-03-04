from __future__ import annotations

from dataclasses import dataclass

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Dummy(BaseFeature):
    """Return a reconstruction of the input data (negative reference — do not copy).

    This feature exists as a negative reference showing what a poorly written
    feature looks like. The implementation round-trips the data through numpy,
    preserving only the array values, dimension names, and sampling rate; all
    other metadata (subjectID, groupID, condition, extra, coordinates) is
    dropped. Do not use this as a template for new features.

    Args:
        mandatory_arg: Required integer argument (unused in computation).
        optional_arg: Optional integer argument (unused in computation).

    Returns:
        xarray DataArray with the same shape and dimension names as the input
        but without coordinate labels or other metadata. Values are identical
        to the input signal.

    Example:
        >>> result = cb.feature.Dummy(mandatory_arg=1).apply(data)
    """

    mandatory_arg: int
    optional_arg: int = 0

    def __call__(self, data: Data) -> Data:
        if "time" not in data.data.dims:
            raise ValueError("data must have 'time' dimension")
        if "space" not in data.data.dims:
            raise ValueError("data must have 'space' dimension")
        a = data.to_numpy()
        sampling_rate = data.sampling_rate
        return Data.from_numpy(a, dims=list(data.data.dims), sampling_rate=sampling_rate)
