from dataclasses import dataclass

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Dummy(BaseFeature):
    """Dummy feature that returns the input data."""

    mandatory_arg: int
    optional_arg: int = 0

    def __call__(self, data: Data) -> Data:
        a = data.to_numpy()
        sampling_rate = data.sampling_rate
        return Data.from_numpy(a, dims=list(data.data.dims), sampling_rate=sampling_rate)
