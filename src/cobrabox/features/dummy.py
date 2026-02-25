from __future__ import annotations

from ..data import Data
from ..function_wrapper import feature


@feature
def dummy(data: Data) -> Data:
    """Dummy feature that returns the input data."""
    a = data.asnumpy()
    sampling_rate = data.sampling_rate
    whatever = data.extra.get("whatever", None)

    print("whatever", whatever)

    return Data.from_numpy(a, dims=list(data.data.dims), sampling_rate=sampling_rate)
