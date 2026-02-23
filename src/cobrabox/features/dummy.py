from __future__ import annotations

from ..data import Dataset
from ..function_wrapper import feature


@feature
def dummy(
    data: Dataset,
) -> Dataset:
    """Dummy feature that returns the input data."""
    a = data.asnumpy()
    sampling_rate = data.sampling_rate
    whatever = data.extra.get("whatever", None)

    return Dataset.from_numpy(a, dims=data.data.dims, sampling_rate=sampling_rate)
