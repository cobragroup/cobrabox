from ..data import Data
from ..function_wrapper import feature


@feature
def dummy(data: Data, mandatory_arg: int, optional_arg: int = 0) -> Data:
    """Dummy feature that returns the input data."""
    a = data.to_numpy()
    sampling_rate = data.sampling_rate
    whatever = data.extra.get("whatever", None)

    print("whatever", whatever)

    return Data.from_numpy(a, dims=list(data.data.dims), sampling_rate=sampling_rate)
