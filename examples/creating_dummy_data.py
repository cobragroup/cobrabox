"""EEG example. From project root: python examples/creating_dummy_data.py"""

import numpy as np

import cobrabox as cb

rng = np.random.default_rng(42)
arr = rng.standard_normal((1000, 16))
base = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0, subjectID="sub-001")
eeg = cb.EEG(base.data, sampling_rate=base.sampling_rate, subjectID=base.subjectID)

feat = (cb.feature.SlidingWindow() | cb.feature.LineLength() | cb.feature.MeanAggregate()).apply(
    eeg
)

print("input type:", type(eeg).__name__)
print("feature.data.shape:", feat.data.shape)
print("feature.history:", feat.history)
