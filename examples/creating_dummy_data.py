"""EEG example. From project root: python examples/creating_dummy_data.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cobrabox as cb

rng = np.random.default_rng(42)
arr = rng.standard_normal((1000, 16))
base = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0, subjectID="sub-001")
eeg = cb.EEG(base.data, sampling_rate=base.sampling_rate, subjectID=base.subjectID)

wdata = cb.feature.sliding_window(eeg)
feat = cb.feature.line_length(wdata)

print("input type:", type(eeg).__name__)
print("feature.data.shape:", feat.data.shape)
print("feature.history:", feat.history)
