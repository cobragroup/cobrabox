"""Minimal example. From project root: python examples/feature_pipeline_demo.py"""

import cobrabox as cb

# cb.gorkastyle()

data = cb.dataset("dummy_chain")[0]
wdata = cb.feature.sliding_window(data, window_size=10, step_size=5)
win_min = cb.feature.min(wdata, dim="window_index")
win_max = cb.feature.max(wdata, dim="window_index")
feat = cb.feature.line_length(data)
dummy = cb.feature.dummy(feat, mandatory_arg=1, optional_arg=2)

print("min over windows shape:", win_min.data.shape)
print("min over windows history:", win_min.history)
print("max over windows shape:", win_max.data.shape)
print("max over windows history:", win_max.history)
print("feature.data.shape:", feat.data.shape)
print("feature.history:", feat.history)

print("dummy", dummy.data)
