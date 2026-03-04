"""Minimal example. From project root: python examples/feature_pipeline_demo.py"""

import cobrabox as cb

# cb.gorkastyle()

data = cb.dataset("dummy_chain")[0]

pipeline = cb.feature.SlidingWindow(window_size=10, step_size=5) | cb.feature.Min(
    dim="window_index"
)
win_min = pipeline.apply(data)

win_max = (
    cb.feature.SlidingWindow(window_size=10, step_size=5) | cb.feature.Max(dim="window_index")
).apply(data)
feat = cb.feature.LineLength().apply(data)
dummy = cb.feature.Dummy(mandatory_arg=1, optional_arg=2).apply(feat)

# Compute coherence
coh = cb.feature.coherence(data)

print("min over windows shape:", win_min.data.shape)
print("min over windows history:", win_min.history)
print("max over windows shape:", win_max.data.shape)
print("max over windows history:", win_max.history)
print("feature.data.shape:", feat.data.shape)
print("feature.history:", feat.history)
print("coherence shape:", coh.data.shape)
print("coherence history:", coh.history)

print("dummy", dummy.data)
