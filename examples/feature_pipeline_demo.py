"""Minimal example. From project root: python examples/feature_pipeline_demo.py"""

import cobrabox as cb

# cb.gorkastyle()

data = cb.dataset("dummy_chain")[0]

# Simple pipeline using | syntax
pipeline = cb.feature.Min(dim="time") | cb.feature.Max(dim="time")
simple_out = pipeline.apply(data)
print("simple pipeline history:", simple_out.history)
print("simple pipeline shape:", simple_out.data.shape)

win_min = (
    cb.feature.SlidingWindow(window_size=10, step_size=5)
    | cb.feature.Min(dim="time")
    | cb.feature.MeanAggregate()
).apply(data)

win_max = (
    cb.feature.SlidingWindow(window_size=10, step_size=5)
    | cb.feature.Max(dim="time")
    | cb.feature.MeanAggregate()
).apply(data)

feat = cb.feature.LineLength().apply(data)
dummy = cb.feature.Dummy(mandatory_arg=1, optional_arg=2).apply(data)

# Compute coherence
coh = cb.feature.Coherence().apply(data)

# SlidingWindowReduce — simpler alternative to Chord for basic windowed stats
win_reduce = cb.feature.SlidingWindowReduce(
    window_size=10, step_size=5, dim="time", agg="mean"
).apply(data)

print("min over windows shape:", win_min.data.shape)
print("min over windows history:", win_min.history)
print("max over windows shape:", win_max.data.shape)
print("max over windows history:", win_max.history)
print("feature.data.shape:", feat.data.shape)
print("feature.history:", feat.history)
print("coherence shape:", coh.data.shape)
print("coherence history:", coh.history)
print("SlidingWindowReduce shape:", win_reduce.data.shape)
print("SlidingWindowReduce history:", win_reduce.history)

print("dummy", dummy.data)
