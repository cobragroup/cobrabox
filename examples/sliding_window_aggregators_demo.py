"""Showcase MeanAggregate vs ConcatAggregate with SlidingWindow.

Run from the project root:
    uv run python examples/sliding_window_aggregators_demo.py

Key difference:
  - MeanAggregate  reduces all windows to a single result (same shape as one window)
  - ConcatAggregate preserves every window as a slice along a new 'window' dimension
"""

import numpy as np

import cobrabox as cb

# A small synthetic recording: 5 channels, 10 s at 100 Hz. For real-looking
# data, use cb.load_dataset("swiss_demo") instead (downloaded on first use).
rng = np.random.default_rng(0)
data = cb.SignalData.from_numpy(
    rng.standard_normal((1000, 5)).cumsum(axis=0),
    dims=["time", "space"],
    sampling_rate=100.0,
    subjectID="sub-01",
)
print(f"Input shape:  {dict(data.data.sizes)}")
print(f"Input dims:   {data.data.dims}")
print()

# A simple per-window feature: line length collapses the time axis
feature = cb.LineLength()

# --- MeanAggregate: averages results across windows ---
mean_result = (cb.SlidingWindow(window_size=50, step_size=25) | feature | cb.MeanAggregate()).apply(
    data
)

print("=== MeanAggregate ===")
print(f"  Shape:   {dict(mean_result.data.sizes)}")
print(f"  Dims:    {mean_result.data.dims}")
print(f"  History: {mean_result.history}")
print()

# --- ConcatAggregate: stacks all window results along a new 'window' dimension ---
concat_result = (
    cb.SlidingWindow(window_size=50, step_size=25) | feature | cb.ConcatAggregate()
).apply(data)

print("=== ConcatAggregate ===")
print(f"  Shape:   {dict(concat_result.data.sizes)}")
print(f"  Dims:    {concat_result.data.dims}")
print(f"  History: {concat_result.history}")
print()

# Inspect individual windows from ConcatAggregate
n_windows = concat_result.data.sizes["window"]
print(f"Number of windows retained: {n_windows}")
print(f"Window 0 values (first channel): {concat_result.data.isel(window=0).values[:5]} ...")
print(f"Window 1 values (first channel): {concat_result.data.isel(window=1).values[:5]} ...")
