"""Minimal example demonstrating the partial correlation feature."""

import numpy as np

import cobrabox as cb

rng = np.random.default_rng(42)

data = cb.SignalData.from_numpy(
    rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
)

print("Data shape:", data.data.shape)
print("Space coordinates:", list(data.data.coords["space"].values))
print()

print("=" * 60)
print("Example 1: Single partial correlation")
print("=" * 60)
# PartialCorrelation always returns a (space_to, space_from) matrix; restrict the
# output to the pair with coords= and condition on channel 2 via control_vars=.
r = cb.feature.PartialCorrelation(coords=[0, 1], control_vars=[2]).apply(data)
pair = r.data.sel(space_to=0, space_from=1)
print(f"Partial correlation (electrode 0 vs 1, controlling for 2): {pair.values.item():.4f}")
print(f"History: {r.history}")
print()

print("=" * 60)
print("Example 2: Partial correlation matrix")
print("=" * 60)
m = cb.feature.PartialCorrelation(coords=[0, 1, 2], control_vars=[3]).apply(data)
print("Pairwise partial correlations (controlling for electrode 3):")
print(m.data.values)
print()

print("=" * 60)
print("Example 3: Default coordinates (full matrix over all channels)")
print("=" * 60)
m_all = cb.feature.PartialCorrelation().apply(data)
print("Pairwise partial correlations (all coordinates condition on each other):")
print(m_all.data.values)
print(f"Shape: {m_all.data.shape}")
print(f"Coordinates used: {list(m_all.data.coords['space_to'].values)}")
print()

print("=" * 60)
print("Example 4: Validation errors")
print("=" * 60)

print("\nInvalid coordinate:")
try:
    cb.feature.PartialCorrelation(coords=[99, 1], control_vars=[2]).apply(data)
except ValueError as e:
    print(f"  Error: {e}")

print("\nEmpty coords:")
try:
    cb.feature.PartialCorrelation(coords=[], control_vars=[3]).apply(data)
except ValueError as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
