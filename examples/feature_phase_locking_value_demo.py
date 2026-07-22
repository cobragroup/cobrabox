"""Minimal example demonstrating the phase locking value feature."""

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
print("Example 1: Single phase locking value")
print("=" * 60)
# PhaseLockingValue always returns a (space_to, space_from) matrix; restrict it
# to the pair of interest with coords=, then select the single value.
r = cb.PhaseLockingValue(coords=[0, 1]).apply(data)
pair = r.data.sel(space_to=0, space_from=1)
print(f"Phase locking value (electrode 0 vs 1): {pair.values.item():.4f}")
print(f"History: {r.history}")
print()

print("=" * 60)
print("Example 2: Phase locking value matrix")
print("=" * 60)
m = cb.PhaseLockingValue(coords=[0, 1, 2]).apply(data)
print("Pairwise phase locking value:")
print(m.data.values)
print()

print("=" * 60)
print("Example 3: Default coordinates (all space coordinates)")
print("=" * 60)
m_all = cb.PhaseLockingValue().apply(data)
print("Pairwise phase locking value (all coordinates):")
print(m_all.data.values)
print(f"Shape: {m_all.data.shape}")
print(f"Coordinates used: {list(m_all.data.coords['space_to'].values)}")
print()

print("=" * 60)
print("Example 4: Validation errors")
print("=" * 60)

print("\nInvalid coordinate:")
try:
    cb.PhaseLockingValue(coords=[99, 1]).apply(data)
except ValueError as e:
    print(f"  Error: {e}")

print("\nEmpty coords:")
try:
    cb.PhaseLockingValue(coords=[]).apply(data)
except ValueError as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
