"""Minimal example demonstrating phase locking value feature."""

import numpy as np

import cobrabox as cb

rng = np.random.default_rng(42)

data = cb.from_numpy(rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

print("Data shape:", data.data.shape)
print("Space coordinates:", list(data.data.coords["space"].values))
print()

print("=" * 60)
print("Example 1: Single phase locking value")
print("=" * 60)
r = cb.feature.phase_locking_value(data, 0, 1)
print(f"Phase locking value (electrode 0 vs 1): {r.data.values.item():.4f}")
print(f"History: {r.history}")
print()

print("=" * 60)
print("Example 2: Phase locking value matrix")
print("=" * 60)
m = cb.feature.phase_locking_value_matrix(data, [0, 1, 2])
print("Pairwise phase locking value:")
print(m.data.values[0, :, :, 0])
print()

print("=" * 60)
print("Example 3: Validation errors")
print("=" * 60)


print("\nInvalid coordinate:")
try:
    cb.feature.phase_locking_value(data, 99, 1)
except ValueError as e:
    print(f"  Error: {e}")

print("\nEmpty coords in matrix:")
try:
    cb.feature.phase_locking_value_matrix(data, [])
except ValueError as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
