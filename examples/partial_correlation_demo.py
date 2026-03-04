"""Minimal example demonstrating partial correlation features."""

import numpy as np

import cobrabox as cb

rng = np.random.default_rng(42)

data = cb.from_numpy(rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

print("Data shape:", data.data.shape)
print("Space coordinates:", list(data.data.coords["space"].values))
print()

print("=" * 60)
print("Example 1: Single partial correlation")
print("=" * 60)
r = cb.feature.partial_correlation(data, 0, 1, control_vars=[2])
print(f"Partial correlation (electrode 0 vs 1, controlling for 2): {r.data.values.item():.4f}")
print(f"History: {r.history}")
print()

print("=" * 60)
print("Example 2: Partial correlation matrix")
print("=" * 60)
m = cb.feature.partial_correlation_matrix(data, [0, 1, 2], control_vars=[3])
print("Pairwise partial correlations (controlling for electrode 3):")
print(m.data.values[0, :, :, 0])
print()

print("=" * 60)
print("Example 3: Validation errors")
print("=" * 60)

print("Empty control_vars:")
try:
    cb.feature.partial_correlation(data, 0, 1, control_vars=[])
except ValueError as e:
    print(f"  Error: {e}")

print("\nInvalid coordinate:")
try:
    cb.feature.partial_correlation(data, 99, 1, control_vars=[2])
except ValueError as e:
    print(f"  Error: {e}")

print("\nEmpty coords in matrix:")
try:
    cb.feature.partial_correlation_matrix(data, [], control_vars=[3])
except ValueError as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
