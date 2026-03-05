"""Minimal example demonstrating partial correlation features."""

import numpy as np

import cobrabox as cb

rng = np.random.default_rng(42)

data = cb.data.SignalData.from_numpy(
    rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
)

print("Data shape:", data.data.shape)
print("Space coordinates:", list(data.data.coords["space"].values))
print()

print("=" * 60)
print("Example 1: Single partial correlation")
print("=" * 60)
r = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)
print(f"Partial correlation (electrode 0 vs 1, controlling for 2): {r.data.values.item():.4f}")
print(f"History: {r.history}")
print()

print("=" * 60)
print("Example 2: Partial correlation matrix")
print("=" * 60)
m = cb.feature.PartialCorrelationMatrix(coords=[0, 1, 2], control_vars=[3]).apply(data)
print("Pairwise partial correlations (controlling for electrode 3):")
print(m.data.values)
print()

print("=" * 60)
print("Example 3: Validation errors")
print("=" * 60)

print("Empty control_vars:")
try:
    cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[]).apply(data)
except ValueError as e:
    print(f"  Error: {e}")

print("\nInvalid coordinate:")
try:
    cb.feature.PartialCorrelation(coord_x=99, coord_y=1, control_vars=[2]).apply(data)
except ValueError as e:
    print(f"  Error: {e}")

print("\nEmpty coords in matrix:")
try:
    cb.feature.PartialCorrelationMatrix(coords=[], control_vars=[3]).apply(data)
except ValueError as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
