"""Minimal example demonstrating Fourier transform surrogates feature.

FourierTransformSurrogates generates surrogate time series that preserve the
power spectrum (autocorrelation) while randomizing the phases. This is useful
for statistical testing and null hypothesis generation.
"""

import numpy as np

import cobrabox as cb

rng = np.random.default_rng(42)

# Create sample time-series data
# Shape: (timepoints, channels)
arr = rng.normal(size=(200, 4))
data = cb.data.SignalData.from_numpy(
    arr, dims=["time", "space"], sampling_rate=100.0, subjectID="sub-01"
)

print("Data shape:", data.data.shape)
print(f"Original data range: [{data.to_numpy().min():.3f}, {data.to_numpy().max():.3f}]")
print()

print("=" * 60)
print("Example 1: Generate 5 surrogates")
print("=" * 60)
feat = cb.feature.FourierTransformSurrogates(n_surrogates=5, random_state=42)
surrogates = list(feat(data))
print(f"Generated {len(surrogates)} items (original + 5 surrogates)")
print(f"First item is original data: {np.array_equal(surrogates[0].to_numpy(), data.to_numpy())}")
print()

print("=" * 60)
print("Example 2: Surrogates preserve power spectrum")
print("=" * 60)
# Generate one surrogate for comparison
feat_single = cb.feature.FourierTransformSurrogates(
    n_surrogates=1, random_state=42, return_data=False
)
surrogate = next(feat_single(data))

# Compare power spectra of the same channel (should be similar)
# Data is stored as (space, time), so channel is first dimension
channel_idx = 0
orig_power = np.abs(np.fft.rfft(data.to_numpy()[channel_idx, :])) ** 2
surrogate_power = np.abs(np.fft.rfft(surrogate.to_numpy()[channel_idx, :])) ** 2
correlation = np.corrcoef(orig_power, surrogate_power)[0, 1]
print(f"Power spectrum correlation for channel {channel_idx}: {correlation:.4f}")
print("  (Values > 0.99 indicate good preservation)")
print()

print("=" * 60)
print("Example 3: Multivariate vs Independent mode")
print("=" * 60)
# Create correlated data with shape (time, space)
n_samples = 1000
correlated_arr = np.column_stack(
    [
        rng.normal(size=n_samples),
        rng.normal(size=n_samples) * 0.3 + rng.normal(size=n_samples) * 0.7,
    ]
)
correlated_data = cb.data.SignalData.from_numpy(
    correlated_arr, dims=["time", "space"], sampling_rate=100.0
)

# Check original correlation between channels (space dimension)
orig_vals = correlated_data.to_numpy()  # Shape: (space, time)
orig_cross_corr = np.corrcoef(orig_vals[0, :], orig_vals[1, :])[0, 1]
print(f"Original cross-channel correlation: {orig_cross_corr:.4f}")

# Multivariate mode (preserves cross-channel correlations)
feat_multi = cb.feature.FourierTransformSurrogates(
    n_surrogates=1, multivariate=True, random_state=42, return_data=False
)
surrogate_multi = next(feat_multi(correlated_data))
surrogate_vals_multi = surrogate_multi.to_numpy()
corr_multi = np.corrcoef(surrogate_vals_multi[0, :], surrogate_vals_multi[1, :])[0, 1]
print(f"Multivariate mode cross-channel correlation: {corr_multi:.4f} (preserved)")

# Independent mode (destroys cross-channel correlations)
feat_ind = cb.feature.FourierTransformSurrogates(
    n_surrogates=1, multivariate=False, random_state=42, return_data=False
)
surrogate_ind = next(feat_ind(correlated_data))
surrogate_vals_ind = surrogate_ind.to_numpy()
corr_ind = np.corrcoef(surrogate_vals_ind[0, :], surrogate_vals_ind[1, :])[0, 1]
print(f"Independent mode cross-channel correlation: {corr_ind:.4f} (destroyed)")
print()

print("=" * 60)
print("Example 4: Metadata preservation")
print("=" * 60)
print(f"Original subjectID: {data.subjectID}")
print(f"Surrogate subjectID: {surrogate.subjectID}")
print(f"Original sampling_rate: {data.sampling_rate}")
print(f"Surrogate sampling_rate: {surrogate.sampling_rate}")
print()

print("=" * 60)
print("Example 5: Validation errors")
print("=" * 60)

print("\nNegative n_surrogates:")
try:
    cb.feature.FourierTransformSurrogates(n_surrogates=-1)
except ValueError as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
