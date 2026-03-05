"""Granger causality: tests if one time-series predicts another."""

import numpy as np

import cobrabox as cb


def _create_neuronal_signal(n_samples: int = 200) -> np.ndarray:
    """Create 2-channel signal where Neuron A causes Neuron B."""
    neuron_a = np.zeros(n_samples)
    neuron_b = np.zeros(n_samples)
    rng = np.random.default_rng(42)

    for t in range(n_samples):
        noise_a = rng.normal(0, 0.2)
        noise_b = rng.normal(0, 0.15)
        if t == 0:
            neuron_a[t] = 0.05 + noise_a
            neuron_b[t] = 0.05 + noise_b
        else:
            external_drive = 3.0 * np.sin(0.25 * (t - 30)) if 30 <= t < 100 else 0
            neuron_a[t] = 0.3 * neuron_a[t - 1] + external_drive + noise_a
            neuron_b[t] = 0.8 * neuron_a[t - 1] + 0.15 * noise_b

    return np.column_stack([neuron_a, neuron_b])


data = cb.data.SignalData.from_numpy(
    _create_neuronal_signal(n_samples=200), dims=["time", "space"], sampling_rate=100.0
)

print("Data shape:", data.data.shape)
print("Background: Two coupled neurons, A drives B (feedforward circuit)")
print()

print("=" * 60, "Single test", "=" * 60)
r_a_to_b = cb.feature.GrangerCausality(coord_x=1, coord_y=0, lag=2).apply(data)
r_b_to_a = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2).apply(data)
print(f"A → B: GC = {r_a_to_b.data.values.item():.6f} (strong causality)")
print(f"B → A: GC = {r_b_to_a.data.values.item():.6f} (weak coupling)")

print()
print("=" * 60, "Multiple lags", "=" * 60)
r_multi = cb.feature.GrangerCausality(coord_x=1, coord_y=0, maxlag=4).apply(data)
print(f"Lags 1-4 GC: {r_multi.data.values}")

print()
print("=" * 60, "Matrix", "=" * 60)
m = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)
print("Causality matrix:")
print(m.data.values)
print(f"Diagonal: A→B={m.data.values[1, 0]:.4f}, B→A={m.data.values[0, 1]:.4f}")
print(f"Ratio: {m.data.values[1, 0] / m.data.values[0, 1]:.0f}x stronger from A to B")

print()
print("=" * 60, "Interpretation", "=" * 60)
print("GC > 0: past values improve prediction (indicates causality)")
print(f"GC > 1.0: strong causality (A→B = {r_a_to_b.data.values.item():.2f})")
print("=" * 60)
