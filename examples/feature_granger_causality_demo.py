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


data = cb.SignalData.from_numpy(
    _create_neuronal_signal(n_samples=200), dims=["time", "space"], sampling_rate=100.0
)

print("Data shape:", data.data.shape)
print("Background: Two coupled neurons, A drives B (feedforward circuit)")
print()

print("=" * 60, "Single test", "=" * 60)
# GrangerCausality returns a (space_to, space_from) matrix; select a single
# direction with .sel(space_to=target, space_from=source). A is channel 0,
# B is channel 1, so A -> B is space_to=1, space_from=0.
m2 = cb.GrangerCausality(lag=2).apply(data)
gc_a_to_b = m2.data.sel(space_to=1, space_from=0)
gc_b_to_a = m2.data.sel(space_to=0, space_from=1)
print(f"A -> B: GC = {gc_a_to_b.values.item():.6f} (strong causality)")
print(f"B -> A: GC = {gc_b_to_a.values.item():.6f} (weak coupling)")

print()
print("=" * 60, "Multiple lags", "=" * 60)
# maxlag adds a lag_index dimension; select the A -> B direction across lags.
r_multi = cb.GrangerCausality(maxlag=4).apply(data)
print(f"Lags 1-4 GC: {r_multi.data.sel(space_to=1, space_from=0).values}")

print()
print("=" * 60, "Matrix", "=" * 60)
m = cb.GrangerCausality(coords=[0, 1], lag=2).apply(data)
print("Causality matrix (rows=space_to, cols=space_from):")
print(m.data.values)
a_to_b = float(m.data.sel(space_to=1, space_from=0))
b_to_a = float(m.data.sel(space_to=0, space_from=1))
print(f"Directed: A->B={a_to_b:.4f}, B->A={b_to_a:.4f}")
print(f"Ratio: {a_to_b / b_to_a:.0f}x stronger from A to B")

print()
print("=" * 60, "Interpretation", "=" * 60)
print("GC > 0: past values improve prediction (indicates causality)")
print(f"GC > 1.0: strong causality (A->B = {gc_a_to_b.values.item():.2f})")
print("=" * 60)
