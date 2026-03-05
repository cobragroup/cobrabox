"""Minimal example demonstrating Granger causality features with neuronal data."""

import numpy as np

import cobrabox as cb


def _create_neuronal_signal(n_samples: int = 200) -> np.ndarray:
    """Create simulated neuronal signal: two coupled oscillator neurons.

    Scenario: Two neurons with coupling. Neuron A receives external drive
    (simulating sensory input) and fires rhythmically. Neuron B receives
    excitatory synaptic input from Neuron A, causing its firing to be
    driven by Neuron A's activity. This creates unidirectional causality.

    Model: Stochastic coupled oscillators
    Neuron A: x(t) = 0.3*x(t-1) + 0.2*noise + I(t)
    Neuron B: y(t) = 0.8*x(t-1) + 0.15*noise

    Where coupling from A→B is very strong (0.8), B→A is absent (0.0)
    """
    neuron_a = np.zeros(n_samples)
    neuron_b = np.zeros(n_samples)

    rng = np.random.default_rng(42)

    coupling_ab = 0.8

    for t in range(n_samples):
        noise_a = rng.normal(0, 0.2)
        noise_b = rng.normal(0, 0.15)

        if t == 0:
            neuron_a[t] = 0.05 + noise_a
            neuron_b[t] = 0.05 + noise_b
        else:
            external_drive = 0.0
            if 30 <= t < 100:
                external_drive = 3.0 * np.sin(0.25 * (t - 30))

            neuron_a[t] = 0.3 * neuron_a[t - 1] + external_drive + noise_a
            neuron_b[t] = coupling_ab * neuron_a[t - 1] + 0.15 * noise_b

    return np.column_stack([neuron_a, neuron_b])


data = cb.data.SignalData.from_numpy(
    _create_neuronal_signal(n_samples=200), dims=["time", "space"], sampling_rate=100.0
)

print("Data shape:", data.data.shape)
print(f"Space coordinates: {list(data.data.coords['space'].values)}")
print("(0: Neuron A, 1: Neuron B)")
print()

print("=" * 60)
print("Background: Neural Circuit Causal Relationship")
print("=" * 60)
print("Scenario: Two coupled neurons with directional synaptic connection")
print()
print("Biological model:")
print("  - Neuron A: Receives external sensory input (driven oscillator)")
print("  - Neuron B: Receives strong excitatory synaptic input from Neuron A")
print("  - Coupling: A → B is strong (w_AB = 0.8), B → A is absent (w_BA = 0)")
print()
print("Simulated timeline:")
print("  - t=0-30: Baseline activity (low-frequency oscillation)")
print("  - t=30-100: Sensory stimulus increases (higher-frequency oscillation)")
print("  - t=100-200: Return to baseline")
print()
print("Causal mechanism: Neuron A's firing pattern drives Neuron B's")
print("membrane potential through synaptic transmission. Neuron B does")
print("NOT send feedback to Neuron A (feedforward circuit architecture).")
print()

print("=" * 60)
print("Example 1: Single Granger causality test")
print("=" * 60)

print("\nTest if Neuron A causes Neuron B:")
r_a_to_b = cb.feature.GrangerCausality(coord_x=1, coord_y=0, lag=2).apply(data)
print(f"  Granger causality: {r_a_to_b.data.values.item():.6f}")
if r_a_to_b.data.values.item() > 0:
    print("  Result: Significant causality detected")
    print("  Interpretation: Neuron A Granger-causes Neuron B activity")
    print("  (Positive value = reduction in prediction error variance)")
else:
    print("  Result: No significant causality")

print("\nTest if Neuron B causes Neuron A (expected to be weaker):")
r_b_to_a = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2).apply(data)
print(f"  Granger causality: {r_b_to_a.data.values.item():.6f}")
ratio = r_a_to_b.data.values.item() / max(1e-10, r_b_to_a.data.values.item())
print(f"  Ratio A→B / B→A: {ratio:.1f}x")
print("  Interpretation: Much stronger coupling from A to B than B to A")

print(f"\nHistory: {r_a_to_b.history}")
print()

print("=" * 60)
print("Example 2: Test multiple lags")
print("=" * 60)
r_multi = cb.feature.GrangerCausality(coord_x=1, coord_y=0, maxlag=4).apply(data)
print("Testing lags 1-4 for causality from Neuron A to Neuron B:")
for _i, (lag, gc) in enumerate(
    zip(r_multi.data.coords["lag_index"].values, r_multi.data.values, strict=True)
):
    print(f"  lag={lag}: GC = {gc:.6f}", end="")
    if gc > 0.01:
        print(" (significant)")
    else:
        print()
print()
print("Lag interpretation: Shows how synaptic transmission delays persist")
print("across different time horizons (single-spike vs temporal integration)")
print()

print("=" * 60)
print("Example 3: Granger causality matrix")
print("=" * 60)
m = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)
print("Granger causality matrix (log-ratio method):")
print("  Rows: target neuron, Columns: source neuron")
print("  result[i,j] = causality from j to i")
print("\n  Variable mapping: 0=Neuron A, 1=Neuron B")
print(f"\n{m.data.values}")
print("\nMatrix interpretation:")
print(f"  Neuron A → Neuron B (row 1, col 0): GC = {m.data.values[1, 0]:.6f}")
if m.data.values[1, 0] > 1.0:
    print("    → Very significant: Strong synaptic coupling from A to B")
print(f" Neuron B → Neuron A (row 0, col 1): GC = {m.data.values[0, 1]:.6f}")
directionality_ratio = m.data.values[1, 0] / max(1e-10, m.data.values[0, 1])
print(f"    → Directionality ratio: {directionality_ratio:.0f}x")
print()

print("=" * 60)
print("Example 4: Interpret Granger causality values")
print("=" * 60)
print("Granger causality values (log-ratios) directly quantify coupling strength:")
print(f"  Neuron A → Neuron B: GC = {r_a_to_b.data.values.item():.4f}")
print(f"  Neuron B → Neuron A: GC = {r_b_to_a.data.values.item():.4f}")
print()
print("Interpretation guidelines:")
print("  - GC > 0: Adding past values improves prediction (causality)")
print("  - GC > 0.5-1.0: Moderate causality")
print("  - GC > 2.0+: Strong causality")
print("  - Higher values = stronger predictive improvement")
coupling_ratio = r_a_to_b.data.values.item() / r_b_to_a.data.values.item()
print(f"  - {coupling_ratio:.0f}x stronger coupling in forward direction")
print()

print("=" * 60)
print("All examples completed successfully!")
print("=" * 60)
