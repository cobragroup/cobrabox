"""Complex example: Linear network with pairwise interactions."""

import numpy as np

import cobrabox as cb


def _linear_network(
    n_nodes: int = 5, n_samples: int = 1000, coupling_strength: float = 0.5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate linear network with directed interactions.

    Each node follows: x_i[t] = sum_j(W[i,j] * x_j[t-1]) + noise_i[t]

    Args:
        n_nodes: Number of nodes
        n_samples: Number of time samples
        coupling_strength: Max spectral radius (stability bound)
        seed: Random seed

    Returns:
        (time_series, weight_matrix) where time_series is (n_nodes, n_samples)
        and weight_matrix[i,j] = coupling from j to i
    """
    rng = np.random.default_rng(seed)

    weight_matrix = rng.uniform(0.1, 0.5, (n_nodes, n_nodes))
    np.fill_diagonal(weight_matrix, 0)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and weight_matrix[i, j] < 0.3:
                weight_matrix[i, j] = 0

    spectral_radius = max(abs(np.linalg.eigvals(weight_matrix)))
    if spectral_radius > coupling_strength:
        weight_matrix *= coupling_strength / spectral_radius

    time_series = np.zeros((n_nodes, n_samples))
    time_series[:, 0] = rng.normal(0, 0.1, n_nodes)

    for t in range(1, n_samples):
        for i in range(n_nodes):
            coupling_term = sum(weight_matrix[i, j] * time_series[j, t - 1] for j in range(n_nodes))
            time_series[i, t] = coupling_term + rng.normal(0, 0.1)

    return time_series, weight_matrix


print("=" * 60, "Linear Network Analysis", "=" * 60)
print(f"Simulating {5} coupled nodes with directed interactions")

time_series, weight_matrix = _linear_network(n_nodes=5, n_samples=1000, coupling_strength=0.5)
time_series = time_series.T
print(f"Generated time series: shape {time_series.shape}")

print()
print("=" * 60, "Ground Truth Interaction Matrix", "=" * 60)
print("Weight matrix W (W[i,j] = coupling from j to i):")
for i in range(5):
    row_str = " ".join([f"{weight_matrix[i, j]:.2f}" for j in range(5)])
    print(f"  {row_str}")

print()
print("Binary threshold (W > 0):")
binary_W = (weight_matrix > 0).astype(int)
for i in range(5):
    row_str = " ".join([f"{binary_W[i, j]}" for j in range(5)])
    print(f"  {row_str}")

print()
print("=" * 60, "Granger Causality Analysis", "=" * 60)
data = cb.data.SignalData.from_numpy(time_series, dims=["time", "space"], sampling_rate=100.0)

gc_matrix = cb.feature.GrangerCausalityMatrix(lag=2).apply(data)
print("Granger causality matrix (lag=2):")
for i in range(5):
    row_str = " ".join([f"{gc_matrix.data.values[i, j]:.4f}" for j in range(5)])
    print(f"  {row_str}")

print()
print("=" * 60, "Comparison: Ground Truth vs Granger Causality", "=" * 60)
threshold = 0.01
binary_gc = (gc_matrix.data.values > threshold).astype(int)

print(f"Granger causality binary (threshold={threshold}):")
for i in range(5):
    row_str = " ".join([f"{binary_gc[i, j]}" for j in range(5)])
    print(f"  {row_str}")

print()
print("Metrics:")
binary_W_no_diag = binary_W.copy()
np.fill_diagonal(binary_W_no_diag, 0)
binary_gc_no_diag = binary_gc.copy()
np.fill_diagonal(binary_gc_no_diag, 0)

n_true_edges = np.sum(binary_W_no_diag)
n_detected_edges = np.sum(binary_gc_no_diag)
n_correct = np.sum((binary_W_no_diag == 1) & (binary_gc_no_diag == 1))

precision = n_correct / max(n_detected_edges, 1)
recall = n_correct / max(n_true_edges, 1)
f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

print(f"  True edges in W: {n_true_edges}")
print(f"  GC threshold: {threshold}")
print(f"  Detected edges in GC: {n_detected_edges}")
print(f"  Correct detections: {n_correct}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1-score: {f1:.3f}")

print()
print("=" * 60, "Analysis", "=" * 60)
print("Granger causality recovers directed network structure from")
print("time-series data. Stronger coupling leads to higher GC values.")
print(f"F1={f1:.2f} indicates good recovery of the network topology.")
print("=" * 60)
