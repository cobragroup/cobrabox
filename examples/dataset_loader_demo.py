"""Run the structured dummy dataset loader demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cobrabox as cb


def main() -> None:
    # Load a dataset that has "fs" in the JSON sidecar (realistic_swiss or dummy_noise)
    try:
        datasets = cb.dataset("realistic_swiss")
    except FileNotFoundError:
        datasets = cb.dataset("dummy_noise")
    print(datasets)
    print(f"Loaded parts: {len(datasets)}")
    print(f"First part shape: {datasets[0].data.shape}")
    data = datasets[0]
    print(f"Sampling rate (from Settings['fs']): {data.sampling_rate} Hz")
    # Preview: first 2 time steps, first 4 space dims (to_pandas needs named DataArray)
    print("First part preview (2 time x 4 space):", data.to_numpy()[:2, :4].tolist())
    first_subject = data.subjectID
    print(f"First subject: {first_subject}")


if __name__ == "__main__":
    main()
