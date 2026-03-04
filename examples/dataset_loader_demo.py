"""Run the structured dummy dataset loader demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cobrabox as cb


def main() -> None:
    datasets = cb.dataset("dummy_chain")
    print(datasets)
    print(f"Loaded parts: {len(datasets)}")
    print(f"First part shape: {datasets[0].data.shape}")
    print(datasets[0].to_pandas().head())
    data = datasets[0]
    first_subject = data.subjectID
    print(f"First subject: {first_subject}")


if __name__ == "__main__":
    main()
