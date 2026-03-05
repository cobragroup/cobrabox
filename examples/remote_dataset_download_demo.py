"""Demo for downloading large remote datasets and inspecting the raw files.

This script does *not* parse the EEG files yet. It triggers the download
pipeline for the remote datasets and prints a short summary of what was
fetched, so you can inspect the directory structure before implementing
full loaders.
"""

from __future__ import annotations

from cobrabox.remote_datasets import ensure_remote_files, get_remote_dataset_spec


def _summarise_dataset(identifier: str) -> None:
    print(f"\n=== {identifier} ===")

    spec = get_remote_dataset_spec(identifier)
    if spec is None:
        print("No RemoteDatasetSpec registered for this identifier.")
        return

    # Trigger download (or reuse existing cache) and get the local directory.
    dataset_dir = ensure_remote_files(spec)
    print(f"Local dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        print("Dataset directory does not exist (unexpected).")
        return

    paths = sorted(p for p in dataset_dir.iterdir() if p.is_file())
    if not paths:
        print("No files were found in the dataset directory.")
        return

    print(f"Total files: {len(paths)}")
    print("First few files:")
    for p in paths[:10]:
        size = p.stat().st_size
        print(f"  - {p.name} ({size} bytes)")


def main() -> None:
    # swiss_eeg_short: short-duration Swiss EEG files (zipped).
    _summarise_dataset("swiss_eeg_short")

    # Optional: uncomment to test Bonn EEG download once you have Kaggle access
    # configured. This may fail with a RuntimeError if authentication is
    # required.
    # try:
    #     _summarise_dataset("bonn_eeg")
    # except RuntimeError as e:
    #     print("\nbonn_eeg download raised a RuntimeError:")
    #     print(f"  {e}")


if __name__ == "__main__":
    main()
