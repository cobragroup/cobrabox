"""Demo loading the realistic Swiss VAR dataset and inspecting metadata."""

import cobrabox as cb


def main() -> None:
    datasets = cb.dataset("realistic_swiss")
    print(f"Loaded parts: {len(datasets)}")

    first = datasets[0]
    print("First part shape:", first.data.shape)
    print("First part dims:", first.data.dims)

    # Show sidecar metadata loaded into extra (e.g. Settings / SOZ)
    print("Extra keys:", list(first.extra.keys()))
    settings = first.extra.get("Settings", {})
    print("Settings keys:", list(settings.keys()))
    if "SOZ" in settings:
        print("SOZ:", settings["SOZ"])


if __name__ == "__main__":
    main()
