"""Compare old sliding_window feature vs new window() method.

Run from project root:
    python examples/compare_windowing.py
"""

import numpy as np

import cobrabox as cb


def compare_old_vs_new() -> None:
    data = cb.dataset("dummy_chain")[0]
    print(f"Original data shape: {data.data.shape}")
    print(f"Original data time points: {data.data.sizes['time']}")
    print()

    print("=" * 60)
    print("OLD APPROACH: sliding_window feature")
    print("=" * 60)
    sw_result = cb.feature.sliding_window(data, window_size=10, step_size=5)
    print(f"After sliding_window shape: {sw_result.data.shape}")
    print(f"  dims: {sw_result.data.dims}")
    print(f"  window_index size: {sw_result.data.sizes.get('window_index', 'N/A')}")

    sw_mean = cb.feature.mean(sw_result, dim="window_index")
    print(f"After mean(dim='window_index') shape: {sw_mean.data.shape}")
    print(f"  time coords (first 5): {sw_mean.data.coords['time'].values[:5]}")
    print(f"  history: {sw_mean.history}")
    print()

    print("=" * 60)
    print("NEW APPROACH: window() method")
    print("=" * 60)
    wdata = data.window(length=10, stride=5)
    print(
        f"After window(10, 5): windowed={wdata.windowed}, \
            length={wdata.window_length}, stride={wdata.window_stride}"
    )

    new_mean = cb.feature.mean(wdata, dim="time")
    print(f"After mean(dim='time') shape: {new_mean.data.shape}")
    print(f"  time coords (first 5): {new_mean.data.coords['time'].values[:5]}")
    print(f"  history: {new_mean.history}")
    print()

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    shapes_match = sw_mean.data.shape == new_mean.data.shape
    times_match = np.allclose(
        sw_mean.data.coords["time"].values, new_mean.data.coords["time"].values
    )
    values_match = np.allclose(sw_mean.data.values, new_mean.data.values)

    print(f"Shapes match: {shapes_match}")
    print(f"Time coords match: {times_match}")
    print(f"Values match: {values_match}")
    print()

    if shapes_match and times_match and values_match:
        print("SUCCESS: Old and new approaches produce identical results!")
    else:
        print("MISMATCH: Results differ between approaches")


def demo_other_features() -> None:
    """Demonstrate window() with different features."""
    data = cb.dataset("dummy_chain")[0]
    wdata = data.window(length=10, stride=5)

    print()
    print("=" * 60)
    print("DEMO: Other features with window()")
    print("=" * 60)

    features = [
        ("mean", lambda: cb.feature.mean(wdata, dim="time")),
        ("max", lambda: cb.feature.max(wdata, dim="time")),
        ("min", lambda: cb.feature.min(wdata, dim="time")),
        ("line_length", lambda: cb.feature.line_length(wdata)),
    ]

    for name, fn in features:
        result = fn()
        print(f"{name}: shape={result.data.shape}, history={result.history}")


if __name__ == "__main__":
    compare_old_vs_new()
    demo_other_features()
