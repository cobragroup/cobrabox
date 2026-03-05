"""Interactive visualization example with event positions.

This example demonstrates:
1. Creating EEG data with events
2. Creating an InteractiveExplorer instance with point_positions
3. Displaying the interactive plot with event markers
4. Navigating events with prev/next buttons
5. Navigating through the signal with the slider
"""

import sys
from pathlib import Path

import numpy as np

# Add src directory to path so we can import cobrabox
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cobrabox.visualization import InteractiveExplorer

import cobrabox as cb


def main() -> None:
    """Run basic visualization example with events."""
    # Create dummy data
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((1000, 16))
    base = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0, subjectID="sub-001")
    eeg = cb.EEG(base.data, sampling_rate=base.sampling_rate, subjectID=base.subjectID)
    data = eeg

    print(f"  Data shape: {data.data.shape}")
    print(f"  Dimensions: {data.data.dims}")
    print(f"  Coordinates: {list(data.data.coords.keys())}")

    # Create artificial event positions (10 evenly distributed points)
    n_events = 10
    point_positions = np.linspace(100, 900, n_events, dtype=int)
    print(f"\n  Event positions: {point_positions}")

    # Create interactive explorer with events
    print("\nCreating InteractiveExplorer with events...")
    explorer = InteractiveExplorer(
        data=data,
        point_positions=point_positions,  # Add event positions
        fs=100.0,  # Sampling frequency
        window_size=1000,  # Samples to show in lower plot
        pos=500,  # Initial position (closer to first event)
        x_offset=0.0,  # Vertical offset between space traces
        y_limits=(-3.0, 3.0),  # Y-axis limits
    )
    print(f"  Created: {explorer}")

    # Display interactive plot
    print("\nDisplaying interactive plot...")
    print("  Controls:")
    print("    - Slider: Navigate through time")
    print("    - < Previous: Jump to previous event")
    print("    - Next >: Jump to next event")
    print("  Visualization:")
    print("    - Upper plot (blue): Full signal with orange event markers")
    print("    - Lower plot (colored): Magnified window around position")
    print("    - Red line: Current position marker")
    explorer.vis()


if __name__ == "__main__":
    main()
