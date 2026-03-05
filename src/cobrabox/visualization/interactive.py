"""Interactive visualization module for cobrabox time-series data.

This module provides interactive plotting capabilities for exploring EEG/fMRI
time-series data with detailed time-window analysis.

Integration with cobrabox:
    The InteractiveExplorer class accepts a cobrabox Data object and enables
    interactive navigation through the signal with time-window magnification.

    Typical workflow:

    1. Load or create Data object
    2. Create InteractiveExplorer with the Data object
    3. Call .vis() to display interactive plot

Example:
    >>> import cobrabox as cb
    >>> from cobrabox.visualization import InteractiveExplorer
    >>>
    >>> # Load data
    >>> data = cb.dataset("dummy_noise")
    >>>
    >>> # Create and display interactive explorer
    >>> explorer = InteractiveExplorer(
    ...     data=data,
    ...     fs=100,
    ...     window_size=1000,
    ... )
    >>> explorer.vis()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backend_bases import Event
from matplotlib.widgets import Button, Slider

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..data import Data


class InteractiveExplorer:
    """Interactive visualization and exploration of time-series data.

    Creates an interactive figure for exploring time-series data with:
    - Upper plot: full signal averaged over space dimension
    - Lower plot: individual space-indexed traces (line plots or heatmap)
    - Interactive controls: sliders and buttons for navigation

    The data object must have 'time' and 'space' dimensions as required by
    cobrabox Data containers.

    Attributes:
        data: cobrabox Data object to visualize
        fs: Sampling frequency in Hz
        spacing_scale: Scaling factor for automatic trace spacing (line plot mode)
        window_size: Window size for lower plot magnification
        pos: Current position in signal
        space_indices: List of space indices to visualize
        y_limits: Y-axis limits tuple (auto-scaled if None, line plot mode)
        use_heatmap: Whether to use heatmap visualization (True) or line plots (False)
        spacing: Calculated vertical spacing between traces (computed in vis(), line plot mode)
    """

    def __init__(
        self,
        data: Data,
        point_positions: np.ndarray | None = None,
        fs: float = 1000.0,
        spacing_scale: float = 1.2,
        window_size: int = 200,
        pos: int = 100,
        space_indices: list[int] | None = None,
        y_limits: tuple[float, float] | None = None,
        use_heatmap: bool = False,
    ) -> None:
        """Initialize InteractiveExplorer.

        Args:
            data: cobrabox Data object with 'time' and 'space' dimensions.
                Upper plot shows average over space.
                Lower plot shows individual space-indexed traces or heatmap.
            point_positions: Event positions array (1D) in time dimension for marking.
                If None, no events are marked. Example: np.array([100, 500, 1000])
            fs: Sampling frequency in Hz.
            spacing_scale: Scaling factor for automatic spacing calculation.
                Spacing is calculated as spacing_scale * data_range.
                Default 1.2 prevents overlap for most data distributions.
                Only used for line plot mode (use_heatmap=False).
            window_size: Window size for lower plot magnification.
            pos: Initial position to display.
            space_indices: Optional list of space indices to visualize.
                If None, plots all channels.
            y_limits: (min, max) for y-axis range of each trace. If None, auto-scales.
                Only used for line plot mode (use_heatmap=False).
            use_heatmap: If True, display traces as heatmap. If False (default),
                display as offset line plots.

        Raises:
            ValueError: If data lacks required dimensions
        """
        # Validate main data
        if not hasattr(data, "data"):
            msg = "data must be a cobrabox Data object"
            raise ValueError(msg)

        xr_data = data.data
        if not isinstance(xr_data, xr.DataArray):
            msg = "data.data must be an xarray DataArray"
            raise ValueError(msg)

        if "time" not in xr_data.dims:
            msg = "data must have 'time' dimension"
            raise ValueError(msg)
        if "space" not in xr_data.dims:
            msg = "data must have 'space' dimension"
            raise ValueError(msg)

        # Store data (work with xarray directly)
        self.data = data
        self.xr_data = xr_data

        # Validate and store point_positions
        if point_positions is not None:
            self.point_positions = np.asarray(point_positions)
            if self.point_positions.ndim != 1:
                msg = "point_positions must be 1D array"
                raise ValueError(msg)
        else:
            self.point_positions = np.array([], dtype=int)

        # Determine space indices to plot
        n_space = self.xr_data.sizes["space"]
        if space_indices is None:
            self.space_indices = list(range(n_space))
        else:
            self.space_indices = list(space_indices)
            # Validate indices are within bounds
            if any(idx < 0 or idx >= n_space for idx in self.space_indices):
                msg = f"space_indices must be in range [0, {n_space})"
                raise ValueError(msg)

        # Store parameters
        self.fs = float(fs)
        self.spacing_scale = float(spacing_scale)
        self.window_size = int(window_size)
        self.pos = int(pos)
        self.y_limits = y_limits
        self.use_heatmap = bool(use_heatmap)

        # Spacing will be calculated in vis() based on data range
        self.spacing = None

        # Event/annotation state
        self._current_event_idx = 0  # Index in point_positions array
        self.annotations = 2 * np.ones(len(self.point_positions))  # 0/1/2

        # Plotting state (initialized in vis())
        self.fig = None
        self.ax = None
        self.upper_slider = None
        self.button_prev = None
        self.button_next = None
        self.line = None
        self.event_markers = None
        self.ax_lower = None
        self.heatmap_vmin = None  # Global min for heatmap colorbar
        self.heatmap_vmax = None  # Global max for heatmap colorbar

        # Time window bounds (initialized in vis())
        self.vis_down = 0
        self.vis_up = self.window_size

        # Prepare data structures from xarray
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data structures from xarray Data object.

        Computes:
        - data_glob: Average of data over 'space' dimension (xarray)
        - space_coords: Space dimension coordinates for labels
        """
        # Compute data_glob: mean over space dimension, keep as xarray
        self.data_glob_xr = self.xr_data.mean(dim="space")

        # Extract space coordinate for individual traces
        # This will be used for trace labels and indexing
        self.space_coords = self.xr_data.coords["space"]
        self.n_space = len(self.space_coords)

    def vis(self) -> None:
        """Display interactive visualization window.

        Creates interactive matplotlib figure with:
        - Upper plot: full signal (data_glob) with position marker
        - Event markers: orange dots at point_positions indices
        - Lower plot: individual space traces in time window
        - Slider: for navigating through time dimension
        - Prev/Next buttons: for jumping between events (if point_positions provided)
        """
        # Clear any previous plots
        plt.clf()
        plt.close("all")

        # Get time dimension values
        time_coords = self.xr_data.coords["time"].values
        n_time = len(time_coords)

        # Calculate spacing and heatmap range from ENTIRE time series
        # This ensures consistent spacing and coloring even when time_range changes
        # Use only selected space_indices for range calculation
        all_values = self.xr_data.isel(space=self.space_indices).values
        data_min = np.nanmin(all_values)
        data_max = np.nanmax(all_values)
        data_range = data_max - data_min

        # Handle edge case where all values are the same
        if data_range == 0:
            data_range = 1.0

        # Calculate spacing based on data range
        self.spacing = self.spacing_scale * data_range

        # Store global min/max for heatmap colorbar (used for all windows)
        self.heatmap_vmin = data_min
        self.heatmap_vmax = data_max

        # Create figure and subplots
        self.fig, self.ax = plt.subplots(2, 1, figsize=(12, 8))

        # =====================================================================
        # UPPER PLOT: Full signal (data_glob)
        # =====================================================================
        ax_upper = self.ax[0]
        data_glob_np = self.data_glob_xr.values

        # Plot full signal
        ax_upper.plot(time_coords, data_glob_np, linewidth=1.0, color="tab:blue")
        ax_upper.set_xlabel("Time")
        ax_upper.set_ylabel("Amplitude")
        ax_upper.set_title("Full Signal (averaged over space)")
        ax_upper.grid(True, alpha=0.3)

        # Add event markers (orange dots) if point_positions provided
        if len(self.point_positions) > 0:
            event_times = time_coords[self.point_positions]
            event_values = data_glob_np[self.point_positions]
            self.event_markers = ax_upper.scatter(
                event_times, event_values, color="orange", s=100, zorder=5, label="Events"
            )

        # Add position marker (vertical line at current position)
        self.line = ax_upper.axvline(
            x=time_coords[self.pos], color="tab:red", linewidth=2.0, label="Current position"
        )
        ax_upper.legend()

        # =====================================================================
        # LOWER PLOT: Space traces in time window
        # =====================================================================
        ax_lower = self.ax[1]

        # Get initial window bounds
        time_bounds = self._get_time_bounds(self.pos)
        self.vis_down, self.vis_up = time_bounds

        # Plot space traces
        self._update_lower_plot(ax_lower)

        # =====================================================================
        # SLIDER: Position navigation
        # =====================================================================
        ax_slider_pos = self.fig.add_axes((0.2, 0.06, 0.6, 0.03))
        self.upper_slider = Slider(
            ax=ax_slider_pos,
            label="Position",
            valmin=0,
            valmax=n_time - 1,
            valinit=self.pos,
            valstep=1,
        )
        self.upper_slider.on_changed(self._on_slider_change)

        # =====================================================================
        # BUTTONS: Event navigation (if point_positions provided)
        # =====================================================================
        if len(self.point_positions) > 0:
            # Find closest event to current position
            self._current_event_idx = np.argmin(np.abs(self.point_positions - self.pos))

            # Create prev button
            ax_button_prev = self.fig.add_axes((0.05, 0.06, 0.08, 0.04))
            self.button_prev = Button(ax_button_prev, "< Previous")
            self.button_prev.on_clicked(self._on_prev_event)

            # Create next button
            ax_button_next = self.fig.add_axes((0.87, 0.06, 0.08, 0.04))
            self.button_next = Button(ax_button_next, "Next >")
            self.button_next.on_clicked(self._on_next_event)

            # Create info text displaying current event
            info_text = f"Event {self._current_event_idx + 1} of {len(self.point_positions)}"
            self.fig.text(0.5, 0.01, info_text, ha="center", fontsize=10)

        # Store axes for later updates
        self.ax_lower = ax_lower

        # Create colorbar once for heatmap mode (constant, never updated)
        if self.use_heatmap:
            # Create a mappable object for the colorbar with global range
            sm = plt.cm.ScalarMappable(
                cmap="RdBu_r", norm=plt.Normalize(vmin=self.heatmap_vmin, vmax=self.heatmap_vmax)
            )
            sm.set_array([])
            cbar = self.fig.colorbar(sm, ax=ax_lower, shrink=0.8)
            cbar.set_label("Amplitude", fontsize=9)

        # Display
        self.fig.tight_layout(rect=(0, 0.08 if len(self.point_positions) > 0 else 0.05, 1, 1))
        plt.show()

    def _get_time_bounds(self, pos: int) -> tuple[int, int]:
        """Get time window bounds for given position.

        Args:
            pos: Center position index

        Returns:
            (vis_down, vis_up): Lower and upper indices of visible window
        """
        n_time = len(self.xr_data.coords["time"])
        half_window = self.window_size // 2

        vis_down = max(0, pos - half_window)
        vis_up = min(n_time, pos + half_window)

        return vis_down, vis_up

    def _update_lower_plot(self, ax_lower: Axes) -> None:
        """Update lower plot with space traces in current window.

        Supports two visualization modes:
        - Line plots (use_heatmap=False): Individual offset traces
        - Heatmap (use_heatmap=True): 2D color-coded visualization

        Args:
            ax_lower: Matplotlib axes for lower plot
        """
        # Clear previous plot
        ax_lower.clear()

        time_coords = self.xr_data.coords["time"].values
        time_window_coords = time_coords[self.vis_down : self.vis_up]

        if self.use_heatmap:
            # Heatmap visualization
            self._plot_heatmap_lower(ax_lower, time_window_coords)
        else:
            # Line plot visualization
            self._plot_linetraces_lower(ax_lower, time_window_coords)

    def _plot_linetraces_lower(self, ax_lower: Axes, time_window_coords: np.ndarray) -> None:
        """Plot individual space traces as offset line plots.

        Args:
            ax_lower: Matplotlib axes for lower plot
            time_window_coords: Time coordinates for the window
        """
        # Ensure spacing is initialized (set in vis())
        assert self.spacing is not None, "spacing must be set by vis() before plotting"

        # Plot each selected space trace
        for plot_idx, space_idx in enumerate(self.space_indices):
            # Extract space trace
            trace = self.xr_data.isel(space=space_idx).values
            trace_window = trace[self.vis_down : self.vis_up]

            # Apply vertical offset based on calculated spacing
            offset_y = plot_idx * self.spacing
            trace_vis = trace_window + offset_y

            # Get space coordinate for label
            space_label = self.space_coords.values[space_idx]

            # Plot
            ax_lower.plot(
                time_window_coords,
                trace_vis,
                color="0.3",
                label=f"space[{space_label}]",
                linewidth=1.0,
            )

        # Set labels and formatting
        ax_lower.set_xlabel("Time")
        ax_lower.set_ylabel("Amplitude")
        ax_lower.set_title(f"Expanded view (window: {self.window_size} samples)")
        ax_lower.grid(True, alpha=0.3)
        # ax_lower.legend(loc="upper right", fontsize="small")

    def _plot_heatmap_lower(self, ax_lower: Axes, time_window_coords: np.ndarray) -> None:
        """Plot space traces as heatmap visualization.

        Uses global data range (vmin/vmax) for consistent coloring across all windows.

        Args:
            ax_lower: Matplotlib axes for lower plot
            time_window_coords: Time coordinates for the window
        """
        # Extract data for selected space indices in time window
        heatmap_data = []
        space_labels = []

        for space_idx in self.space_indices:
            trace = self.xr_data.isel(space=space_idx).values
            trace_window = trace[self.vis_down : self.vis_up]
            heatmap_data.append(trace_window)
            space_label = self.space_coords.values[space_idx]
            space_labels.append(f"space[{space_label}]")

        # Convert to 2D array: (space, time)
        heatmap_array = np.array(heatmap_data)

        # Create heatmap with global range for consistent coloring
        extent = (
            time_window_coords[0],
            time_window_coords[-1],
            len(self.space_indices) - 0.5,
            -0.5,
        )
        ax_lower.imshow(
            heatmap_array,
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            interpolation="bilinear",
            vmin=self.heatmap_vmin,
            vmax=self.heatmap_vmax,
        )

        # Set labels and formatting
        ax_lower.set_xlabel("Time")
        ax_lower.set_ylabel("Space")
        ax_lower.set_title(f"Heatmap view (window: {self.window_size} samples)")
        ax_lower.set_yticks(range(len(self.space_indices)))
        ax_lower.set_yticklabels(space_labels, fontsize=8)

    def _on_slider_change(self, val: float) -> None:
        """Handle slider position change.

        Args:
            val: New slider value (position index)
        """
        # Ensure visualization objects are initialized (set in vis())
        assert self.line is not None, "line marker must be initialized by vis()"
        assert self.ax_lower is not None, "ax_lower must be initialized by vis()"
        assert self.fig is not None, "fig must be initialized by vis()"

        # Update position
        self.pos = int(val)

        # Get new time bounds
        self.vis_down, self.vis_up = self._get_time_bounds(self.pos)

        # Update position marker in upper plot
        time_coords = self.xr_data.coords["time"].values
        self.line.set_xdata([time_coords[self.pos]])

        # Update lower plot
        self._update_lower_plot(self.ax_lower)

        # Redraw
        self.fig.canvas.draw_idle()

    # =========================================================================
    # EVENT NAVIGATION METHODS
    # =========================================================================

    def _jump_to_event(self, event_idx: int) -> None:
        """Jump to specified event and update all visualizations.

        Args:
            event_idx: Index in point_positions array

        Raises:
            IndexError: If event_idx is out of bounds
        """
        # Ensure slider is initialized (set in vis() when events present)
        assert self.upper_slider is not None, "upper_slider must be initialized by vis()"

        if len(self.point_positions) == 0:
            return

        # Convert to int in case of numpy integer type
        event_idx = int(event_idx)

        if not 0 <= event_idx < len(self.point_positions):
            msg = f"event_idx {event_idx} out of range [0, {len(self.point_positions) - 1}]"
            raise IndexError(msg)

        # Update current event index
        self._current_event_idx = event_idx

        # Jump to event position
        new_pos = self.point_positions[event_idx]
        self.upper_slider.set_val(new_pos)

    def _on_prev_event(self, event: Event) -> None:
        """Button callback: jump to previous event."""
        # Ensure figure is initialized (set in vis() when events present)
        assert self.fig is not None, "fig must be initialized by vis()"

        if len(self.point_positions) == 0:
            return

        if self._current_event_idx > 0:
            new_idx = int(self._current_event_idx - 1)
            self._jump_to_event(new_idx)

            # Update info text
            info_text = f"Event {new_idx + 1} of {len(self.point_positions)}"
            # Find and update info text in figure
            for txt in self.fig.texts:
                if "Event" in txt.get_text():
                    txt.set_text(info_text)
            self.fig.canvas.draw_idle()

    def _on_next_event(self, event: Event) -> None:
        """Button callback: jump to next event."""
        # Ensure figure is initialized (set in vis() when events present)
        assert self.fig is not None, "fig must be initialized by vis()"

        if len(self.point_positions) == 0:
            return

        if self._current_event_idx < len(self.point_positions) - 1:
            new_idx = int(self._current_event_idx + 1)
            self._jump_to_event(new_idx)

            # Update info text
            info_text = f"Event {new_idx + 1} of {len(self.point_positions)}"
            # Find and update info text in figure
            for txt in self.fig.texts:
                if "Event" in txt.get_text():
                    txt.set_text(info_text)
            self.fig.canvas.draw_idle()

    # =========================================================================
    # FUTURE METHODS (TODO)
    # =========================================================================

    # TODO: Implement annotations system:
    #   - Mark each point as yes/no/unset
    #   - YES/NO buttons to annotate current position
    #   - Visual feedback on upper plot (color coding)
    #   - Method to export annotations

    # TODO: Implement feature integration:
    #   - Parameter: features: list[Data] | None
    #   - Compute feature statistics for current time window
    #   - Display as info text: "feature_name = X.XXX"
    #   - Consider how to handle feature dimensionality
    #   - TODO: Define power calculation for xarray features
    #     (currently uses numpy power_time_window, needs xarray adaptation)

    # TODO: Implement add_info() method for custom metrics:
    #   - Accept callables that compute values from position
    #   - Display alongside feature statistics

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"InteractiveExplorer("
            f"data_shape={self.xr_data.shape}, "
            f"fs={self.fs}Hz, "
            f"window_size={self.window_size})"
        )
