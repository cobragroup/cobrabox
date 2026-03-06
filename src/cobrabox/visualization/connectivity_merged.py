"""Connectivity matrix visualization for cobrabox Data objects.

This module provides unified functionality to visualize connectivity matrices
from Data objects with xarray support.

For 2D connectivity data (exactly 2 non-time dimensions), provides static
visualization. For 3D+ data, provides interactive exploration with dimension
selection and slice navigation.

Classes:
    ConnectivityPlotter: Unified visualization for connectivity matrices
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backend_bases import Event
from matplotlib.colorbar import Colorbar
from matplotlib.widgets import Button

if TYPE_CHECKING:
    from cobrabox.data import Data


class ConnectivityPlotter:
    """Unified plotter for connectivity matrices with adaptive visualization.

    Handles visualization of connectivity matrices from Data objects with
    xarray support. Automatically adapts visualization mode based on data
    dimensionality:

    - **2 non-time dimensions**: Static heat map visualization
    - **3+ non-time dimensions**: Interactive exploration with:
      - Dynamic dimension pair selection via buttons
      - Slice navigation for remaining dimensions
      - Global colorbar for consistent coloring across all views

    The data should have 'time' dimension plus at least 2 other dimensions
    representing connectivity relationships.

    Examples:
        >>> import cobrabox as cb
        >>> from cobrabox.visualization import ConnectivityPlotter
        >>> import xarray as xr
        >>> import numpy as np
        >>>
        >>> # Create 2D connectivity data (static visualization)
        >>> conn_2d = np.random.randn(10, 10)
        >>> data_2d = cb.Data(xr.DataArray(
        ...     conn_2d,
        ...     dims=['source', 'target'],
        ...     coords={'source': range(10), 'target': range(10)}
        ... ))
        >>> plotter = ConnectivityPlotter()
        >>> fig, ax = plotter.plot(data_2d)
        >>>
        >>> # Create 4D connectivity data (interactive visualization)
        >>> conn_4d = np.random.randn(6, 6, 3, 2)
        >>> data_4d = cb.Data(xr.DataArray(
        ...     conn_4d,
        ...     dims=['source', 'target', 'band', 'region'],
        ... ))
        >>> plotter = ConnectivityPlotter()
        >>> plotter.vis(data_4d)  # Interactive mode
    """

    def __init__(
        self, figsize: tuple[float, float] = (10, 8), dpi: int = 100, cmap: str = "RdBu_r"
    ) -> None:
        """Initialize ConnectivityPlotter.

        Args:
            figsize: Figure size as (width, height) in inches. Default (10, 8)
                scaled for interactive mode. Will be adjusted to (8, 7) for
                static 2D plots.
            dpi: Dots per inch for figure resolution
            cmap: Colormap name (default: 'RdBu_r' for symmetric data)
        """
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap

        # Runtime state (initialized in plot/vis methods)
        self.data: Data | None = None
        self.xr_data: xr.DataArray | None = None
        self.non_time_dims: list[str] = []
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.cbar: Colorbar | None = None

        # For interactive mode only
        self.dimension_pairs: list[tuple[str, str]] = []
        self.current_pair_idx: int = 0
        self.matrix_dims: list[str] = []
        self.extra_dims: list[str] = []
        self.slice_indices: dict[str, int] = {}
        self.vmin: float | None = None
        self.vmax: float | None = None
        self.button_prev_dims: Button | None = None
        self.button_next_dims: Button | None = None
        self.button_prev_slice: Button | None = None
        self.button_next_slice: Button | None = None
        self.info_text: plt.Text | None = None
        self.dims_text: plt.Text | None = None

    def _validate_and_prepare_data(self, data: Data) -> None:
        """Validate data and extract non-time dimensions.

        Args:
            data: Data object with xarray DataArray

        Raises:
            ValueError: If data doesn't have required structure or dimensions
        """
        if not hasattr(data, "data"):
            msg = "data must be a cobrabox Data object"
            raise ValueError(msg)

        xr_data = data.data
        if not isinstance(xr_data, xr.DataArray):
            msg = "data.data must be an xarray DataArray"
            raise ValueError(msg)

        # Handle time dimension: connectivity visualization is not for time-series
        if "time" in xr_data.dims:
            time_len = len(xr_data.coords["time"])
            if time_len == 1:
                # Drop the time dimension if it has only one coordinate
                xr_data = xr_data.squeeze("time", drop=True)
            elif time_len > 1:
                # Raise error if time dimension has multiple coordinates
                msg = (
                    f"Connectivity visualization is not designed for time-series data. "
                    f"Time dimension has {time_len} coordinates. "
                    f"Use data with time dimension length of 1 (will be dropped) only."
                )
                raise ValueError(msg)

        # Find all non-time dimensions
        self.non_time_dims = [str(dim) for dim in xr_data.dims if dim != "time"]

        # Validate we have at least 2 non-time dimensions
        if len(self.non_time_dims) < 2:
            msg = (
                f"Data must have at least 2 non-time dimensions for connectivity matrix. "
                f"Found: {self.non_time_dims}"
            )
            raise ValueError(msg)

        self.data = data
        self.xr_data = xr_data

    def _prepare_interactive_mode(self) -> None:
        """Prepare interactive mode for 3D+ data."""
        assert self.xr_data is not None, "xr_data must be initialized"

        # Generate all possible dimension pairs
        self.dimension_pairs = list(combinations(self.non_time_dims, 2))

        # Start with first pair
        self.current_pair_idx = 0
        self._update_matrix_dims()

        # Compute global vmin/vmax for consistent coloring
        all_values = self.xr_data.values
        self.vmin = np.nanmin(all_values)
        self.vmax = np.nanmax(all_values)

    def _update_matrix_dims(self) -> None:
        """Update matrix_dims and extra_dims based on current_pair_idx."""
        # Get current dimension pair
        self.matrix_dims = list(self.dimension_pairs[self.current_pair_idx])

        # Extra dims are all non-selected non-time dims
        self.extra_dims = [dim for dim in self.non_time_dims if dim not in self.matrix_dims]

        # Reset slice indices for extra dimensions
        assert self.xr_data is not None, "xr_data must be initialized"
        self.slice_indices = dict.fromkeys(self.extra_dims, 0)

    def plot(
        self,
        data: Data,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        add_colorbar: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a static connectivity matrix plot (2D data only).

        For 2D data (exactly 2 non-time dimensions), returns a static plot.
        For 3D+ data, use vis() for interactive exploration instead.

        Args:
            data: Data object with xarray DataArray containing connectivity matrix.
                Must have exactly 2 non-time dimensions for this method.
            vmin: Minimum value for colormap. If None, auto-scales to data range.
            vmax: Maximum value for colormap. If None, auto-scales to data range.
            title: Plot title. If None, generates from data dimensions.
            add_colorbar: Whether to add a colorbar (default: True)

        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes)

        Raises:
            ValueError: If data doesn't have exactly 2 non-time dimensions
        """
        # Validate and prepare data
        self._validate_and_prepare_data(data)

        # For plot() method, require exactly 2D (static mode)
        if len(self.non_time_dims) != 2:
            msg = (
                f"plot() method requires exactly 2 non-time dimensions for static visualization. "
                f"Found {len(self.non_time_dims)}: {self.non_time_dims}. "
                f"Use vis() instead for interactive visualization."
            )
            raise ValueError(msg)

        assert self.xr_data is not None, "xr_data must be initialized"
        dim1, dim2 = self.non_time_dims[0], self.non_time_dims[1]

        # Extract 2D matrix
        matrix_data = self.xr_data.values

        # Ensure 2D
        if matrix_data.ndim != 2:
            msg = f"Matrix data must be 2D, got shape {matrix_data.shape}"
            raise ValueError(msg)

        # Auto-scale colormap limits if not provided
        if vmin is None:
            vmin = np.nanmin(matrix_data)
        if vmax is None:
            vmax = np.nanmax(matrix_data)

        # Create figure with adjusted size for static mode
        static_figsize = (8, 7)
        self.fig, self.ax = plt.subplots(figsize=static_figsize, dpi=self.dpi)

        # Plot heatmap
        im = self.ax.imshow(
            matrix_data, cmap=self.cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper"
        )

        # Add colorbar
        if add_colorbar:
            self.cbar = self.fig.colorbar(im, ax=self.ax)
            self.cbar.set_label("Connectivity", fontsize=10)

        # Set labels
        self.ax.set_xlabel(str(dim2), fontsize=11)
        self.ax.set_ylabel(str(dim1), fontsize=11)

        # Set title
        if title is None:
            title = f"Connectivity Matrix: {dim1} x {dim2}"

        self.ax.set_title(title, fontsize=12, fontweight="bold")

        # Add grid for better readability
        self.ax.grid(False)

        self.fig.tight_layout()

        return self.fig, self.ax

    def vis(self, data: Data) -> None:
        """Display connectivity matrix visualization (static or interactive).

        For 2D data, displays static heat map.
        For 3D+ data, displays interactive explorer with dimension selection
        and slice navigation buttons.

        Args:
            data: Data object with xarray DataArray. Must have 3+ non-time
                dimensions for interactive mode, exactly 2 for static mode.

        Raises:
            ValueError: If data has fewer than 2 non-time dimensions
        """
        # Validate and prepare data
        self._validate_and_prepare_data(data)

        assert self.xr_data is not None, "xr_data must be initialized"

        # Check dimensionality
        is_2d = len(self.non_time_dims) == 2
        is_interactive = len(self.non_time_dims) >= 3

        if is_2d:
            # Static mode: show 2D plot
            self._vis_static_2d()
        elif is_interactive:
            # Interactive mode: show with buttons
            self._prepare_interactive_mode()
            self._vis_interactive()
        else:
            msg = (
                f"Cannot visualize data with {len(self.non_time_dims)} non-time dimensions. "
                f"Need at least 2."
            )
            raise ValueError(msg)

    def _vis_static_2d(self) -> None:
        """Display static 2D connectivity matrix visualization."""
        assert self.xr_data is not None, "xr_data must be initialized"

        # Create figure
        static_figsize = (8, 7)
        self.fig, self.ax = plt.subplots(figsize=static_figsize, dpi=self.dpi)

        dim1, dim2 = self.non_time_dims[0], self.non_time_dims[1]

        # Get data
        matrix_data = self.xr_data.values

        # Ensure 2D
        if matrix_data.ndim != 2:
            msg = f"Matrix data must be 2D, got shape {matrix_data.shape}"
            raise ValueError(msg)

        # Auto-scale colormap limits
        vmin = np.nanmin(matrix_data)
        vmax = np.nanmax(matrix_data)

        # Plot heatmap
        im = self.ax.imshow(
            matrix_data, cmap=self.cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper"
        )

        # Add colorbar
        self.cbar = self.fig.colorbar(im, ax=self.ax)
        self.cbar.set_label("Connectivity", fontsize=10)

        # Set labels
        self.ax.set_xlabel(str(dim2), fontsize=11)
        self.ax.set_ylabel(str(dim1), fontsize=11)

        # Set title
        title = f"Connectivity Matrix: {dim1} x {dim2}"
        self.ax.set_title(title, fontsize=12, fontweight="bold")

        self.ax.grid(False)

        self.fig.tight_layout()
        plt.show()

    def _vis_interactive(self) -> None:
        """Display interactive connectivity explorer for 3D+ data."""
        assert self.xr_data is not None, "xr_data must be initialized"

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create initial plot
        self._update_plot()

        # Create colorbar once (constant, never recreated on button clicks)
        sm = plt.cm.ScalarMappable(
            cmap=self.cmap, norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax)
        )
        sm.set_array([])
        self.cbar = self.fig.colorbar(sm, ax=self.ax)
        self.cbar.set_label("Connectivity", fontsize=10)

        # =====================================================================
        # BUTTONS: Select dimensions and navigate slices
        # =====================================================================
        button_height = 0.05

        # Buttons for changing matrix dimensions (always available if > 1 pair)
        if len(self.dimension_pairs) > 1:
            # Prev dims button
            ax_button_prev_dims = self.fig.add_axes((0.15, 0.02, 0.12, button_height))
            self.button_prev_dims = Button(ax_button_prev_dims, "< Prev Dims")
            self.button_prev_dims.on_clicked(self._on_prev_dims)

            # Next dims button
            ax_button_next_dims = self.fig.add_axes((0.73, 0.02, 0.12, button_height))
            self.button_next_dims = Button(ax_button_next_dims, "Next Dims >")
            self.button_next_dims.on_clicked(self._on_next_dims)

            # Dims text
            self.dims_text = self.fig.text(0.5, 0.145, "", ha="center", fontsize=9, weight="bold")
            self._update_dims_text()

        # Buttons for navigating extra dimensions (only if there are extra dims)
        if len(self.extra_dims) > 0:
            # Prev slice button
            ax_button_prev_slice = self.fig.add_axes((0.3, 0.02, 0.1, button_height))
            self.button_prev_slice = Button(ax_button_prev_slice, "< Prev Slice")
            self.button_prev_slice.on_clicked(self._on_prev_slice)

            # Next slice button
            ax_button_next_slice = self.fig.add_axes((0.6, 0.02, 0.1, button_height))
            self.button_next_slice = Button(ax_button_next_slice, "Next Slice >")
            self.button_next_slice.on_clicked(self._on_next_slice)

            # Info text for slices
            self.info_text = self.fig.text(0.5, 0.08, "", ha="center", fontsize=9)
            self._update_info_text()

        assert self.fig is not None, "fig must be initialized by vis()"
        extra_space = 0.15 if len(self.extra_dims) > 0 else 0.1
        extra_space = max(extra_space, 0.15 if len(self.dimension_pairs) > 1 else 0.1)
        self.fig.tight_layout(rect=(0, extra_space, 1, 1))
        plt.show()

    def _update_plot(self) -> None:
        """Update the connectivity matrix visualization."""
        assert self.ax is not None, "ax must be initialized by vis()"
        assert self.fig is not None, "fig must be initialized by vis()"
        assert self.xr_data is not None, "xr_data must be initialized"

        self.ax.clear()

        # Build selection dictionary for extra dimensions
        selection = {dim: self.slice_indices[dim] for dim in self.extra_dims}

        # Select data and extract 2D matrix
        matrix_data = self.xr_data.isel(selection).values

        # Ensure 2D
        if matrix_data.ndim != 2:
            msg = f"Matrix data must be 2D, got shape {matrix_data.shape}"
            raise ValueError(msg)

        # Plot heatmap
        self.ax.imshow(
            matrix_data,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            aspect="auto",
            origin="upper",
        )

        # Set labels
        dim1, dim2 = self.matrix_dims
        self.ax.set_xlabel(str(dim2), fontsize=11)
        self.ax.set_ylabel(str(dim1), fontsize=11)

        # Set title
        title = f"Connectivity: {dim1} x {dim2}"
        if len(self.extra_dims) > 0:
            self.ax.set_title(title, fontsize=12, fontweight="bold")

        self.fig.canvas.draw_idle()

    def _update_info_text(self) -> None:
        """Update the info text showing current slice selection."""
        if self.info_text is None:
            return

        # Build slice description
        slice_strs = [f"{dim}[{self.slice_indices[dim]}]" for dim in self.extra_dims]
        info = f"Slices: {' | '.join(slice_strs)}"
        self.info_text.set_text(info)

    def _on_prev_slice(self, event: Event | None = None) -> None:
        """Button callback: move to previous slice."""
        if len(self.extra_dims) == 0:
            return

        # Go to previous dimension or previous slice
        current_dim = self.extra_dims[-1]

        if self.slice_indices[current_dim] > 0:
            self.slice_indices[current_dim] -= 1
        elif self.slice_indices[current_dim] == 0 and len(self.extra_dims) > 1:
            # Move to previous dimension at max index
            for i in range(len(self.extra_dims) - 2, -1, -1):
                dim = self.extra_dims[i]
                if self.slice_indices[dim] > 0:
                    self.slice_indices[dim] -= 1
                    assert self.xr_data is not None, "xr_data must be initialized"
                    self.slice_indices[current_dim] = self.xr_data.sizes[current_dim] - 1
                    break

        self._update_plot()
        self._update_info_text()

    def _on_next_slice(self, event: Event | None = None) -> None:
        """Button callback: move to next slice."""
        if len(self.extra_dims) == 0:
            return

        assert self.xr_data is not None, "xr_data must be initialized"

        # Go to next dimension or next slice
        current_dim = self.extra_dims[-1]
        max_idx = self.xr_data.sizes[current_dim] - 1

        if self.slice_indices[current_dim] < max_idx:
            self.slice_indices[current_dim] += 1
        elif self.slice_indices[current_dim] == max_idx and len(self.extra_dims) > 1:
            # Move to next dimension at start index
            for i in range(len(self.extra_dims) - 2, -1, -1):
                dim = self.extra_dims[i]
                max_dim_idx = self.xr_data.sizes[dim] - 1
                if self.slice_indices[dim] < max_dim_idx:
                    self.slice_indices[dim] += 1
                    self.slice_indices[current_dim] = 0
                    break

        self._update_plot()
        self._update_info_text()

    def _update_dims_text(self) -> None:
        """Update the dimensions text showing current matrix selection."""
        if self.dims_text is None:
            return

        dim1, dim2 = self.matrix_dims
        num_pairs = len(self.dimension_pairs)
        current_num = self.current_pair_idx + 1
        dims_str = f"Matrix: {dim1} x {dim2}  ({current_num}/{num_pairs})"
        self.dims_text.set_text(dims_str)

    def _on_prev_dims(self, event: Event | None = None) -> None:
        """Button callback: move to previous dimension pair.

        Args:
            event: Matplotlib button event (unused)
        """
        if len(self.dimension_pairs) <= 1:
            return

        # Move to previous pair (wrap around)
        self.current_pair_idx = (self.current_pair_idx - 1) % len(self.dimension_pairs)
        self._update_matrix_dims()

        # Redraw everything
        self._update_plot()
        self._update_dims_text()
        if len(self.extra_dims) > 0:
            self._update_info_text()

    def _on_next_dims(self, event: Event | None = None) -> None:
        """Button callback: move to next dimension pair.

        Args:
            event: Matplotlib button event (unused)
        """
        if len(self.dimension_pairs) <= 1:
            return

        # Move to next pair (wrap around)
        self.current_pair_idx = (self.current_pair_idx + 1) % len(self.dimension_pairs)
        self._update_matrix_dims()

        # Redraw everything
        self._update_plot()
        self._update_dims_text()
        if len(self.extra_dims) > 0:
            self._update_info_text()

    def __repr__(self) -> str:
        """Return string representation."""
        mode = "interactive" if len(self.non_time_dims) >= 3 else "static"
        return f"ConnectivityPlotter(dimensions={self.non_time_dims}, mode={mode})"


def plot_connectivity(
    data: Data,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 7),
    dpi: int = 100,
) -> tuple[plt.Figure, plt.Axes]:
    """Convenience function to create a connectivity matrix plot.

    For 2D data (exactly 2 non-time dimensions), creates a static plot.
    For 3D+ data, use ConnectivityPlotter(...).vis(data) instead.

    Args:
        data: Data object with xarray DataArray containing connectivity matrix.
            Must have exactly 2 non-time dimensions.
        cmap: Colormap name (default: 'RdBu_r')
        vmin: Minimum value for colormap (auto-scales if None)
        vmax: Maximum value for colormap (auto-scales if None)
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for figure resolution

    Returns:
        Tuple of (matplotlib Figure, matplotlib Axes)

    Raises:
        ValueError: If data doesn't have exactly 2 non-time dimensions
    """
    plotter = ConnectivityPlotter(figsize=figsize, dpi=dpi, cmap=cmap)
    return plotter.plot(data, vmin=vmin, vmax=vmax, title=title)
