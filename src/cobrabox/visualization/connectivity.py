"""Connectivity matrix visualization for cobrabox Data objects.

This module provides functionality to visualize connectivity matrices
from Data objects with xarray support. Supports both static and interactive modes.

Classes:
    ConnectivityPlotter: Static visualization of connectivity matrices
    InteractiveConnectivityExplorer: Interactive exploration of 3D+ connectivity data
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import Event
from matplotlib.colorbar import Colorbar
from matplotlib.widgets import Button

if TYPE_CHECKING:
    from cobrabox.data import Data


class ConnectivityPlotter:
    """Plotter for connectivity matrices from Data objects.

    This class handles visualization of connectivity matrices stored in
    Data.data xarray DataArray. The data should have at least two non-time
    dimensions representing the connectivity pairs.

    Examples:
        >>> import cobrabox as cb
        >>> from cobrabox.visualization import ConnectivityPlotter
        >>>
        >>> # Create synthetic connectivity data
        >>> conn_data = np.random.randn(10, 10)  # 10x10 connectivity matrix
        >>> data = cb.Data(xr.DataArray(
        ...     conn_data,
        ...     dims=['node_from', 'node_to'],
        ...     coords={'node_from': range(10), 'node_to': range(10)}
        ... ))
        >>>
        >>> # Visualize
        >>> plotter = ConnectivityPlotter()
        >>> fig, ax = plotter.plot(data)
    """

    def __init__(self, figsize: tuple[float, float] = (8, 7), dpi: int = 100) -> None:
        """Initialize ConnectivityPlotter.

        Args:
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot(
        self,
        data: Data,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        add_colorbar: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a connectivity matrix plot from a Data object.

        Args:
            data: Data object with xarray DataArray containing connectivity matrix.
                Must have at least two non-time dimensions.
            cmap: Colormap name (default: 'RdBu_r' for symmetric data)
            vmin: Minimum value for colormap. If None, auto-scales to data range.
            vmax: Maximum value for colormap. If None, auto-scales to data range.
            title: Plot title. If None, generates from data dimensions.
            add_colorbar: Whether to add a colorbar (default: True)

        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes)

        Raises:
            ValueError: If data doesn't have at least two non-time dimensions
        """
        xr_data = data.data

        # Find all non-time dimensions
        non_time_dims = [dim for dim in xr_data.dims if dim != "time"]

        # Validate that we have at least 2 non-time dimensions
        if len(non_time_dims) < 2:
            msg = (
                f"Data must have at least 2 non-time dimensions for connectivity matrix. "
                f"Found: {non_time_dims}"
            )
            raise ValueError(msg)

        # Use first two non-time dimensions
        dim1, dim2 = non_time_dims[0], non_time_dims[1]

        # Extract 2D matrix (take first slice if more than 2 dimensions)
        if len(non_time_dims) > 2:
            # Select first slice of extra dimensions
            extra_dims = non_time_dims[2:]
            slices = dict.fromkeys(extra_dims, 0)
            matrix_data = xr_data.isel(slices).values
        else:
            matrix_data = xr_data.values

        # Ensure 2D
        if matrix_data.ndim != 2:
            msg = f"Matrix data must be 2D after dimension selection, got shape {matrix_data.shape}"
            raise ValueError(msg)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Auto-scale colormap limits if not provided
        if vmin is None:
            vmin = np.nanmin(matrix_data)
        if vmax is None:
            vmax = np.nanmax(matrix_data)

        # Plot heatmap
        im = ax.imshow(matrix_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Connectivity", fontsize=10)

        # Set labels
        ax.set_xlabel(str(dim2), fontsize=11)
        ax.set_ylabel(str(dim1), fontsize=11)

        # Set title
        if title is None:
            if len(non_time_dims) > 2:
                title = f"Connectivity Matrix: {dim1} x {dim2} (slice 0 of {extra_dims})"
            else:
                title = f"Connectivity Matrix: {dim1} x {dim2}"

        ax.set_title(title, fontsize=12, fontweight="bold")

        # Add grid for better readability
        ax.grid(False)

        fig.tight_layout()

        return fig, ax


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

    Args:
        data: Data object with xarray DataArray containing connectivity matrix
        cmap: Colormap name (default: 'RdBu_r')
        vmin: Minimum value for colormap (auto-scales if None)
        vmax: Maximum value for colormap (auto-scales if None)
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for figure resolution

    Returns:
        Tuple of (matplotlib Figure, matplotlib Axes)
    """
    plotter = ConnectivityPlotter(figsize=figsize, dpi=dpi)
    return plotter.plot(data, cmap=cmap, vmin=vmin, vmax=vmax, title=title)


class InteractiveConnectivityExplorer:
    """Interactive exploration of connectivity matrices with 3+ dimensions.

    This class enables interactive navigation through high-dimensional connectivity
    data. Users can select which two dimensions form the connectivity matrix and
    navigate through slices of other dimensions using buttons.

    Compatible with data having 3+ non-time dimensions. For 2D connectivity data,
    use ConnectivityPlotter instead.

    Examples:
        >>> import cobrabox as cb
        >>> from cobrabox.visualization import InteractiveConnectivityExplorer
        >>>
        >>> # Load or create 3D+ connectivity data
        >>> explorer = InteractiveConnectivityExplorer(data=data)
        >>> explorer.vis()
    """

    def __init__(
        self,
        data: Data,
        cmap: str = "RdBu_r",
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 100,
    ) -> None:
        """Initialize InteractiveConnectivityExplorer.

        Args:
            data: Data object with xarray DataArray containing connectivity matrix.
                Must have 3+ non-time dimensions.
            cmap: Colormap name (default: 'RdBu_r')
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution

        Raises:
            ValueError: If data has fewer than 3 non-time dimensions
        """
        import xarray as xr

        xr_data = data.data

        if not isinstance(xr_data, xr.DataArray):
            msg = "data.data must be an xarray DataArray"
            raise ValueError(msg)

        # Find all non-time dimensions
        self.non_time_dims = [str(dim) for dim in xr_data.dims if dim != "time"]

        if len(self.non_time_dims) < 3:
            msg = (
                f"InteractiveConnectivityExplorer requires 3+ non-time dimensions. "
                f"Found: {self.non_time_dims}. Use ConnectivityPlotter for 2D data."
            )
            raise ValueError(msg)

        self.data = data
        self.xr_data = xr_data
        self.cmap = cmap
        self.figsize = figsize
        self.dpi = dpi

        # Generate all possible dimension pairs for the connectivity matrix
        # (all 2-element combinations from non_time_dims)
        from itertools import combinations

        self.dimension_pairs = list(combinations(self.non_time_dims, 2))

        if len(self.dimension_pairs) == 0:
            msg = (
                f"Cannot create 2D matrix combinations from {len(self.non_time_dims)} "
                f"non-time dimensions. Need at least 2."
            )
            raise ValueError(msg)

        # Start with first pair
        self.current_pair_idx = 0
        self._update_matrix_dims()

        # Plot state
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.im: plt.cm.ScalarMappable | None = None
        self.cbar: Colorbar | None = None
        self.button_prev_dims: Button | None = None
        self.button_next_dims: Button | None = None
        self.button_prev_slice: Button | None = None
        self.button_next_slice: Button | None = None
        self.info_text: plt.Text | None = None
        self.dims_text: plt.Text | None = None

        # Compute global vmin/vmax for consistent coloring across all slices
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
        self.slice_indices: dict[str, int] = dict.fromkeys(self.extra_dims, 0)

    def vis(self) -> None:
        """Display the interactive connectivity explorer."""
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
        self.im = self.ax.imshow(
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
                    self.slice_indices[current_dim] = self.xr_data.sizes[current_dim] - 1
                    break

        self._update_plot()
        self._update_info_text()

    def _on_next_slice(self, event: Event | None = None) -> None:
        """Button callback: move to next slice."""
        if len(self.extra_dims) == 0:
            return

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
        """Button callback: move to previous dimension pair."""
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
        """Button callback: move to next dimension pair."""
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
