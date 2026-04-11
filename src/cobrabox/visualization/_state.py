"""Shared reactive state for SpaceExplorer.

``ExplorerState`` holds all param-driven state (dimension selectors, position,
window size, visualization mode, colormap settings, etc.) and the data retrieval
helpers consumed by ``DataPlot`` and ``ControlsPanel``.

This module has **no** Panel/HoloViews/Matplotlib imports — it only depends on
``param`` and ``numpy``, which makes it independently testable.

Constants
---------
_COLORMAPS, _VIZ_MODES, _SCATTER_PLOT_TYPES, _MARKER_TYPES, _BOX_PLOT_TYPES,
_HIST_PLOT_TYPES, _JOINT_KINDS, _NAMED_COLORS
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import param

from cobrabox.data import Data

if TYPE_CHECKING:
    import pandas as pd
    import xarray

# ---------------------------------------------------------------------------
# Constants shared with ControlsPanel and DataPlot
# ---------------------------------------------------------------------------

_COLORMAPS: list[str] = [
    # Diverging — good for signals centred around zero
    "RdBu_r",
    "RdBu",
    # Perceptually uniform sequential
    "viridis",
    "plasma",
    "inferno",
    "magma",
    # Sequential
    "hot",
    "Blues",
    "Reds",
    "Greens",
    "Greys",
]

_VIZ_MODES: list[str] = [
    "time_series",
    "heatmap",
    "scatter",
    "boxplot",
    "histogram",
    "histogram_2d",
]

_SCATTER_PLOT_TYPES: list[str] = ["scatterplot", "regplot"]
_MARKER_TYPES: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]
_BOX_PLOT_TYPES: list[str] = ["boxplot", "violinplot", "swarmplot", "barplot"]
_HIST_PLOT_TYPES: list[str] = ["histplot", "kdeplot"]
_JOINT_KINDS: list[str] = ["scatter", "hist", "reg", "kde"]
_NAMED_COLORS: list[str] = [
    "steelblue",
    "tomato",
    "mediumseagreen",
    "mediumpurple",
    "sandybrown",
    "deepskyblue",
    "crimson",
    "darkorange",
    "teal",
    "slategray",
]


# ---------------------------------------------------------------------------
# Shared reactive state
# ---------------------------------------------------------------------------


class ExplorerState(param.Parameterized):
    """Shared reactive state for all explorer components.

    Holds the ``Data`` object and parameters that drive every view:
    dimension selectors, current position, window size, visualization mode,
    and colormap settings.

    Parameters
    ----------
    data : Data
        Cobrabox ``Data`` object to explore.
    sweep_dim : str
        Dimension mapped to the X axis (e.g. ``"time"``).
    trace_dim : str
        Dimension whose entries become individual traces / heatmap rows.
    position : int
        Current index along ``sweep_dim``.
    window_size : int
        Number of samples shown in the detail view.
    viz_mode : str
        Visualization mode: "time_series" or "heatmap".
    cmap : str
        Heatmap colormap name.
    clim_min : float
        Colormap lower bound.
    clim_max : float
        Colormap upper bound.
    """

    sweep_dim = param.Selector(doc="Dimension mapped to X axis")
    trace_dim = param.Selector(doc="Dimension mapped to traces / rows")
    position = param.Integer(default=0, bounds=(0, 1), doc="Index along sweep_dim")
    window_size = param.Integer(default=200, bounds=(10, 1000), doc="Detail window width")
    viz_mode = param.Selector(default="heatmap", objects=_VIZ_MODES, doc="Visualization mode")
    # Colormap controls (heatmap / time_series)
    cmap = param.Selector(default="RdBu_r", objects=_COLORMAPS, doc="Heatmap colormap")
    clim_min = param.Number(default=0.0, doc="Colormap lower bound")
    clim_max = param.Number(default=1.0, doc="Colormap upper bound")
    extra_dim_indices = param.Dict(default={}, doc="Current index per non-visualized dimension")
    extra_dim_aggregate = param.Dict(
        default={}, doc="Dims to aggregate over in histogram mode (dim → bool)"
    )
    # --- seaborn shared ---
    sns_color = param.Selector(default="steelblue", objects=_NAMED_COLORS, doc="Plot color")
    # --- scatter ---
    scatter_trace_selection = param.List(
        default=[], item_type=str, doc="String labels of trace_dim indices shown in scatter mode"
    )
    scatter_plot_type = param.Selector(default="scatterplot", objects=_SCATTER_PLOT_TYPES)
    scatter_marker = param.Selector(default="o", objects=_MARKER_TYPES, doc="Marker type")
    scatter_marker_size = param.Integer(default=40, bounds=(5, 200), doc="Marker size")
    # --- boxplot ---
    box_plot_type = param.Selector(default="boxplot", objects=_BOX_PLOT_TYPES)
    # --- histogram ---
    hist_plot_type = param.Selector(default="histplot", objects=_HIST_PLOT_TYPES)
    hist_bins = param.Integer(default=20, bounds=(2, 40), doc="Number of bins")
    hist_bin_width = param.Number(
        default=0.0, bounds=(0.0, None), doc="Bin width (0 = use bins slider)"
    )
    # --- histogram_2d / jointplot ---
    joint_show_marginals = param.Boolean(default=True, doc="Show marginal distributions")
    joint_kind = param.Selector(default="scatter", objects=_JOINT_KINDS, doc="Joint plot kind")
    # --- custom ---
    custom_dims = param.List(
        default=[], item_type=str, doc="Dims passed as data_vis to custom_plot"
    )

    def __init__(self, data: Data, **params: object) -> None:
        # Resolve available dims before super().__init__ so Selectors
        # have their ``objects`` populated.
        xr_data = data.data
        dims = list(xr_data.dims)
        if len(dims) < 2:
            msg = "Data must have at least 2 dimensions for exploration"
            raise ValueError(msg)

        # Pick sensible defaults: sweep = 'time' if present, else first dim.
        default_sweep = "time" if "time" in dims else dims[0]
        remaining = [d for d in dims if d != default_sweep]
        default_trace = remaining[0]

        params.setdefault("sweep_dim", default_sweep)
        params.setdefault("trace_dim", default_trace)

        # Pre-configure Selector objects *before* super().__init__ so that
        # param validation passes when initial values are set.
        self.param.sweep_dim.objects = dims
        self.param.trace_dim.objects = dims

        # Set position and window_size upper bounds before super().__init__
        # validates initial values.
        sweep = params.get("sweep_dim", default_sweep)
        trace = params.get("trace_dim", default_trace)
        n = xr_data.sizes[sweep]
        self.param.position.bounds = (0, n - 1)
        self.param.window_size.bounds = (min(10, n), n)

        # Cap window_size to actual signal length (and respect lower bound).
        params["window_size"] = max(min(int(params.get("window_size", 200)), n), min(10, n))  # type: ignore

        # Initialise extra-dimension indices to zero for each non-sweep/trace dim.
        extra_dims_init = [d for d in dims if d not in (sweep, trace)]
        params.setdefault("extra_dim_indices", dict.fromkeys(extra_dims_init, 0))
        params.setdefault("extra_dim_aggregate", {d: False for d in dims if d != sweep})

        # Compute global data range and use as colormap defaults.
        data_vals = xr_data.to_numpy()
        data_min = float(np.nanmin(data_vals))
        data_max = float(np.nanmax(data_vals))
        params.setdefault("clim_min", data_min)
        params.setdefault("clim_max", data_max)

        super().__init__(**params)

        # Store data (plain attribute, not a param — immutable after init).
        self._data = data
        self._xr = xr_data
        # Exposed to ControlsPanel for slider range bounds.
        self._data_min = data_min
        self._data_max = data_max

        # Initialise scatter trace selection based on default trace dim.
        self._reset_scatter_trace_selection()

    # -- helpers -------------------------------------------------------------

    @property
    def data(self) -> Data:
        """The underlying cobrabox ``Data`` object."""
        return self._data

    @property
    def xr(self) -> xarray.DataArray:
        """Shortcut to the underlying ``xr.DataArray``."""
        return self._xr

    @param.depends("viz_mode", watch=True)  # type: ignore
    def _on_viz_mode_change(self) -> None:
        """When entering boxplot mode ensure at least one dim is aggregated."""
        if self.viz_mode == "boxplot":
            self._ensure_boxplot_aggregate()

    def _ensure_boxplot_aggregate(self) -> None:
        """Auto-aggregate the first non-sweep dim if none are currently aggregated."""
        extra = self._boxplot_extra_dims()
        if extra and not any(self.extra_dim_aggregate.get(d, False) for d in extra):
            self.extra_dim_aggregate = {**self.extra_dim_aggregate, extra[0]: True}

    @param.depends("sweep_dim", watch=True)  # type: ignore
    def _update_position_bounds(self) -> None:
        """Update position/window bounds when sweep_dim changes."""
        n = self._xr.sizes[self.sweep_dim]
        self.param.position.bounds = (0, n - 1)
        self.param.window_size.bounds = (min(2, n), n)
        # Clamp current values into new bounds.
        self.position = min(self.position, n - 1)
        self.window_size = max(min(self.window_size, n), min(2, n))
        # Clear the aggregate flag for the dim that just became sweep_dim so
        # its stale True doesn't re-activate when it moves back to extra dims.
        if self.sweep_dim in self.extra_dim_aggregate:
            self.extra_dim_aggregate = {**self.extra_dim_aggregate, self.sweep_dim: False}
        # Refresh extra-dim indices for new set of extra dims.
        self._reset_extra_dim_indices()
        # Re-check aggregate default if already in boxplot mode.
        if self.viz_mode == "boxplot":
            self._ensure_boxplot_aggregate()
        # Reset scatter selection (sweep change may change available trace coords).
        self._reset_scatter_trace_selection()

    @param.depends("trace_dim", watch=True)  # type: ignore
    def _update_extra_dim_indices(self) -> None:
        """Reset extra_dim_indices and scatter trace selection when trace_dim changes."""
        self._reset_extra_dim_indices()
        self._reset_scatter_trace_selection()

    def _reset_extra_dim_indices(self) -> None:
        """Rebuild extra_dim_indices to match the current sweep/trace selection."""
        extra = [d for d in self._xr.dims if d not in (self.sweep_dim, self.trace_dim)]
        self.extra_dim_indices = {
            d: min(self.extra_dim_indices.get(d, 0), self._xr.sizes[d] - 1) for d in extra
        }

    def get_window_slice(self) -> tuple[int, int]:
        """Return ``(lo, hi)`` index bounds for the detail window (time_series only)."""
        n = self._xr.sizes[self.sweep_dim]
        half = self.window_size // 2
        lo = max(0, self.position - half)
        hi = min(n, self.position + half)
        return lo, hi

    def get_2d(self) -> xarray.DataArray:
        """Return a full 2-D DataArray sliced at extra_dim_indices (no windowing)."""
        return self.full_2d()

    def _extra_dims(self) -> list[str]:
        """Dims that are neither ``sweep_dim`` nor ``trace_dim``."""
        return [str(d) for d in self._xr.dims if d not in (self.sweep_dim, self.trace_dim)]

    def _hist_extra_dims(self) -> list[str]:
        """All dims except ``sweep_dim`` — used as navigable dims in histogram/boxplot mode."""
        return [str(d) for d in self._xr.dims if d != self.sweep_dim]

    # boxplot uses the same set of navigable dims as histogram
    _boxplot_extra_dims = _hist_extra_dims

    def get_hist_values(self) -> np.ndarray:
        """Return flattened values for histogram.

        Slices each non-sweep dimension at its current index unless that
        dimension has ``extra_dim_aggregate`` set to ``True``, in which case
        all values along that dimension are included.
        """
        xr_data = self._xr
        for d in self._hist_extra_dims():
            if not self.extra_dim_aggregate.get(d, False):
                xr_data = xr_data.isel(**{d: self.extra_dim_indices.get(d, 0)})
        return xr_data.to_numpy().ravel()

    def _custom_extra_dims(self) -> list[str]:
        """All dims not in ``custom_dims`` — navigated via arrows in custom mode."""
        return [str(d) for d in self._xr.dims if d not in self.custom_dims]

    def get_custom_data(self) -> np.ndarray:
        """Return numpy array for custom plot.

        Only the dimensions listed in ``custom_dims`` are kept as axes.
        All other dimensions are either:
        - sliced to their current index (default), or
        - reduced by taking the mean when ``extra_dim_aggregate[d]`` is ``True``.
        """
        xr_data = self._xr
        for d in self._custom_extra_dims():
            if self.extra_dim_aggregate.get(d, False):
                xr_data = xr_data.mean(dim=d)
            else:
                xr_data = xr_data.isel(**{d: self.extra_dim_indices.get(d, 0)})
        return xr_data.to_numpy()

    def get_boxplot_df(self) -> pd.DataFrame:
        """Return long-form DataFrame for boxplot with columns [sweep_dim, value].

        Non-sweep dims are sliced at their current index unless marked for
        aggregation in ``extra_dim_aggregate``, in which case all their values
        are included in the distribution shown for each sweep category.
        """
        import pandas as pd

        xr_data = self._xr
        for d in self._boxplot_extra_dims():
            if not self.extra_dim_aggregate.get(d, False):
                xr_data = xr_data.isel(**{d: self.extra_dim_indices.get(d, 0)})
        sweep_coords = xr_data.coords[self.sweep_dim].to_numpy()
        rows = []
        for si, slabel in enumerate(sweep_coords):
            vals = xr_data.isel(**{self.sweep_dim: si}).to_numpy().ravel()  # type: ignore
            for v in vals:
                rows.append({self.sweep_dim: str(slabel), "value": float(v)})
        return pd.DataFrame(rows)

    def _reset_scatter_trace_selection(self) -> None:
        """Populate scatter_trace_selection with sensible defaults.

        ≤ 8 trace indices → all selected; > 8 → first 8 selected.
        """
        labels = [str(v) for v in self._xr.coords[self.trace_dim].to_numpy()]
        if len(labels) <= 8:
            self.scatter_trace_selection = labels[:]
        else:
            self.scatter_trace_selection = labels[:8]

    def get_scatter_df(self) -> pd.DataFrame:
        """Return long-form DataFrame for scatter with columns [sweep_dim, trace_dim, value].

        Extra dims are sliced at their current index unless ``extra_dim_aggregate[d]``
        is True (all values along that dim included).  Only trace_dim indices whose
        string label is in ``scatter_trace_selection`` are returned.
        """
        import pandas as pd

        if not self.scatter_trace_selection:
            raise ValueError("No trace indices selected. Tick at least one in 'Show traces'.")

        xr_data = self._xr
        # Slice extra dims (neither sweep nor trace)
        extra = self._extra_dims()
        for d in extra:
            if not self.extra_dim_aggregate.get(d, False):
                xr_data = xr_data.isel(**{d: self.extra_dim_indices.get(d, 0)})

        sweep_labels = xr_data.coords[self.sweep_dim].to_numpy()
        trace_labels = [str(v) for v in xr_data.coords[self.trace_dim].to_numpy()]
        selected_set = set(self.scatter_trace_selection)

        rows = []
        for ti, tlabel in enumerate(trace_labels):
            if tlabel not in selected_set:
                continue
            for si, slabel in enumerate(sweep_labels):
                val = xr_data.isel(**{self.trace_dim: ti, self.sweep_dim: si}).to_numpy()  # type: ignore
                for v in np.asarray(val).ravel():
                    rows.append({self.sweep_dim: slabel, self.trace_dim: tlabel, "value": float(v)})
        return pd.DataFrame(rows)

    def sweep_coords(self) -> np.ndarray:
        """Coordinate values for the current ``sweep_dim``."""
        return self._xr.coords[self.sweep_dim].to_numpy()

    def windowed_2d(self, lo: int, hi: int) -> xarray.DataArray:
        """Return a 2-D DataArray (trace_dim x sweep_dim) for the window.

        Extra dimensions beyond ``sweep_dim`` and ``trace_dim`` are
        sliced at their current index from ``extra_dim_indices``.
        """
        window = self._xr.isel(**{self.sweep_dim: slice(lo, hi)})  # type: ignore
        extra = self._extra_dims()
        if extra:
            selection = {d: self.extra_dim_indices.get(d, 0) for d in extra}
            window = window.isel(**selection)
        return window

    def full_2d(self) -> xarray.DataArray:
        """Return full 2-D DataArray (trace_dim x sweep_dim).

        Extra dimensions are mean-reduced when ``extra_dim_aggregate[d]`` is
        ``True``, otherwise sliced at their current index.
        """
        xr_data = self._xr
        for d in self._extra_dims():
            if self.extra_dim_aggregate.get(d, False):
                xr_data = xr_data.mean(dim=d)
            else:
                xr_data = xr_data.isel(**{d: self.extra_dim_indices.get(d, 0)})
        return xr_data
