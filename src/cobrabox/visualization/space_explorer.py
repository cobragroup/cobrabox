"""Panel-based interactive data explorer for cobrabox Data objects.

Provides a modular, reactive visualization app that works in both
Jupyter notebooks (inline) and served mode (``panel serve``).

In contrast to the more specialized ``SeizureExplorer`` in
``cobrabox.visualization.seizure_explorer``, this ``SpaceExplorer``
is designed to work with any cobrabox ``Data`` object with ≥ 2 dimensions,
making it suitable for exploring spatial patterns in EEG, fMRI,
or other multichannel data. It provides flexible visualization modes
(time series or heatmap) and focuses on dimension selection and windowed views.

Module layout
-------------
_state.py     — ExplorerState + constants (no Panel/HoloViews/Matplotlib)
_controls.py  — ControlsPanel (sidebar widgets)
space_explorer.py — DataPlot + SpaceExplorer (this file) + serve entry-point

Classes:
    ExplorerState: Shared reactive state (dimensions, position, window, viz mode).
    DataPlot: Single plot that switches between time_series and heatmap visuals.
    ControlsPanel: Widgets for navigation, mode selection, and metadata.
    SpaceExplorer: Top-level shell composing the above components.

Example — notebook::

    import cobrabox as cb
    from cobrabox.visualization import SpaceExplorer

    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    SpaceExplorer(data)

Example — served::

    panel serve src/cobrabox/visualization/space_explorer.py --show --dev
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("agg")

import holoviews as hv  # noqa: I001
import panel as pn

from cobrabox.data import Data
from cobrabox.visualization._controls import ControlsPanel
from cobrabox.visualization._state import (  # re-exported for backwards compat
    ExplorerState,
    _BOX_PLOT_TYPES,  # noqa: F401
    _COLORMAPS,  # noqa: F401
    _HIST_PLOT_TYPES,  # noqa: F401
    _JOINT_KINDS,  # noqa: F401
    _MARKER_TYPES,  # noqa: F401
    _NAMED_COLORS,  # noqa: F401
    _SCATTER_PLOT_TYPES,  # noqa: F401
    _VIZ_MODES,  # noqa: F401
)

if TYPE_CHECKING:
    import pandas as pd

hv.extension("bokeh")  # type: ignore

# Consistent matplotlib styling applied inside all DataPlot._plot_* methods
# via rc_context so as not to affect the user's notebook-wide rcParams.
_MPL_STYLE: dict = {
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# ---------------------------------------------------------------------------
# Data plot — switches between time_series and heatmap modes
# ---------------------------------------------------------------------------


class DataPlot(pn.viewable.Viewer):
    """Single plot showing data in either time_series or heatmap mode.

    Automatically switches visualization based on the ``viz_mode``
    parameter in shared state.
    """

    def __init__(self, state: ExplorerState, **params: object) -> None:
        super().__init__(**params)
        self._state = state
        self._hv_pane = pn.pane.HoloViews(
            self._hv_plot(), sizing_mode="stretch_width", min_height=400
        )
        _init_mpl_fig = self._mpl_plot()
        self._mpl_pane = pn.pane.Matplotlib(
            _init_mpl_fig, sizing_mode="stretch_width", min_height=400, tight=True
        )
        plt.close(_init_mpl_fig)
        self._custom_pane = pn.Column(sizing_mode="stretch_width", min_height=400)
        self._pane = pn.Column(
            self._hv_pane, self._mpl_pane, self._custom_pane, sizing_mode="stretch_width"
        )
        if state.viz_mode == "custom":
            self._refresh_custom()
        self._update_pane_visibility()

        # Drive all updates via explicit param.watch so cross-object reactivity
        # is not reliant on @param.depends resolving a non-param attribute path.
        state.param.watch(
            lambda *_: self._refresh_hv(),
            [
                "viz_mode",
                "sweep_dim",
                "trace_dim",
                "position",
                "window_size",
                "cmap",
                "clim_min",
                "clim_max",
                "extra_dim_indices",
                "extra_dim_aggregate",
            ],
        )
        state.param.watch(
            lambda *_: self._refresh_mpl(),
            [
                "viz_mode",
                "sweep_dim",
                "trace_dim",
                "extra_dim_indices",
                "extra_dim_aggregate",
                "scatter_trace_selection",
                "sns_color",
                "scatter_plot_type",
                "scatter_marker",
                "scatter_marker_size",
                "box_plot_type",
                "hist_plot_type",
                "hist_bins",
                "hist_bin_width",
                "joint_show_marginals",
                "joint_kind",
            ],
        )
        state.param.watch(
            lambda *_: self._refresh_custom(),
            ["viz_mode", "extra_dim_indices", "extra_dim_aggregate", "custom_dims"],
        )
        state.param.watch(lambda *_: self._update_pane_visibility(), ["viz_mode"])

    _HV_MODES = frozenset({"time_series", "heatmap"})

    def _refresh_hv(self) -> None:
        self._hv_pane.object = self._hv_plot()

    def _refresh_mpl(self) -> None:
        if self._state.viz_mode in self._HV_MODES or self._state.viz_mode == "custom":
            return
        fig = self._mpl_plot()
        self._mpl_pane.object = fig
        plt.close(fig)

    def _refresh_custom(self) -> None:
        if self._state.viz_mode != "custom":
            return
        self._custom_pane[:] = [self._build_custom_panel()]

    def _update_pane_visibility(self) -> None:
        mode = self._state.viz_mode
        self._hv_pane.visible = mode in self._HV_MODES
        self._mpl_pane.visible = mode not in self._HV_MODES and mode != "custom"
        self._custom_pane.visible = mode == "custom"

    def _hv_plot(self) -> hv.Element | hv.NdOverlay:
        s = self._state
        if s.sweep_dim == s.trace_dim:
            return hv.Curve([], kdims=["x"], vdims=["y"])
        if s.viz_mode == "time_series":
            return self._plot_time_series()
        if s.viz_mode == "heatmap":
            return self._plot_heatmap()
        return hv.Curve([], kdims=["x"], vdims=["y"])

    def _mpl_plot(self) -> plt.Figure:
        s = self._state
        # In histogram/boxplot mode trace_dim is not used, so the sweep==trace
        # guard must not block rendering (e.g. sweep_dim == default trace_dim).
        if s.sweep_dim == s.trace_dim and s.viz_mode not in ("histogram", "boxplot"):
            fig, ax = plt.subplots()
            ax.axis("off")
            return fig
        try:
            if s.viz_mode == "scatter":
                return self._plot_scatter()
            if s.viz_mode == "boxplot":
                return self._plot_boxplot()
            if s.viz_mode == "histogram":
                return self._plot_histogram()
            if s.viz_mode == "histogram_2d":
                return self._plot_histogram_2d()
        except Exception as exc:
            with matplotlib.rc_context(_MPL_STYLE):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(
                    0.5,
                    0.5,
                    str(exc),
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    wrap=True,
                    color="red",
                )
                ax.axis("off")
            return fig
        fig, ax = plt.subplots()
        ax.axis("off")
        return fig

    def _plot_time_series(self) -> hv.NdOverlay:
        """Render windowed time series with offset traces."""
        s = self._state
        lo, hi = s.get_window_slice()
        coords = s.sweep_coords()[lo:hi]
        window = s.windowed_2d(lo, hi)

        # Spacing from global data range for consistent scaling while navigating.
        all_vals = s.xr.to_numpy()
        data_range = float(np.nanmax(all_vals) - np.nanmin(all_vals))
        if data_range == 0:
            data_range = 1.0
        spacing = 1.2 * data_range

        # Build one Curve per trace over the windowed signal.
        traces = {}
        trace_labels = window.coords[s.trace_dim].to_numpy()
        for i, label in enumerate(trace_labels):
            vals = window.isel(**{s.trace_dim: i}).to_numpy() + i * spacing  # type: ignore
            traces[str(label)] = hv.Curve((coords, vals), kdims=[s.sweep_dim], vdims=["amplitude"])

        # Y-ticks: one per channel at its baseline offset; subsample when many traces.
        _MAX_YTICKS = 20
        if len(trace_labels) <= _MAX_YTICKS:
            yticks = [(i * spacing, str(label)) for i, label in enumerate(trace_labels)]
        else:
            _idx = np.linspace(0, len(trace_labels) - 1, _MAX_YTICKS, dtype=int)
            yticks = [(int(i) * spacing, str(trace_labels[i])) for i in _idx]

        return hv.NdOverlay(traces, kdims=[s.trace_dim]).opts(
            hv.opts.Curve(color=hv.Cycle("Category10"), line_width=0.8),
            hv.opts.NdOverlay(
                responsive=True,
                min_height=400,
                title=f"Time Series ({hi - lo} samples)",
                xlabel=s.sweep_dim,
                ylabel=s.trace_dim,
                yticks=yticks,
                show_legend=False,
            ),
        )

    # ------------------------------------------------------------------
    # Helpers shared by seaborn modes
    # ------------------------------------------------------------------

    def _get_flat_df(self) -> pd.DataFrame:
        """Return a long-form DataFrame with columns [sweep_dim, trace_dim, value].

        Uses full data (no windowing) sliced at extra_dim_indices.
        """
        import pandas as pd

        s = self._state
        data_2d = s.get_2d()  # (trace_dim, sweep_dim) after isel of extras
        sweep_labels = data_2d.coords[s.sweep_dim].to_numpy()
        trace_labels = data_2d.coords[s.trace_dim].to_numpy()
        rows = []
        for ti, tlabel in enumerate(trace_labels):
            vals = data_2d.isel(**{s.trace_dim: ti}).to_numpy()  # type: ignore
            for si, slabel in enumerate(sweep_labels):
                rows.append({s.sweep_dim: slabel, s.trace_dim: tlabel, "value": vals[si]})
        return pd.DataFrame(rows)

    @staticmethod
    def _subsample_ticks(labels: np.ndarray, max_ticks: int = 20) -> list:
        """Return at most ``max_ticks`` (index, label) pairs, evenly spaced.

        Always includes the first and last label. Returns all labels when
        ``len(labels) <= max_ticks``.
        """
        n = len(labels)
        if n <= max_ticks:
            return [(i, str(labels[i])) for i in range(n)]
        indices = np.linspace(0, n - 1, max_ticks, dtype=int)
        seen: set = set()
        result = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                result.append((int(idx), str(labels[idx])))
        return result

    def _make_context_str(self, fixed_dims: list[str]) -> str:
        """Build a compact context annotation showing the current coordinate
        value for each fixed (non-aggregated) extra dimension.

        Returns an empty string when there is nothing to annotate.
        Example: ``"feature: line_length, patient: 0"``
        """
        if not fixed_dims:
            return ""
        s = self._state
        parts = []
        for d in fixed_dims:
            idx = s.extra_dim_indices.get(d, 0)
            if d in s.xr.coords:
                coord_vals = s.xr.coords[d].to_numpy()
                label = str(coord_vals[idx]) if idx < len(coord_vals) else str(idx)
            else:
                label = str(idx)
            parts.append(f"{d}: {label}")
        return ", ".join(parts)

    def _plot_scatter(self) -> plt.Figure:
        """Render scatter / regplot.

        X = sweep_dim categories, series = selected trace_dim indices.
        """
        s = self._state
        df = s.get_scatter_df()
        df[s.sweep_dim] = df[s.sweep_dim].astype(str)
        fixed_dims = [d for d in s._extra_dims() if not s.extra_dim_aggregate.get(d, False)]
        main_title = f"Scatter  ·  {s.sweep_dim} x {s.trace_dim}"
        context = self._make_context_str(fixed_dims)
        title = f"{main_title}\n{context}" if context else main_title
        n_cats = len(df[s.sweep_dim].unique())
        n_traces = df[s.trace_dim].nunique()
        palette = sns.color_palette("tab10", n_colors=max(n_traces, 1))
        with matplotlib.rc_context(_MPL_STYLE):
            fig, ax = plt.subplots(figsize=(min(16, max(7, n_cats * 0.6)), 5))
            if s.scatter_plot_type == "scatterplot":
                sns.scatterplot(
                    data=df,
                    x=s.sweep_dim,
                    y="value",
                    hue=s.trace_dim,
                    marker=s.scatter_marker,
                    s=s.scatter_marker_size,
                    ax=ax,
                    legend="auto",
                    palette=palette,
                )
            else:  # regplot — one line per trace
                for idx, (name, grp) in enumerate(df.groupby(s.trace_dim, sort=False)):
                    sns.regplot(
                        data=grp,
                        x=s.sweep_dim,
                        y="value",
                        marker=s.scatter_marker,
                        scatter_kws={"s": s.scatter_marker_size},
                        color=palette[idx % len(palette)],
                        label=str(name),
                        ax=ax,
                    )
                ax.legend(title=s.trace_dim)
            ax.set_title(title)
            ax.set_xlabel(s.sweep_dim)
            ax.set_ylabel("value")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
        return fig

    def _plot_boxplot(self) -> plt.Figure:
        """Render box/violin/swarm/bar plot with sweep_dim as category."""
        s = self._state
        n_cats = s.xr.sizes[s.sweep_dim]
        if n_cats > 30:
            raise ValueError(
                f"X-axis dimension '{s.sweep_dim}' has {n_cats} indices (> 30). "
                "Choose a dimension with ≤ 30 indices for categorical plots."
            )
        df = s.get_boxplot_df()
        fixed_dims = [d for d in s._boxplot_extra_dims() if not s.extra_dim_aggregate.get(d, False)]
        main_title = f"{s.box_plot_type.capitalize()}  ·  {s.sweep_dim}"
        context = self._make_context_str(fixed_dims)
        title = f"{main_title}\n{context}" if context else main_title
        plot_fn = {
            "boxplot": sns.boxplot,
            "violinplot": sns.violinplot,
            "swarmplot": sns.swarmplot,
            "barplot": sns.barplot,
        }[s.box_plot_type]
        with matplotlib.rc_context(_MPL_STYLE):
            fig, ax = plt.subplots(figsize=(min(14, max(7, n_cats * 0.5)), 5))
            plot_fn(data=df, x=s.sweep_dim, y="value", color=s.sns_color, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(s.sweep_dim)
            ax.set_ylabel("value")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
        return fig

    def _plot_histogram(self) -> plt.Figure:
        """Render histogram / KDE of sweep_dim values."""
        s = self._state
        all_vals = s.get_hist_values()
        fixed_dims = [d for d in s._hist_extra_dims() if not s.extra_dim_aggregate.get(d, False)]
        main_title = f"Histogram  ·  {s.sweep_dim}"
        context = self._make_context_str(fixed_dims)
        title = f"{main_title}\n{context}" if context else main_title
        with matplotlib.rc_context(_MPL_STYLE):
            fig, ax = plt.subplots(figsize=(8, 5))
            if s.hist_plot_type == "histplot":
                kws: dict = {"bins": s.hist_bins, "color": s.sns_color, "ax": ax}
                if s.hist_bin_width > 0:
                    kws["binwidth"] = s.hist_bin_width
                    kws.pop("bins")
                sns.histplot(all_vals, **kws)
            else:  # kdeplot
                sns.kdeplot(all_vals, color=s.sns_color, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("value")
            fig.tight_layout()
        return fig

    def _plot_histogram_2d(self) -> plt.Figure:
        """Render jointplot of sweep_dim vs trace_dim."""
        s = self._state
        df = self._get_flat_df()
        marginal_kws: dict = {} if s.joint_show_marginals else {"fill": False, "color": "none"}
        with matplotlib.rc_context(_MPL_STYLE):
            g = sns.jointplot(
                data=df,
                x=s.sweep_dim,
                y="value",
                kind=s.joint_kind,
                color=s.sns_color,
                marginal_kws=marginal_kws if s.joint_kind in ("kde", "hist") else None,
            )
            if not s.joint_show_marginals:
                g.ax_marg_x.set_visible(False)
                g.ax_marg_y.set_visible(False)
            g.set_axis_labels(s.sweep_dim, "value")
            g.figure.suptitle(f"Joint  ·  {s.sweep_dim} x {s.trace_dim}", y=1.01)
            g.figure.tight_layout()
        return g.figure

    def _build_custom_panel(self) -> pn.viewable.Viewable:
        """Build a Panel viewable from the user-supplied ``custom_plot`` function.

        The function must have the signature ``custom_plot(data_vis: np.ndarray)``.
        It may return any Panel-compatible object:

        * ``plt.Figure`` — handled with ``plt.close`` to avoid Jupyter auto-display
        * ``plotly.graph_objects.Figure`` → rendered via ``pn.pane.Plotly``
        * ``holoviews.Element`` → rendered via ``pn.pane.HoloViews``
        * Any other Panel viewable — used directly

        ``data_vis`` is a numpy array whose axes correspond to ``state.custom_dims``.
        """
        s = self._state
        fn = getattr(s, "_custom_plot", None)
        _sizing: dict = {"sizing_mode": "stretch_width", "min_height": 400}

        def _mpl_placeholder(msg: str, color: str = "black") -> pn.viewable.Viewable:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(
                0.5,
                0.5,
                msg,
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
                wrap=True,
                color=color,
            )
            ax.axis("off")
            pane = pn.pane.Matplotlib(fig, tight=True, **_sizing)
            plt.close(fig)
            return pane

        if fn is None:
            return _mpl_placeholder("No custom_plot function provided.")
        if not s.custom_dims:
            return _mpl_placeholder("Select dimensions to visualize\nin the sidebar.")
        try:
            result = fn(s.get_custom_data())
        except Exception as exc:
            return _mpl_placeholder(str(exc), color="red")
        if isinstance(result, plt.Figure):
            pane = pn.pane.Matplotlib(result, tight=True, **_sizing)
            plt.close(result)
            return pane
        # Generic Panel-compatible object: Plotly, HoloViews, ipywidgets, ...
        return pn.panel(result, **_sizing)  # type: ignore

    def _plot_heatmap(self) -> hv.Image:
        """Render full heatmap (all sweep values, no windowing)."""
        s = self._state
        data_2d = s.full_2d()
        coords = data_2d.coords[s.sweep_dim].to_numpy()
        trace_labels = data_2d.coords[s.trace_dim].to_numpy()
        n_sweep = len(coords)
        n_trace = len(trace_labels)

        # 2-D array: (trace, sweep)
        matrix = np.stack([data_2d.isel(**{s.trace_dim: i}).to_numpy() for i in range(n_trace)])  # type: ignore

        # Use integer pixel-center bounds for both axes so the heatmap renders
        # correctly regardless of whether the dimension coordinates are numeric
        # or categorical (string).  Explicit tick labels below carry the actual
        # coord values back to the axes.
        x0, x1 = -0.5, n_sweep - 0.5
        y0, y1 = -0.5, n_trace - 0.5

        xticks = self._subsample_ticks(coords)
        yticks = self._subsample_ticks(trace_labels)

        clim = (s.clim_min, s.clim_max)

        return hv.Image(
            matrix, bounds=(x0, y0, x1, y1), kdims=[s.sweep_dim, s.trace_dim], vdims=["amplitude"]
        ).opts(
            cmap=s.cmap,
            clim=clim,
            colorbar=True,
            responsive=True,
            min_height=400,
            title=f"Heatmap ({n_sweep} samples)",
            xlabel=s.sweep_dim,
            ylabel=s.trace_dim,
            xticks=xticks,
            yticks=yticks,
        )

    def __panel__(self) -> pn.viewable.Viewable:
        return self._pane


# ---------------------------------------------------------------------------
# Top-level explorer
# ---------------------------------------------------------------------------


class SpaceExplorer(pn.viewable.Viewer):
    """Interactive data explorer for cobrabox ``Data`` objects.

    Provides flexible visualization modes (time_series or heatmap) for
    exploring spatial patterns in multichannel data.

    Works in Jupyter notebooks (via ``__panel__``) and as a served
    app (via ``create_app``).

    Parameters
    ----------
    data : Data
        Cobrabox ``Data`` object with ≥ 2 dimensions.
    window_size : int
        Initial detail-window width in samples.
    position : int
        Initial position index along the sweep dimension.

    Example
    -------
    >>> import cobrabox as cb
    >>> data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    >>> explorer = SpaceExplorer(data)
    >>> explorer  # renders in notebook
    """

    def __init__(
        self,
        data: Data,
        *,
        window_size: int = 200,
        position: int = 0,
        custom_plot: object = None,
        dims: list[str] | None = None,
        **params: object,
    ) -> None:
        """Create a SpaceExplorer.

        Parameters
        ----------
        data:
            Cobrabox ``Data`` object with ≥ 2 dimensions.
        window_size:
            Initial detail-window width in samples.
        position:
            Initial position index along the sweep dimension.
        custom_plot:
            Optional callable ``custom_plot(data_vis: np.ndarray) -> plt.Figure``.
            When provided a ``"custom"`` option is added to the visualization-mode
            dropdown and it becomes the default mode.

            ``data_vis`` is a numpy array whose axes correspond to the dimensions
            listed in ``dims``.  All other dimensions are navigated via arrow
            buttons in the sidebar (with optional aggregation).

            Example::

                def my_viz(data_vis):
                    fig, ax = plt.subplots()
                    ax.imshow(data_vis)
                    return fig

                explorer = SpaceExplorer(data, custom_plot=my_viz, dims=["space"])
        dims:
            List of dimension names from ``data`` that will be kept as axes of the
            array passed to ``custom_plot``.  All other dimensions become navigable
            sliders in the sidebar.

            If omitted while ``custom_plot`` is provided, a ``MultiChoice`` widget
            in the sidebar lets the user pick dimensions interactively before any
            plot is rendered.
        """
        super().__init__(**params)

        # Validate input
        if not isinstance(data, Data):
            raise TypeError(f"Expected Data, got {type(data).__name__}")
        if len(data.data.dims) < 2:
            raise ValueError("Data must have at least 2 dimensions")
        if custom_plot is not None and not callable(custom_plot):
            raise TypeError("custom_plot must be a callable")
        if dims is not None:
            unknown = [d for d in dims if d not in data.data.dims]
            if unknown:
                raise ValueError(f"dims contains unknown dimension(s): {unknown}")

        # Build shared state.
        self._state = ExplorerState(data, position=position, window_size=window_size)

        # If a custom plot function is provided, extend the mode list, attach the function,
        # set custom_dims if provided, and default to custom mode.
        if custom_plot is not None:
            self._state._custom_plot = custom_plot  # type: ignore
            current_modes = list(self._state.param["viz_mode"].objects)
            if "custom" not in current_modes:
                self._state.param["viz_mode"].objects = [*current_modes, "custom"]
            if dims is not None:
                self._state.custom_dims = list(dims)
            self._state.viz_mode = "custom"

        # Build child components.
        with pn.config.set(sizing_mode="stretch_width"):
            self._controls = ControlsPanel(self._state)
            self._plot = DataPlot(self._state)

            # Controls in sidebar, plot fills main area.
            self._layout = pn.Row(
                pn.Column(self._controls, width=300),
                pn.Column(self._plot, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            )

    def __panel__(self) -> pn.viewable.Viewable:
        return self._layout

    @classmethod
    def create_app(
        cls, data: Data, *, window_size: int = 200, position: int = 0
    ) -> pn.template.FastListTemplate:
        """Build a served Panel app with sidebar and main area.

        Returns a ``FastListTemplate`` ready to call ``.servable()``.
        """
        instance = cls(data, window_size=window_size, position=position)
        return pn.template.FastListTemplate(
            title="Space Explorer",
            sidebar=[instance._controls],
            main=[instance._plot],
            main_layout=None,
        )


# ---------------------------------------------------------------------------
# Served entry-point
# ---------------------------------------------------------------------------

if pn.state.served:
    from cobrabox.dataset_loader import load_noise_dummy

    _demo = load_noise_dummy()[0]

    SpaceExplorer.create_app(_demo).servable()
