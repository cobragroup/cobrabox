"""Panel-based interactive data explorer for cobrabox Data objects.

Provides a modular, reactive visualization app that works in both
Jupyter notebooks (inline) and served mode (``panel serve``).

Classes:
    ExplorerState: Shared reactive state (dimensions, position, window).
    OverviewPlot:  Reduced signal overview with position marker.
    DetailPlot:    Windowed trace view (line traces or heatmap).
    ControlsPanel: Widgets for navigation, dim selection, and metadata.
    SeizureExplorer:  Top-level shell composing the above components.

Example — notebook::

    import cobrabox as cb
    from cobrabox.visualization import SeizureExplorer

    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    SeizureExplorer(data)

Example — served::

    panel serve src/cobrabox/visualization/seizure_explorer.py --show --dev
"""

from __future__ import annotations

import matplotlib
import numpy as np
import param

matplotlib.use("agg")

import holoviews as hv
import panel as pn

from cobrabox.data import Data

hv.extension("bokeh")

# ---------------------------------------------------------------------------
# x-range linker — shares horizontal axis between channel and average plots
# ---------------------------------------------------------------------------


class _XRangeLinker:
    """Shares a single Bokeh ``x_range`` between two HoloViews figures.

    Inject one instance into both ``ChannelPlot`` and ``AveragePlot`` via
    their ``linker=`` constructor argument.  On first render the Bokeh
    ``Range1d`` is captured; subsequent renders of either plot reuse it so
    that horizontal panning stays in sync while each plot keeps its own
    ``y_range`` for fully independent vertical zoom.

    Call ``reset()`` whenever the sweep dimension changes so that the new
    coordinate range is adopted on the next render pair.
    """

    def __init__(self) -> None:
        self._x_range: object = None

    def reset(self) -> None:
        """Forget the stored range so the next render re-adopts it."""
        self._x_range = None

    def hook(self, plot: object, element: object) -> None:
        """Bokeh post-render hook: capture or reuse the shared x_range."""
        if self._x_range is None:
            self._x_range = plot.handles["x_range"]
        else:
            plot.handles["plot"].x_range = self._x_range


# ---------------------------------------------------------------------------
# Shared reactive state
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


class ExplorerState(param.Parameterized):
    """Shared reactive state for all explorer components.

    Holds the ``Data`` object and parameters that drive every view:
    dimension selectors, current position, window size, display mode,
    and optional event markers.

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
    point_positions : np.ndarray | None
        1-D array of event marker indices along ``sweep_dim``.
    event_index : int
        Index into ``point_positions`` for the currently selected event.
    """

    sweep_dim = param.Selector(doc="Dimension mapped to X axis")
    trace_dim = param.Selector(doc="Dimension mapped to traces / rows")
    position = param.Integer(default=0, bounds=(0, 1), doc="Index along sweep_dim")
    window_size = param.Integer(default=200, bounds=(10, 1000), doc="Detail window width")
    zoom_window = param.Boolean(default=True, doc="Zoom heatmap x-axis to the current window")
    point_positions = param.Array(default=None, doc="1-D event marker indices")
    event_index = param.Integer(default=0, bounds=(0, 0), doc="Current event marker index")
    # Colormap controls
    cmap = param.Selector(default="RdBu_r", objects=_COLORMAPS, doc="Heatmap colormap")
    clim_min = param.Number(default=0.0, doc="Colormap lower bound")
    clim_max = param.Number(default=1.0, doc="Colormap upper bound")

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
        n = xr_data.sizes[sweep]
        self.param.position.bounds = (0, n - 1)
        self.param.window_size.bounds = (10, n)

        # Set event_index bounds if point_positions given.
        pp = params.get("point_positions")
        if pp is not None and len(pp) > 0:
            self.param.event_index.bounds = (0, len(pp) - 1)

        # Compute global data range and use as colormap defaults.
        data_vals = xr_data.values
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

    # -- helpers -------------------------------------------------------------

    @property
    def data(self) -> Data:
        """The underlying cobrabox ``Data`` object."""
        return self._data

    @property
    def xr(self):
        """Shortcut to the underlying ``xr.DataArray``."""
        return self._xr

    @param.depends("sweep_dim", watch=True)
    def _update_position_bounds(self) -> None:
        n = self._xr.sizes[self.sweep_dim]
        self.param.position.bounds = (0, n - 1)
        self.param.window_size.bounds = (min(2, n), n)
        # Clamp current values into new bounds.
        self.position = min(self.position, n - 1)
        self.window_size = max(min(self.window_size, n), min(2, n))

    def _update_event_bounds(self) -> None:
        pp = self.point_positions
        if pp is not None and len(pp) > 0:
            self.param.event_index.bounds = (0, len(pp) - 1)
        else:
            self.param.event_index.bounds = (0, 0)

    @param.depends("event_index", watch=True)
    def _sync_event_position(self) -> None:
        """Keep ``position`` in sync with the selected event index."""
        pp = self.point_positions
        if pp is not None and 0 <= self.event_index < len(pp):
            self.position = int(pp[self.event_index])

    def get_window_slice(self) -> tuple[int, int]:
        """Return ``(lo, hi)`` index bounds for the detail window."""
        n = self._xr.sizes[self.sweep_dim]
        half = self.window_size // 2
        lo = max(0, self.position - half)
        hi = min(n, self.position + half)
        return lo, hi

    def _extra_dims(self) -> list[str]:
        """Dims that are neither ``sweep_dim`` nor ``trace_dim``."""
        return [d for d in self._xr.dims if d not in (self.sweep_dim, self.trace_dim)]

    def reduced_data(self) -> np.ndarray:
        """Mean over all non-sweep dims → 1-D array along ``sweep_dim``."""
        reduce_dims = [d for d in self._xr.dims if d != self.sweep_dim]
        return self._xr.mean(dim=reduce_dims).values

    def sweep_coords(self) -> np.ndarray:
        """Coordinate values for the current ``sweep_dim``."""
        return self._xr.coords[self.sweep_dim].values

    def windowed_2d(self, lo: int, hi: int):
        """Return a 2-D DataArray (trace_dim × sweep_dim) for the window.

        Extra dimensions beyond ``sweep_dim`` and ``trace_dim`` are
        averaged out so that detail plots always receive 2-D data.
        """
        window = self._xr.isel(**{self.sweep_dim: slice(lo, hi)})
        extra = self._extra_dims()
        if extra:
            window = window.mean(dim=extra)
        return window


# ---------------------------------------------------------------------------
# Channel plot (top) — full-length offset traces with window indicator
# ---------------------------------------------------------------------------


class ChannelPlot(pn.viewable.Viewer):
    """Full-length offset line traces for each entry along ``trace_dim``.

    Shows every channel as a vertically-offset time series over the
    entire sweep dimension.  A red vertical line marks the current
    ``position`` and a shaded band shows the detail-window extent.

    Position and window changes trigger a re-render so the span
    annotations are always up-to-date (works in both Jupyter and
    ``panel serve``).  Horizontal zoom is preserved by the shared
    ``_XRangeLinker``.
    """

    def __init__(
        self, state: ExplorerState, *, linker: _XRangeLinker | None = None, **params: object
    ) -> None:
        super().__init__(**params)
        self._state = state
        self._linker = linker
        # Bokeh model references set inside the render hook.
        self._bokeh_vline: object = None  # bokeh.models.Span
        self._bokeh_vspan: object = None  # bokeh.models.BoxAnnotation
        self._pane = pn.pane.HoloViews(
            self._plot, sizing_mode="stretch_width", min_height=300, linked_axes=linker is None
        )

    def _make_hook(self, s: ExplorerState):
        """Return a Bokeh post-render hook that adds span annotations and tap handler."""
        from bokeh.events import Tap as BokehTap
        from bokeh.models import BoxAnnotation, Span

        def hook(plot: object, element: object) -> None:
            fig = plot.handles["plot"]
            coords = s.sweep_coords()
            pos_coord = float(coords[s.position]) if s.position < len(coords) else float(coords[-1])
            lo, hi = s.get_window_slice()
            lo_c = float(coords[lo]) if lo < len(coords) else float(coords[-1])
            hi_c = float(coords[min(hi, len(coords) - 1)])

            # If the spans are already attached to *this* figure, just update
            # their positions — do not add a second set (avoids ghost indicators
            # when the hook is called more than once on the same Bokeh figure).
            if self._bokeh_vline is not None and self._bokeh_vline in fig.center:
                self._bokeh_vline.location = pos_coord
                self._bokeh_vspan.left = lo_c
                self._bokeh_vspan.right = hi_c
            else:
                vline = Span(
                    location=pos_coord,
                    dimension="height",
                    line_color="red",
                    line_width=1.5,
                    line_dash="dashed",
                )
                vspan = BoxAnnotation(left=lo_c, right=hi_c, fill_color="red", fill_alpha=0.12)
                fig.add_layout(vline)
                fig.add_layout(vspan)
                self._bokeh_vline = vline
                self._bokeh_vspan = vspan

            # Register Bokeh-level tap handler once per figure.  Using
            # fig.on_event works in both Jupyter (Panel ipywidget comms) and
            # panel-serve (Bokeh server websocket) without re-rendering the
            # HoloViews overlay.
            if not getattr(fig, "_cb_tap_registered", False):
                fig._cb_tap_registered = True

                def _on_tap(event: object) -> None:
                    crds = s.sweep_coords()
                    idx = int(np.argmin(np.abs(crds - event.x)))
                    new_pos = max(0, min(idx, len(crds) - 1))
                    doc = pn.state.curdoc
                    if doc and doc.session_context:
                        doc.add_next_tick_callback(
                            lambda: setattr(self._state, "position", new_pos)
                        )
                    else:
                        self._state.position = new_pos

                fig.on_event(BokehTap, _on_tap)

            if self._linker is not None:
                self._linker.hook(plot, element)

        return hook

    @param.depends("_state.sweep_dim", "_state.trace_dim", "_state.position", "_state.window_size")
    def _plot(self) -> hv.NdOverlay:
        s = self._state
        coords = s.sweep_coords()
        xr_data = s.xr

        # Reduce any extra dims so we always have 2-D (trace × sweep).
        extra = s._extra_dims()
        data_2d = xr_data.mean(dim=extra) if extra else xr_data

        # Spacing from global data range.
        all_vals = xr_data.values
        data_range = float(np.nanmax(all_vals) - np.nanmin(all_vals))
        if data_range == 0:
            data_range = 1.0
        spacing = 1.2 * data_range

        # Build one Curve per trace over the full signal.
        traces = {}
        trace_labels = data_2d.coords[s.trace_dim].values
        for i, label in enumerate(trace_labels):
            vals = data_2d.isel(**{s.trace_dim: i}).values + i * spacing
            traces[str(label)] = hv.Curve((coords, vals), kdims=[s.sweep_dim], vdims=["amplitude"])

        # Y-ticks: one per channel, placed at its baseline offset, labelled by
        # the coordinate value (or falling back to 1-based index).
        yticks = [(i * spacing, str(label)) for i, label in enumerate(trace_labels)]

        return hv.NdOverlay(traces, kdims=[s.trace_dim]).opts(
            hv.opts.Curve(color=hv.Cycle("Category10"), line_width=0.8),
            hv.opts.NdOverlay(
                responsive=True,
                min_height=300,
                title=f"Channels ({s.trace_dim})",
                xlabel=s.sweep_dim,
                ylabel=s.trace_dim,
                yticks=yticks,
                show_legend=False,
                tools=["tap"],
                hooks=[self._make_hook(s)],
            ),
        )

    def __panel__(self):
        return self._pane


# ---------------------------------------------------------------------------
# Heatmap panel (middle) — windowed 2-D colour map
# ---------------------------------------------------------------------------


class HeatmapPanel(pn.viewable.Viewer):
    """Windowed 2-D heatmap showing ``trace_dim`` × ``sweep_dim``.

    Covers exactly the selected window extent so the data fills the
    full plot area.
    """

    def __init__(
        self, state: ExplorerState, *, linker: _XRangeLinker | None = None, **params: object
    ) -> None:
        super().__init__(**params)
        self._state = state
        self._linker = linker
        self._pane = pn.pane.HoloViews(
            self._plot,
            sizing_mode="stretch_width",
            min_height=250,
            linked_axes=not state.zoom_window,  # unlink only when zoom is active
        )

        # When zoom_window changes: flip linked_axes then force a re-render.
        def _on_zoom_toggle(_e: param.parameterized.Event) -> None:
            self._pane.linked_axes = not self._state.zoom_window
            self._pane.param.trigger("object")

        state.param.watch(_on_zoom_toggle, ["zoom_window"])

    @param.depends(
        "_state.position",
        "_state.window_size",
        "_state.zoom_window",
        "_state.sweep_dim",
        "_state.trace_dim",
        "_state.cmap",
        "_state.clim_min",
        "_state.clim_max",
    )
    def _plot(self) -> hv.Image:
        s = self._state
        lo, hi = s.get_window_slice()
        coords = s.sweep_coords()[lo:hi]

        window = s.windowed_2d(lo, hi)
        trace_labels = window.coords[s.trace_dim].values

        # 2-D array: (trace, sweep)
        matrix = np.stack(
            [window.isel(**{s.trace_dim: i}).values for i in range(len(trace_labels))]
        )

        x0, x1 = float(coords[0]), float(coords[-1])
        y0, y1 = -0.5, len(trace_labels) - 0.5

        # Always set xlim explicitly — HoloViews auto-fits to the image bounds
        # (the window slice) otherwise, making it look zoomed in regardless of
        # the toggle.  When off, expand to the full signal range so the window
        # appears at its natural position; when on, constrain to the window for
        # the magnifying-glass effect.
        full_coords = s.sweep_coords()
        full_xlim = (float(full_coords[0]), float(full_coords[-1]))
        xlim = (x0, x1) if s.zoom_window else full_xlim

        clim = (s.clim_min, s.clim_max)

        # Share x-range with channel/average plots only when NOT zoomed in;
        # when zoomed the heatmap intentionally shows a different x extent.
        hooks = [self._linker.hook] if (self._linker is not None and not s.zoom_window) else []

        return hv.Image(
            matrix, bounds=(x0, y0, x1, y1), kdims=[s.sweep_dim, s.trace_dim], vdims=["amplitude"]
        ).opts(
            cmap=s.cmap,
            clim=clim,
            colorbar=True,
            responsive=True,
            min_height=250,
            title=f"Window heatmap ({hi - lo} samples)",
            xlabel=s.sweep_dim,
            ylabel=s.trace_dim,
            xlim=xlim,
            hooks=hooks,
        )

    def __panel__(self):
        return self._pane


# ---------------------------------------------------------------------------
# Average plot (bottom) — mean signal navigator
# ---------------------------------------------------------------------------


class AveragePlot(pn.viewable.Viewer):
    """Mean signal over all non-sweep dims with position marker and events.

    Provides a compact navigation overview at the bottom showing
    where the current window sits in the full signal.

    Position and window changes trigger a re-render so the span
    annotations are always up-to-date.
    """

    def __init__(
        self, state: ExplorerState, *, linker: _XRangeLinker | None = None, **params: object
    ) -> None:
        super().__init__(**params)
        self._state = state
        self._linker = linker
        self._bokeh_vline: object = None  # bokeh.models.Span
        self._bokeh_vspan: object = None  # bokeh.models.BoxAnnotation
        self._pane = pn.pane.HoloViews(
            self._plot, sizing_mode="stretch_width", min_height=180, linked_axes=linker is None
        )

    def _make_hook(self, s: ExplorerState):
        """Return a Bokeh post-render hook that adds span annotations and tap handler."""
        from bokeh.events import Tap as BokehTap
        from bokeh.models import BoxAnnotation, Span

        def hook(plot: object, element: object) -> None:
            fig = plot.handles["plot"]
            coords = s.sweep_coords()
            pos_coord = float(coords[s.position]) if s.position < len(coords) else float(coords[-1])
            lo, hi = s.get_window_slice()
            lo_c = float(coords[lo]) if lo < len(coords) else float(coords[-1])
            hi_c = float(coords[min(hi, len(coords) - 1)])

            if self._bokeh_vline is not None and self._bokeh_vline in fig.center:
                self._bokeh_vline.location = pos_coord
                self._bokeh_vspan.left = lo_c
                self._bokeh_vspan.right = hi_c
            else:
                vline = Span(location=pos_coord, dimension="height", line_color="red", line_width=2)
                vspan = BoxAnnotation(left=lo_c, right=hi_c, fill_color="red", fill_alpha=0.08)
                fig.add_layout(vline)
                fig.add_layout(vspan)
                self._bokeh_vline = vline
                self._bokeh_vspan = vspan

            # Register Bokeh-level tap handler once per figure.
            if not getattr(fig, "_cb_tap_registered", False):
                fig._cb_tap_registered = True

                def _on_tap(event: object) -> None:
                    crds = s.sweep_coords()
                    idx = int(np.argmin(np.abs(crds - event.x)))
                    new_pos = max(0, min(idx, len(crds) - 1))
                    doc = pn.state.curdoc
                    if doc and doc.session_context:
                        doc.add_next_tick_callback(
                            lambda: setattr(self._state, "position", new_pos)
                        )
                    else:
                        self._state.position = new_pos

                fig.on_event(BokehTap, _on_tap)

            if self._linker is not None:
                self._linker.hook(plot, element)

        return hook

    @param.depends(
        "_state.sweep_dim",
        "_state.trace_dim",
        "_state.point_positions",
        "_state.position",
        "_state.window_size",
    )
    def _plot(self) -> hv.Overlay:
        s = self._state
        coords = s.sweep_coords()
        reduced = s.reduced_data()

        curve = hv.Curve((coords, reduced), kdims=[s.sweep_dim], vdims=["amplitude"]).opts(
            color="steelblue",
            line_width=1.0,
            tools=["hover", "tap"],
            xlabel=s.sweep_dim,
            ylabel="Mean amplitude",
            title=f"Average (mean over {s.trace_dim})",
            hooks=[self._make_hook(s)],
        )

        overlay = curve
        pp = s.point_positions
        if pp is not None and len(pp) > 0:
            valid = pp[pp < len(coords)]
            if len(valid) > 0:
                event_coords = coords[valid]
                event_vals = reduced[valid]
                scatter = hv.Scatter(
                    (event_coords, event_vals), kdims=[s.sweep_dim], vdims=["amplitude"]
                ).opts(color="orange", size=8, marker="circle")
                overlay = overlay * scatter

        return overlay.opts(responsive=True, min_height=180)

    def __panel__(self):
        return self._pane


# ---------------------------------------------------------------------------
# Controls panel (sidebar)
# ---------------------------------------------------------------------------


class ControlsPanel(pn.viewable.Viewer):
    """Navigation widgets, dimension selectors, and metadata display.

    Provides interactive controls for ``ExplorerState`` parameters and
    shows metadata from the underlying ``Data`` object (subject ID,
    sampling rate, shape, history).
    """

    def __init__(self, state: ExplorerState, **params: object) -> None:
        super().__init__(**params)
        self._state = state

        with pn.config.set(sizing_mode="stretch_width"):
            # Dimension selectors
            self._sweep_widget = pn.widgets.Select.from_param(
                state.param.sweep_dim, name="Sweep dimension (X axis)"
            )
            self._trace_widget = pn.widgets.Select.from_param(
                state.param.trace_dim, name="Trace dimension"
            )

            # Navigation
            self._position_widget = pn.widgets.IntSlider.from_param(
                state.param.position, name="Position"
            )
            self._window_widget = pn.widgets.IntSlider.from_param(
                state.param.window_size, name="Window size"
            )
            self._zoom_widget = pn.widgets.Checkbox.from_param(
                state.param.zoom_window, name="Zoom to window"
            )

            # Colormap controls
            self._cmap_widget = pn.widgets.Select.from_param(state.param.cmap, name="Colormap")
            _step = max((state._data_max - state._data_min) / 200, 1e-6)
            self._clim_range_slider = pn.widgets.EditableRangeSlider(
                name="Color range",
                start=state._data_min,
                end=state._data_max,
                fixed_start=state._data_min,
                fixed_end=state._data_max,
                value=(state.clim_min, state.clim_max),
                step=_step,
            )

            def _on_clim_range(e: param.parameterized.Event) -> None:
                state.clim_min, state.clim_max = e.new

            self._clim_range_slider.param.watch(_on_clim_range, ["value"])

            # Event navigation buttons
            self._prev_btn = pn.widgets.Button(name="◀ Previous", button_type="default")
            self._next_btn = pn.widgets.Button(name="Next ▶", button_type="default")
            self._prev_btn.on_click(self._on_prev)
            self._next_btn.on_click(self._on_next)
            self._event_row = pn.Row(self._prev_btn, self._next_btn)

            # Event info
            self._event_info = pn.pane.Markdown(self._event_info_text, sizing_mode="stretch_width")

            # Metadata
            self._metadata_pane = pn.pane.Markdown(self._metadata_text, sizing_mode="stretch_width")

            # Layout
            sections: list = [
                # pn.pane.Markdown("### Dimensions", disable_anchors=True),
                # self._sweep_widget,
                # self._trace_widget,
                # pn.layout.Divider(),
                pn.pane.Markdown("### Navigation", disable_anchors=True),
                self._position_widget,
                self._window_widget,
                self._zoom_widget,
            ]

            # Only show event controls when events exist.
            if state.point_positions is not None and len(state.point_positions) > 0:
                sections += [
                    pn.layout.Divider(),
                    pn.pane.Markdown("### Events", disable_anchors=True),
                    self._event_row,
                    self._event_info,
                ]

            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Colormap", disable_anchors=True),
                self._cmap_widget,
                self._clim_range_slider,
                pn.layout.Divider(),
                pn.pane.Markdown("### Metadata", disable_anchors=True),
                self._metadata_pane,
            ]

            self._panel = pn.Column(*sections)

    # -- reactive content ----------------------------------------------------

    @param.depends("_state.sweep_dim", "_state.trace_dim")
    def _metadata_text(self) -> str:
        d = self._state.data
        lines = [
            f"**Subject:** {d.subjectID or '—'}",
            f"**Sampling rate:** {d.sampling_rate or '—'}",
            f"**Shape:** {dict(self._state.xr.sizes)}",
            f"**Dims:** {list(self._state.xr.dims)}",
        ]
        if d.history:
            lines.append(f"**History:** {', '.join(d.history)}")
        return "\n\n".join(lines)

    @param.depends("_state.event_index", "_state.point_positions")
    def _event_info_text(self) -> str:
        pp = self._state.point_positions
        if pp is None or len(pp) == 0:
            return ""
        idx = self._state.event_index
        return f"Event **{idx + 1}** of **{len(pp)}** (index {pp[idx]})"

    # -- button callbacks ----------------------------------------------------

    def _on_prev(self, event) -> None:  # noqa: ANN001
        s = self._state
        if s.point_positions is not None and s.event_index > 0:
            s.event_index -= 1

    def _on_next(self, event) -> None:  # noqa: ANN001
        s = self._state
        pp = s.point_positions
        if pp is not None and s.event_index < len(pp) - 1:
            s.event_index += 1

    def __panel__(self):
        return self._panel


# ---------------------------------------------------------------------------
# Top-level explorer
# ---------------------------------------------------------------------------


class SeizureExplorer(pn.viewable.Viewer):
    """Interactive data explorer for cobrabox ``Data`` objects.

    Composes three stacked plots around a shared ``ExplorerState``:

    - **Top** — ``ChannelPlot``: full-length offset line traces.
    - **Middle** — ``HeatmapPanel``: windowed 2-D heatmap.
    - **Bottom** — ``AveragePlot``: mean signal navigator.

    Works in Jupyter notebooks (via ``__panel__``) and as a served
    app (via ``create_app``).

    Parameters
    ----------
    data : Data
        Cobrabox ``Data`` object with ≥ 2 dimensions.
    point_positions : np.ndarray | None
        1-D event marker indices along the sweep dimension.
    window_size : int
        Initial detail-window width in samples.
    position : int
        Initial position index along the sweep dimension.

    Example
    -------
    >>> import cobrabox as cb
    >>> data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    >>> explorer = SeizureExplorer(data)
    >>> explorer  # renders in notebook
    """

    def __init__(
        self,
        data: Data,
        *,
        point_positions: np.ndarray | None = None,
        window_size: int = 200,
        position: int = 0,
        **params: object,
    ) -> None:
        super().__init__(**params)

        # Build shared state.
        self._state = ExplorerState(
            data, position=position, window_size=window_size, point_positions=point_positions
        )

        # Build child components.
        with pn.config.set(sizing_mode="stretch_width"):
            self._controls = ControlsPanel(self._state)
            # Shared x-range linker keeps channel and average plots in sync
            # horizontally but lets each plot zoom/pan vertically on its own.
            self._x_linker = _XRangeLinker()
            self._channels = ChannelPlot(self._state, linker=self._x_linker)
            self._heatmap = HeatmapPanel(self._state, linker=self._x_linker)
            self._average = AveragePlot(self._state, linker=self._x_linker)
            # Reset shared range when sweep dim changes (new time axis coords).
            self._state.param.watch(lambda _e: self._x_linker.reset(), ["sweep_dim"])

            # Three plots stacked; controls in a fixed-width sidebar.
            self._layout = pn.Row(
                pn.Column(self._controls, width=300),
                pn.Column(self._channels, self._heatmap, self._average, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            )

    def __panel__(self):
        return self._layout

    @classmethod
    def create_app(
        cls,
        data: Data,
        *,
        point_positions: np.ndarray | None = None,
        window_size: int = 200,
        position: int = 0,
    ) -> pn.template.FastListTemplate:
        """Build a served Panel app with sidebar and main area.

        Returns a ``FastListTemplate`` ready to call ``.servable()``.
        """
        instance = cls(
            data, point_positions=point_positions, window_size=window_size, position=position
        )
        return pn.template.FastListTemplate(
            title="Seizure Explorer",
            sidebar=[instance._controls],
            main=[
                pn.Column(
                    instance._channels,
                    instance._heatmap,
                    instance._average,
                    sizing_mode="stretch_both",
                )
            ],
            main_layout=None,
        )


# ---------------------------------------------------------------------------
# Served entry-point
# ---------------------------------------------------------------------------

if pn.state.served:
    from cobrabox.dataset_loader import load_noise_dummy

    _demo = load_noise_dummy()[4]  # dummy_noise_simulated_data_5.csv.xz (0-based index)

    SeizureExplorer.create_app(_demo).servable()
