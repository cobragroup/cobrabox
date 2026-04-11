"""Sidebar controls panel for SpaceExplorer.

``ControlsPanel`` builds and owns all Panel widgets (dimension selectors,
mode picker, navigation arrows, colormap sliders, scatter/histogram options,
custom-dim selector, and metadata pane).  It reacts to ``ExplorerState``
changes via ``param.watch`` and pushes user interactions back into the state.
"""

from __future__ import annotations

import panel as pn
import param

from cobrabox.visualization._state import ExplorerState


class ControlsPanel(pn.viewable.Viewer):
    """Navigation widgets, visualization mode selector, and metadata display.

    Provides interactive controls for ``ExplorerState`` parameters and
    shows metadata from the underlying ``Data`` object.
    """

    def __init__(self, state: ExplorerState, **params: object) -> None:
        super().__init__(**params)
        self._state = state

        with pn.config.set(sizing_mode="stretch_width"):
            # Visualization mode selector
            self._mode_widget = pn.widgets.Select.from_param(
                state.param.viz_mode, name="Visualization mode"
            )

            # Dimension selectors
            self._sweep_widget = pn.widgets.Select.from_param(
                state.param.sweep_dim, name="Sweep dimension (X axis)"
            )
            self._trace_widget = pn.widgets.Select.from_param(
                state.param.trace_dim, name="Trace dimension (Y)"
            )

            # Swap X / Y button
            self._swap_btn = pn.widgets.Button(
                name="\u21c5  Swap X / Y axes", button_type="light", sizing_mode="stretch_width"
            )

            def _on_swap(event: object) -> None:
                old_sweep, old_trace = state.sweep_dim, state.trace_dim
                if old_sweep == old_trace:
                    return
                # Atomically swap both params in one batch so watchers see
                # the final consistent state and from_param widgets auto-sync.
                state.param.update(sweep_dim=old_trace, trace_dim=old_sweep)

            self._swap_btn.on_click(_on_swap)

            # Extra-dimension navigation (one row of arrows per non-vis dim)
            self._extra_dim_column = pn.Column(sizing_mode="stretch_width")
            self._rebuild_extra_dim_controls()

            # Watch dimension/mode/custom_dims changes to rebuild extra-dim nav rows.
            state.param.watch(
                lambda *_: self._rebuild_extra_dim_controls(),
                ["sweep_dim", "trace_dim", "viz_mode", "custom_dims"],
            )
            state.param.watch(
                lambda *_: self._rebuild_scatter_trace_controls(),
                ["sweep_dim", "trace_dim", "viz_mode"],
            )

            # Navigation controls (time_series only)
            self._position_widget = pn.widgets.IntSlider.from_param(
                state.param.position, name="Position"
            )
            self._window_widget = pn.widgets.IntSlider.from_param(
                state.param.window_size, name="Window size"
            )

            # Colormap controls (heatmap mode)
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

            # --- Seaborn shared ---
            self._sns_color_widget = pn.widgets.Select.from_param(
                state.param.sns_color, name="Color"
            )
            # --- Scatter controls ---
            self._scatter_trace_column = pn.Column(sizing_mode="stretch_width")
            self._rebuild_scatter_trace_controls()
            self._scatter_type_widget = pn.widgets.Select.from_param(
                state.param.scatter_plot_type, name="Plot type"
            )
            self._scatter_marker_widget = pn.widgets.Select.from_param(
                state.param.scatter_marker, name="Marker"
            )
            self._scatter_size_widget = pn.widgets.IntSlider.from_param(
                state.param.scatter_marker_size, name="Marker size"
            )
            # --- Boxplot controls ---
            self._box_type_widget = pn.widgets.Select.from_param(
                state.param.box_plot_type, name="Plot type"
            )
            # --- Histogram controls ---
            self._hist_type_widget = pn.widgets.Select.from_param(
                state.param.hist_plot_type, name="Plot type"
            )
            self._hist_bins_widget = pn.widgets.IntSlider.from_param(
                state.param.hist_bins, name="Bins (sets bin count; resets bin width)"
            )
            _data_range = max(state._data_max - state._data_min, 1e-9)
            self._hist_binwidth_widget = pn.widgets.FloatSlider(
                start=0.0,
                end=_data_range,
                step=max(_data_range / 200, 1e-6),
                value=0.0,
                name="Bin width (sets width; resets bin count)",
            )

            def _on_binwidth(e: param.parameterized.Event) -> None:
                state.hist_bin_width = e.new
                if e.new > 0:
                    # Reset bins slider without triggering the bins callback
                    self._hist_bins_widget.value = state.param.hist_bins.bounds[0]

            def _on_bins(e: param.parameterized.Event) -> None:
                if e.new != state.param.hist_bins.bounds[0] or state.hist_bin_width == 0:
                    # Reset bin_width to 0 when user moves the bins slider
                    if state.hist_bin_width != 0:
                        state.hist_bin_width = 0.0
                        self._hist_binwidth_widget.value = 0.0

            self._hist_binwidth_widget.param.watch(_on_binwidth, ["value"])
            self._hist_bins_widget.param.watch(_on_bins, ["value"])
            # --- Histogram 2D / jointplot controls ---
            self._joint_kind_widget = pn.widgets.Select.from_param(
                state.param.joint_kind, name="Kind"
            )
            self._joint_marginals_widget = pn.widgets.Checkbox.from_param(
                state.param.joint_show_marginals, name="Show marginals"
            )

            # --- Custom dim selector ---
            all_dims = list(state.xr.dims)
            self._custom_dim_selector = pn.widgets.MultiChoice(
                name="Dimensions to pass to custom_plot",
                options=all_dims,
                value=list(state.custom_dims),
                sizing_mode="stretch_width",
            )

            def _on_custom_dims(event: param.parameterized.Event) -> None:
                state.custom_dims = list(event.new)
                self._rebuild_extra_dim_controls()

            self._custom_dim_selector.param.watch(_on_custom_dims, ["value"])

            # Metadata
            self._metadata_pane = pn.pane.Markdown(self._metadata_text, sizing_mode="stretch_width")

            # Build layout based on mode
            self._panel = pn.Column()
            self._update_panel_layout()

            # Watch mode changes to rebuild layout
            state.param.watch(lambda *_: self._update_panel_layout(), ["viz_mode"])

    def _rebuild_scatter_trace_controls(self) -> None:
        """Build the trace-selection widget for scatter mode.

        ≤ 8 trace indices → ``CheckBoxGroup`` (all visible at once).
        > 8 trace indices → ``MultiChoice`` (compact, searchable).
        Only built when viz_mode == 'scatter'; cleared otherwise.
        """
        state = self._state
        if state.viz_mode != "scatter":
            self._scatter_trace_column[:] = []
            return
        labels = [str(v) for v in state.xr.coords[state.trace_dim].values]
        current = list(state.scatter_trace_selection)
        # Keep only labels that still exist after a dim change.
        current = [lbl for lbl in current if lbl in labels]
        if not current:
            current = labels[:8] if len(labels) > 8 else labels[:]

        if len(labels) <= 8:
            widget = pn.widgets.CheckBoxGroup(
                name="Show traces", options=labels, value=current, sizing_mode="stretch_width"
            )
        else:
            widget = pn.widgets.MultiChoice(
                name=f"Show traces (out of {len(labels)}, type to filter)",
                options=labels,
                value=current,
                sizing_mode="stretch_width",
            )

        def _on_selection(event: param.parameterized.Event) -> None:
            state.scatter_trace_selection = list(event.new)

        widget.param.watch(_on_selection, ["value"])
        self._scatter_trace_column[:] = [widget]

    def _rebuild_extra_dim_controls(self) -> None:
        """Rebuild the per-extra-dimension prev/next arrow rows.

        In histogram/boxplot/custom mode all dims except ``sweep_dim`` are navigable
        with aggregate checkboxes.  In scatter mode, only extra dims (neither sweep nor
        trace) are shown, also with aggregate checkboxes.
        """
        state = self._state
        is_agg_mode = state.viz_mode in ("histogram", "boxplot", "scatter", "custom", "heatmap")
        # Determine which dims appear as navigation rows
        if state.viz_mode in ("histogram", "boxplot"):
            extra = state._hist_extra_dims()
        elif state.viz_mode == "custom":
            extra = state._custom_extra_dims()
        else:
            extra = state._extra_dims()
        rows: list = []
        for dim in extra:
            n = state.xr.sizes[dim]
            prev_btn = pn.widgets.Button(name="\u25c0", width=42, margin=(2, 2))
            next_btn = pn.widgets.Button(name="\u25b6", width=42, margin=(2, 2))
            idx_label = pn.widgets.StaticText(
                value=f"{dim}: {state.extra_dim_indices.get(dim, 0) + 1} / {n}",
                sizing_mode="stretch_width",
                margin=(5, 5),
            )
            initial_agg = is_agg_mode and state.extra_dim_aggregate.get(dim, False)
            prev_btn.disabled = initial_agg
            next_btn.disabled = initial_agg

            def _make_callbacks(
                d: str,
                label: pn.widgets.StaticText,
                size: int,
                p_btn: pn.widgets.Button,
                n_btn: pn.widgets.Button,
            ) -> tuple:
                def on_prev(event: object) -> None:
                    current = state.extra_dim_indices.get(d, 0)
                    if current > 0:
                        new_val = current - 1
                        state.extra_dim_indices = {**state.extra_dim_indices, d: new_val}
                        label.value = f"{d}: {new_val + 1} / {size}"

                def on_next(event: object) -> None:
                    current = state.extra_dim_indices.get(d, 0)
                    if current < size - 1:
                        new_val = current + 1
                        state.extra_dim_indices = {**state.extra_dim_indices, d: new_val}
                        label.value = f"{d}: {new_val + 1} / {size}"

                def on_agg_change(event: param.parameterized.Event) -> None:
                    state.extra_dim_aggregate = {**state.extra_dim_aggregate, d: event.new}
                    p_btn.disabled = event.new
                    n_btn.disabled = event.new

                return on_prev, on_next, on_agg_change  # noqa: B023

            on_prev, on_next, on_agg_change = _make_callbacks(dim, idx_label, n, prev_btn, next_btn)
            prev_btn.on_click(on_prev)
            next_btn.on_click(on_next)

            nav_row = pn.Row(prev_btn, idx_label, next_btn, sizing_mode="stretch_width")
            if is_agg_mode:
                agg_checkbox = pn.widgets.Checkbox(
                    name=f"Aggregate all '{dim}' values", value=initial_agg
                )
                agg_checkbox.param.watch(on_agg_change, ["value"])
                rows.append(pn.Column(nav_row, agg_checkbox, sizing_mode="stretch_width"))
            else:
                rows.append(nav_row)
        self._extra_dim_column[:] = rows

    def _update_panel_layout(self) -> None:
        """Rebuild the control panel layout based on current viz_mode."""
        mode = self._state.viz_mode
        state = self._state

        # Custom mode has its own dedicated layout.
        if mode == "custom":
            sections: list = [
                pn.pane.Markdown("### Visualization", disable_anchors=True),
                self._mode_widget,
                pn.layout.Divider(),
                pn.pane.Markdown("### Custom plot dimensions", disable_anchors=True),
                self._custom_dim_selector,
            ]
            nav_dims = state._custom_extra_dims()
            if nav_dims:
                sections += [
                    pn.layout.Divider(),
                    pn.pane.Markdown("**Navigate remaining dimensions:**", disable_anchors=True),
                    self._extra_dim_column,
                ]
            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Metadata", disable_anchors=True),
                self._metadata_pane,
            ]
            self._panel[:] = sections
            return

        is_1d_mode = mode in ("histogram", "boxplot")

        sections = [
            pn.pane.Markdown("### Visualization", disable_anchors=True),
            self._mode_widget,
            pn.layout.Divider(),
            pn.pane.Markdown("### Dimensions", disable_anchors=True),
            self._sweep_widget,
        ]

        # In histogram/boxplot mode trace_dim is not used — all other dims are navigable
        if not is_1d_mode:
            sections += [self._trace_widget, self._swap_btn]

        # Extra-dimension navigation
        if mode in ("histogram", "boxplot"):
            extra_count = len(state._hist_extra_dims())
        else:
            extra_count = len(state._extra_dims())
        if extra_count > 0:
            nav_label = (
                "**Navigate dimensions:**" if is_1d_mode else "**Navigate extra dimensions:**"
            )
            sections += [pn.pane.Markdown(nav_label, disable_anchors=True), self._extra_dim_column]

        # Scatter: trace selection widget
        if mode == "scatter":
            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Traces", disable_anchors=True),
                self._scatter_trace_column,
            ]

        # Navigation sliders — time_series only
        if mode == "time_series":
            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Navigation", disable_anchors=True),
                self._position_widget,
                self._window_widget,
            ]

        # Mode-specific options
        if mode == "heatmap":
            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Colormap", disable_anchors=True),
                self._cmap_widget,
                self._clim_range_slider,
            ]
        elif mode == "scatter":
            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Scatter options", disable_anchors=True),
                self._sns_color_widget,
                self._scatter_type_widget,
                self._scatter_marker_widget,
                self._scatter_size_widget,
            ]
        elif mode == "boxplot":
            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Boxplot options", disable_anchors=True),
                self._sns_color_widget,
                self._box_type_widget,
            ]
        elif mode == "histogram":
            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Histogram options", disable_anchors=True),
                self._sns_color_widget,
                self._hist_type_widget,
                self._hist_bins_widget,
                self._hist_binwidth_widget,
            ]
        elif mode == "histogram_2d":
            sections += [
                pn.layout.Divider(),
                pn.pane.Markdown("### Joint plot options", disable_anchors=True),
                self._sns_color_widget,
                self._joint_kind_widget,
                self._joint_marginals_widget,
            ]

        sections += [
            pn.layout.Divider(),
            pn.pane.Markdown("### Metadata", disable_anchors=True),
            self._metadata_pane,
        ]

        self._panel[:] = sections

    # -- reactive content -------------------------------------------------------

    @param.depends("_state.sweep_dim", "_state.trace_dim")  # type: ignore
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

    def __panel__(self) -> pn.viewable.Viewable:
        return self._panel
