# SpaceExplorer — Architecture Reference

> **For AI agents:** Read this file FIRST before modifying any code in
> `src/cobrabox/visualization/space_explorer.py`, `_state.py`, or `_controls.py`.
> Update this file after every change.

---

## File Layout

| File | Role |
|---|---|
| `src/cobrabox/visualization/_state.py` | `ExplorerState` + constants. No Panel/HoloViews/Matplotlib imports. Only `param` + `numpy`. Independently testable. |
| `src/cobrabox/visualization/_controls.py` | `ControlsPanel` — sidebar widgets. Reads/writes `ExplorerState` params. |
| `src/cobrabox/visualization/space_explorer.py` | `DataPlot` + `SpaceExplorer` (top-level shell) + served entry-point block. |

---

## Class Hierarchy

```
ExplorerState (param.Parameterized)
DataPlot      (pn.viewable.Viewer)     — consumes ExplorerState
ControlsPanel (pn.viewable.Viewer)     — consumes ExplorerState
SpaceExplorer (pn.viewable.Viewer)     — composes all three
```

---

## `ExplorerState` (`_state.py`)

All reactive state lives here as `param` parameters. Neither Panel, HoloViews, nor Matplotlib are imported.

### Key Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `sweep_dim` | `param.Selector` | `"time"` or `dims[0]` | X-axis dimension |
| `trace_dim` | `param.Selector` | second dim | Y-axis / trace rows |
| `position` | `param.Integer` | `0` | Current index along `sweep_dim` (time_series only) |
| `window_size` | `param.Integer` | `200` | Number of samples in detail view (time_series only) |
| `viz_mode` | `param.Selector` | `"heatmap"` | Active visualization mode (see below) |
| `cmap` | `param.Selector` | `"RdBu_r"` | Heatmap colormap |
| `clim_min/max` | `param.Number` | global data range | Heatmap color limits |
| `extra_dim_indices` | `param.Dict` | `{dim: 0}` | Current slice index for each non-sweep/trace dim |
| `extra_dim_aggregate` | `param.Dict` | `{dim: False}` | `True` → aggregate entire dim instead of slicing |
| `scatter_trace_selection` | `param.List` | first 8 traces | String labels selected in scatter mode |
| `scatter_plot_type` | `param.Selector` | `"scatterplot"` | `"scatterplot"` or `"regplot"` |
| `scatter_marker` | `param.Selector` | `"o"` | Matplotlib marker |
| `scatter_marker_size` | `param.Integer` | `40` | Marker size |
| `box_plot_type` | `param.Selector` | `"boxplot"` | `"boxplot"`, `"violinplot"`, `"swarmplot"`, `"barplot"` |
| `hist_plot_type` | `param.Selector` | `"histplot"` | `"histplot"` or `"kdeplot"` |
| `hist_bins` | `param.Integer` | `20` | Histogram bin count |
| `hist_bin_width` | `param.Number` | `0.0` | Histogram bin width (0 = use `hist_bins`) |
| `joint_show_marginals` | `param.Boolean` | `True` | Show marginals in jointplot |
| `joint_kind` | `param.Selector` | `"scatter"` | `"scatter"`, `"hist"`, `"reg"`, `"kde"` |
| `custom_dims` | `param.List` | `[]` | Dims passed as axes to `custom_plot` |
| `sns_color` | `param.Selector` | `"steelblue"` | Color for seaborn plots |

### Private Attributes (not params)

| Attribute | Set in | Purpose |
|---|---|---|
| `_data` | `__init__` | Original cobrabox `Data` object |
| `_xr` | `__init__` | `xr.DataArray` shortcut (same as `data.data`) |
| `_data_min/max` | `__init__` | Global data range used by colormap slider |
| `_custom_plot` | Set by `SpaceExplorer.__init__` | User-supplied `custom_plot` callable |

### Constructor Logic

1. Extract dims from `data.data`, require ≥ 2.
2. Pick `sweep_dim` = `"time"` if present, else `dims[0]`; `trace_dim` = first remaining dim.
3. Set `param.Selector.objects` for both dims **before** `super().__init__`.
4. Set `position.bounds = (0, n-1)`, `window_size.bounds = (min(10,n), n)`.
5. Compute global `clim_min/max` from `nanmin/nanmax`.
6. Call `super().__init__(**params)`.
7. `_reset_scatter_trace_selection()` — selects ≤ 8 trace labels by default.

### Reactive Watchers in State

| Watcher method | Triggered by | Action |
|---|---|---|
| `_on_viz_mode_change` | `viz_mode` | When entering `boxplot`, auto-aggregates first non-sweep dim if none aggregated |
| `_update_position_bounds` | `sweep_dim` | Recalculates position/window bounds, clamps values, resets extra-dim indices and scatter selection |
| `_update_extra_dim_indices` | `trace_dim` | Resets `extra_dim_indices` and scatter trace selection |

### Data Access Methods

| Method | Returns | Notes |
|---|---|---|
| `get_window_slice()` | `(lo, hi)` ints | Centered window around `position` ± `window_size//2`, clamped to signal length |
| `windowed_2d(lo, hi)` | `xr.DataArray` 2-D | Slice of sweep_dim + trace_dim at window bounds |
| `full_2d()` | `xr.DataArray` 2-D | Full data: sweep_dim × trace_dim, extra dims sliced at `extra_dim_indices` |
| `get_2d()` | same as `full_2d()` | Alias |
| `get_scatter_df()` | `pd.DataFrame` | Long-form `[sweep_dim, trace_dim, value]`; only selected `scatter_trace_selection` traces |
| `get_boxplot_df()` | `pd.DataFrame` | Long-form `[sweep_dim, value]`; non-sweep dims sliced or aggregated |
| `get_hist_values()` | `np.ndarray` flat | All sweep values; non-sweep dims sliced or aggregated |
| `get_custom_data()` | `np.ndarray` | `custom_dims` kept as axes; others sliced or mean-aggregated |

### Helper Dim Lists

| Method | Returns |
|---|---|
| `_extra_dims()` | Dims that are neither `sweep_dim` nor `trace_dim` |
| `_hist_extra_dims()` | All dims except `sweep_dim` (used in histogram/boxplot) |
| `_boxplot_extra_dims` | Alias for `_hist_extra_dims` |
| `_custom_extra_dims()` | All dims NOT in `custom_dims` |

### Constants (also re-exported from `space_explorer.py`)

```python
_COLORMAPS        # diverging + perceptually-uniform colormaps
_VIZ_MODES        # ["time_series", "heatmap", "scatter", "boxplot", "histogram", "histogram_2d"]
_SCATTER_PLOT_TYPES  # ["scatterplot", "regplot"]
_MARKER_TYPES     # ["o", "s", "^", "D", "v", "P", "X", "*"]
_BOX_PLOT_TYPES   # ["boxplot", "violinplot", "swarmplot", "barplot"]
_HIST_PLOT_TYPES  # ["histplot", "kdeplot"]
_JOINT_KINDS      # ["scatter", "hist", "reg", "kde"]
_NAMED_COLORS     # 10 named CSS colors for seaborn plots
```

---

## `DataPlot` (`space_explorer.py`)

Owns and switches among three Panel panes depending on `viz_mode`.

### Internal Panes

| Pane | Used for | Backed by |
|---|---|---|
| `_hv_pane` (`pn.pane.HoloViews`) | `time_series`, `heatmap` | `hv.NdOverlay` / `hv.Image` |
| `_mpl_pane` (`pn.pane.Matplotlib`) | `scatter`, `boxplot`, `histogram`, `histogram_2d` | Matplotlib figures via seaborn |
| `_custom_pane` (`pn.Column`) | `custom` | User-provided function output |

Visibility is managed by `_update_pane_visibility()` which hides the wrong panes to avoid rendering overhead.

### Update Flow

All refreshes are triggered by `state.param.watch(...)`:

| Watch group | Parameters watched | Calls |
|---|---|---|
| HV group | `viz_mode`, `sweep_dim`, `trace_dim`, `position`, `window_size`, `cmap`, `clim_min/max`, `extra_dim_indices`, `extra_dim_aggregate` | `_refresh_hv()` |
| MPL group | `viz_mode`, `sweep_dim`, `trace_dim`, `extra_dim_indices`, `extra_dim_aggregate`, `scatter_*`, `sns_color`, `box_plot_type`, `hist_*`, `joint_*` | `_refresh_mpl()` |
| Custom group | `viz_mode`, `extra_dim_indices`, `extra_dim_aggregate`, `custom_dims` | `_refresh_custom()` |
| Visibility | `viz_mode` | `_update_pane_visibility()` |

`_refresh_mpl()` is a no-op when `viz_mode` is in `_HV_MODES` or `"custom"`.

### Plot Methods

#### `_plot_time_series()` → `hv.NdOverlay`
- Uses `state.get_window_slice()` → `(lo, hi)`, then `state.windowed_2d(lo, hi)`.
- Spacing = `1.2 × data_range` (computed over full data so scaling is stable while navigating).
- One `hv.Curve` per `trace_dim` value, vertically offset by `i * spacing`.
- Y-ticks show channel labels (subsampled to ≤ 20 when many traces).
- Options: `Category10` color cycle, no legend, responsive.

#### `_plot_heatmap()` → `hv.Image`
- Uses `state.full_2d()` — no windowing.
- Integer pixel-center bounds `(-0.5, n-0.5)` so categorical coords render correctly.
- Color limit taken from `state.clim_min/max`; colormap from `state.cmap`.
- Both axes subsampled to ≤ 20 ticks via `_subsample_ticks()`.

#### `_plot_scatter()` → `plt.Figure` (seaborn)
- Uses `state.get_scatter_df()` — long-form, filtered to `scatter_trace_selection`.
- Mode: `"scatterplot"` (one `sns.scatterplot`) or `"regplot"` (one `sns.regplot` per trace).
- Figure width: `min(16, max(7, n_cats * 0.6))`.

#### `_plot_boxplot()` → `plt.Figure` (seaborn)
- Hard limit: `sweep_dim` must have ≤ 30 indices; raises `ValueError` otherwise.
- Uses `state.get_boxplot_df()`.
- Dispatches to `sns.boxplot`, `sns.violinplot`, `sns.swarmplot`, or `sns.barplot`.

#### `_plot_histogram()` → `plt.Figure` (seaborn)
- Uses `state.get_hist_values()` — flat array.
- Mode: `"histplot"` (uses `hist_bins` or `hist_bin_width`) or `"kdeplot"`.
- `hist_bin_width > 0` overrides `hist_bins`.

#### `_plot_histogram_2d()` → `plt.Figure` (seaborn)
- Uses `_get_flat_df()` — long-form with all data (no extra-dim slicing via state — uses **direct** `full_2d`-like logic).
- `sns.jointplot` with configurable `joint_kind` and optional marginals.

#### `_build_custom_panel()` → `pn.viewable.Viewable`
- Calls `state._custom_plot(state.get_custom_data())`.
- Dispatch: `plt.Figure` → `pn.pane.Matplotlib` (+ `plt.close`), anything else → `pn.panel(...)`.
- Shows placeholder text if `_custom_plot` is `None` or `custom_dims` is empty.
- Catches all exceptions and shows error text in red.

### Static Helpers

| Method | Purpose |
|---|---|
| `_subsample_ticks(labels, max_ticks=20)` | Returns at most `max_ticks` `(index, label)` pairs, evenly spaced, always includes first/last |
| `_get_flat_df()` | Long-form DataFrame `[sweep_dim, trace_dim, value]` from full 2D data |
| `_make_context_str(self, fixed_dims)` | Annotation string like `"feature: line_length, patient: 0"` for plot titles |

### MPL Style

Applied via `matplotlib.rc_context(_MPL_STYLE)` inside all `_plot_*` methods. Does **not** affect notebook-wide rcParams. Key settings: `axes.grid=True`, top/right spines off, small label sizes, semibold title.

---

## `ControlsPanel` (`_controls.py`)

Builds all sidebar widgets in `__init__`. Uses `pn.config.set(sizing_mode="stretch_width")`.

### Widget Binding

Most widgets are bound directly with `pn.widgets.*.from_param(state.param.X)` — changes in the widget automatically update state and vice versa.

Exceptions that need manual watchers:
- `_clim_range_slider` → writes `state.clim_min, state.clim_max` (single `EditableRangeSlider` → two params).
- `_hist_binwidth_widget` → writes `state.hist_bin_width`; when set > 0, resets `_hist_bins_widget` to its minimum.
- `_hist_bins_widget` → when moved (and `bin_width != 0`), resets `state.hist_bin_width = 0`.
- `_custom_dim_selector` → writes `state.custom_dims` and triggers `_rebuild_extra_dim_controls()`.

### Dynamic Sections

| Method | Triggered by | What it rebuilds |
|---|---|---|
| `_rebuild_extra_dim_controls()` | `sweep_dim`, `trace_dim`, `viz_mode`, `custom_dims` changes | One nav row (◀ label ▶ + optional aggregate checkbox) per extra dim |
| `_rebuild_scatter_trace_controls()` | `sweep_dim`, `trace_dim`, `viz_mode` changes | `CheckBoxGroup` (≤ 8 traces) or `MultiChoice` (> 8) for scatter trace selection |
| `_update_panel_layout()` | `viz_mode` changes | Entire layout — which sections are shown |

### Extra-Dim Nav Row

Each navigable dimension gets: `◀ button | "dim: N / total" label | ▶ button` + optional aggregate `Checkbox`.
- `◀`/`▶` write `state.extra_dim_indices = {**current, dim: new_val}` (immutable dict replacement to trigger param reactivity).
- Aggregate checkbox writes `state.extra_dim_aggregate = {**current, dim: bool}` and disables the nav buttons.
- Callbacks created via closure factory `_make_callbacks(d, label, size, p_btn, n_btn)` to avoid late-binding bugs.

### Layout per Mode

| Mode | Sections shown |
|---|---|
| `time_series` | Visualization, Dimensions (sweep + trace + swap), Extra nav, Navigation (position + window sliders), Metadata |
| `heatmap` | Visualization, Dimensions (sweep + trace + swap), Extra nav, Colormap (cmap + range slider), Metadata |
| `scatter` | Visualization, Dimensions (sweep + trace + swap), Extra nav, Traces (trace selection), Scatter options (type + marker + size), Color, Metadata |
| `boxplot` | Visualization, Dimensions (sweep only), Navigate dims (all non-sweep), Boxplot options (type), Color, Metadata |
| `histogram` | Visualization, Dimensions (sweep only), Navigate dims (all non-sweep), Histogram options (type + bins + bin width), Color, Metadata |
| `histogram_2d` | Visualization, Dimensions (sweep + trace + swap), Joint options (kind + marginals), Color, Metadata |
| `custom` | Visualization, Custom plot dimensions (MultiChoice), Navigate remaining dims, Metadata |

### Metadata Pane

`_metadata_text` is a `@param.depends("_state.sweep_dim", "_state.trace_dim")` property that generates Markdown showing: Subject, Sampling rate, Shape, Dims, History (if any).

---

## `SpaceExplorer` (`space_explorer.py`)

Top-level shell. Composes the three components.

### Constructor

```python
SpaceExplorer(data, *, window_size=200, position=0, custom_plot=None, dims=None, **params)
```

1. Creates `ExplorerState(data, position=position, window_size=window_size)`.
2. If `custom_plot` is provided:
   - Attaches it as `state._custom_plot = custom_plot`.
   - Appends `"custom"` to `state.param["viz_mode"].objects` if not already there.
   - Sets `state.custom_dims = list(dims)` if `dims` is given.
   - Sets `state.viz_mode = "custom"`.
3. Creates `ControlsPanel(state)` and `DataPlot(state)`.
4. Layout: `pn.Row(pn.Column(_controls, width=300), pn.Column(_plot, stretch_both), stretch_both)`.

### Layout

```
pn.Row (stretch_both)
├── pn.Column (width=300)
│   └── ControlsPanel
└── pn.Column (stretch_both)
    └── DataPlot
```

### `create_app()` classmethod

Returns a `pn.template.FastListTemplate`:
- `sidebar=[instance._controls]`
- `main=[instance._plot]`
- `main_layout=None`

### Served Entry-Point

```python
if pn.state.served:
    from cobrabox.dataset_loader import load_noise_dummy
    _demo = load_noise_dummy()[0]
    SpaceExplorer.create_app(_demo).servable()
```

Run with: `panel serve src/cobrabox/visualization/space_explorer.py --show --dev`

---

## Data Flow Summary

```
User widget interaction
        │
        ▼
ControlsPanel writes to ExplorerState param
        │
        ▼
ExplorerState param.watch fires (if any internal logic needed)
        │
        ▼
DataPlot param.watch fires → _refresh_hv / _refresh_mpl / _refresh_custom
        │
        ▼
Plot method calls ExplorerState data-access helpers (get_window_slice, full_2d, etc.)
        │
        ▼
New hv.Element or plt.Figure assigned to the active pane
```

---

## Key Design Decisions

- **`_state.py` has no UI imports** — testable with just `param + numpy`.
- **`param.watch` (not `@param.depends`)** — explicit callbacks avoid non-param attribute path resolution issues in cross-object reactivity.
- **Three panes always exist in `DataPlot`** — visibility toggled, not created/destroyed, to avoid re-render cost.
- **`plt.close(fig)` always called** after assigning a figure to `pn.pane.Matplotlib` — prevents memory leaks and Jupyter auto-display.
- **`matplotlib.rc_context(_MPL_STYLE)`** — style applied locally, not globally.
- **Dict replacement for `extra_dim_indices`** (`{**old, dim: val}` not `old[dim] = val`) — immutable update required to trigger param reactivity.
- **`hv.Image` with integer bounds** — categorical x/y coords require integer pixel-center bounds; actual labels are carried via explicit `xticks`/`yticks`.

---

## Tests

Visualization tests are in the workspace root, not `tests/`:

```bash
uv run pytest test_visualization*.py test_space_explorer*.py -q
```

Do **not** run the full `uv run pytest` when iterating on visualization changes.
