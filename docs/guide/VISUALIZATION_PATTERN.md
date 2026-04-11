# Visualization Pattern Guide

**Last Updated:** 2026-03-25

This guide documents best practices for creating new visualization files in the `src/cobrabox/visualization/` module. All visualization classes should follow the patterns established in `seizure_explorer.py` to ensure consistency, maintainability, and seamless integration with both Jupyter notebooks and Panel-served apps.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Principles](#core-principles)
3. [File Structure](#file-structure)
4. [Data Input](#data-input)
5. [Reactive State Management](#reactive-state-management)
6. [Component Hierarchy](#component-hierarchy)
7. [Jupyter Notebook Integration](#jupyter-notebook-integration)
8. [Panel Served App Integration](#panel-served-app-integration)
9. [Interactive Controls](#interactive-controls)
10. [Common Patterns](#common-patterns)
11. [Implementation Checklist](#implementation-checklist)

---

## Architecture Overview

Visualization classes are built as **layered hierarchies** composed of:

1. **State Container** — holds all reactive parameters (using `param.Parameterized`)
2. **Specialized Plot Components** — each renders a part of the visualization (inheriting from `pn.viewable.Viewer`)
3. **Top-Level Viewer** — composes all components into a unified interactive experience
4. **Entry Points** — `__panel__()` for Jupyter and `create_app()` for Panel serve

This pattern enables:
- **Reactive updates** — parameter changes automatically cascade through the UI
- **Dual deployment** — same code works both inline (Jupyter) and served (Panel)
- **Dimension flexibility** — state determines which dimensions are visualized
- **Event handling** — Bokeh-level tap handlers enable click-based interaction

---

## Core Principles

### 1. Accept cobrabox `Data` objects

All visualizations **must** accept a cobrabox `Data` object (or `SignalData`, `EEG`, `FMRI`) as the primary input. Always validate that the data has sufficient dimensions for the visualization type.

```python
from cobrabox.data import Data

def __init__(self, data: Data, **params):
    if not isinstance(data, Data):
        raise TypeError(f"Expected Data, got {type(data)}")
    if len(data.data.dims) < 2:
        raise ValueError("Data must have at least 2 dimensions")
```

### 2. Use `param.Parameterized` for reactive state

All interactive parameters should live in a state class inheriting from `param.Parameterized`. This enables:
- Two-way binding with Panel widgets
- Automatic validation of parameter values
- Reactive cascading of updates (@param.depends decorators)

```python
import param

class VisualizationState(param.Parameterized):
    dimension_x = param.Selector(default="time", objects=[], doc="X-axis dimension")
    dimension_y = param.Selector(default=None, objects=[], doc="Y-axis dimension")
    position = param.Integer(default=0, bounds=(0, 1), doc="Current position")
```

### 3. Keep state **independent of rendering**

State should never contain Plot objects, figures, or any rendering artifacts. It is purely **data and parameters**. This keeps state serializable and simplifies testing.

```python
# ✓ GOOD: pure state
self._state = VisualizationState(data=data, dimension_x="time", ...)

# ✗ BAD: state mixed with rendering
self._state.fig = plt.Figure()  # Don't do this
self._state.axes = fig.add_subplot()  # Don't do this
```

### 4. Preserve input data immutability

Never mutate the input `Data` object. All state changes should be parameterized selections or transformations, not in-place edits to the underlying xarray.

```python
# ✓ GOOD: select via state
window = self._state.xr.isel(**{self._state.sweep_dim: slice(lo, hi)})

# ✗ BAD
self._state.data.data[:] *= 2  # mutates input
```

### 5. Use neuroscience visualization standards

When visualizing neurophysiological data:
- **Time** is typically the sweep dimension (X-axis or fastest-varying)
- **Space** (channels, electrodes, voxels) are traces, rows, or heatmap rows
- **Amplitude** uses perceptually uniform colormaps (viridis, RdBu) by default
- **Frequency domains** use log scaling when appropriate
- Include units (µV, mV, Hz, ms) in axis labels and metadata displays

---

## File Structure

A complete visualization file follows this template:

```python
"""Brief description of the visualization.

Classes:
    StateClass: Docstring describing reactive parameters.
    PlotComponent1: Docstring describing a plot subcomponent.
    PlotComponent2: Another plot subcomponent.
    TopLevelViewer: Main class; accepts Data and composes components.

Example — notebook::

    import cobrabox as cb
    from cobrabox.visualization import TopLevelViewer

    data = cb.from_numpy(arr, dims=["time", "space"])
    TopLevelViewer(data)

Example — served::

    panel serve src/cobrabox/visualization/my_viz.py --show --dev
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

# ───────────────────────────────────────────────────────
# Optional helper classes (utilities, state managers)
# ───────────────────────────────────────────────────────

class _HelperClass:
    """Private helper for cross-component communication (if needed)."""
    pass

# ───────────────────────────────────────────────────────
# 1. State container
# ───────────────────────────────────────────────────────

class VisualizationState(param.Parameterized):
    """Shared reactive state for all viewer components."""
    # Selectors
    # Integers (sliders, positions)
    # Booleans (toggles)
    # Other parameters
    
    def __init__(self, data: Data, **params):
        # Validate and pre-compute bounds
        self._data = data
        self._xr = data.data
        super().__init__(**params)
    
    @property
    def data(self) -> Data:
        return self._data

# ───────────────────────────────────────────────────────
# 2. Plot components (each extends pn.viewable.Viewer)
# ───────────────────────────────────────────────────────

class PlotComponent1(pn.viewable.Viewer):
    """First plot subcomponent."""
    def __init__(self, state: VisualizationState, **params):
        super().__init__(**params)
        self._state = state
        self._pane = pn.pane.HoloViews(self._plot)
    
    @param.depends("_state.param1", "_state.param2")
    def _plot(self) -> hv.Element:
        # Compute and return a HoloViews element (Curve, Image, etc.)
        pass
    
    def __panel__(self):
        return self._pane

class PlotComponent2(pn.viewable.Viewer):
    """Second plot subcomponent."""
    # Similar structure to PlotComponent1
    pass

# ───────────────────────────────────────────────────────
# 3. Controls panel
# ───────────────────────────────────────────────────────

class ControlsPanel(pn.viewable.Viewer):
    """Interactive controls and metadata display."""
    def __init__(self, state: VisualizationState, **params):
        super().__init__(**params)
        self._state = state
        # Create widgets from state parameters
        # Create metadata display using @param.depends
        self._panel = pn.Column(...)
    
    def __panel__(self):
        return self._panel

# ───────────────────────────────────────────────────────
# 4. Top-level viewer
# ───────────────────────────────────────────────────────

class TopLevelViewer(pn.viewable.Viewer):
    """Interactive visualization for cobrabox Data objects.
    
    Composes state, plot components, and controls into a unified viewer
    that works in Jupyter notebooks and Panel-served apps.
    
    Parameters
    ----------
    data : Data
        Cobrabox Data object with dimensions suitable for this visualization.
    **params : dict
        Optional parameters to initialize state (dimension selections, etc.).
    """
    
    def __init__(self, data: Data, **params):
        super().__init__(**params)
        
        # Validate input
        if not isinstance(data, Data):
            raise TypeError(...)
        if len(data.data.dims) < 2:
            raise ValueError(...)
        
        # Build state
        self._state = VisualizationState(data, **params)
        
        # Build components
        with pn.config.set(sizing_mode="stretch_width"):
            self._controls = ControlsPanel(self._state)
            self._plot1 = PlotComponent1(self._state)
            self._plot2 = PlotComponent2(self._state)
            
            # Compose into layout
            self._layout = pn.Row(
                pn.Column(self._controls, width=300),
                pn.Column(self._plot1, self._plot2, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            )
    
    def __panel__(self):
        """Return Panel layout for Jupyter notebooks."""
        return self._layout
    
    @classmethod
    def create_app(cls, data: Data, **params) -> pn.template.FastListTemplate:
        """Build a served Panel app with template.
        
        Returns a FastListTemplate ready to call .servable().
        """
        instance = cls(data, **params)
        return pn.template.FastListTemplate(
            title="Visualization Title",
            sidebar=[instance._controls],
            main=[pn.Column(instance._plot1, instance._plot2)],
            main_layout=None,
        )

# ───────────────────────────────────────────────────────
# 5. Served entry point (optional)
# ───────────────────────────────────────────────────────

if pn.state.served:
    from cobrabox.dataset_loader import load_dummy_dataset
    
    _demo = load_dummy_dataset()[0]
    TopLevelViewer.create_app(_demo).servable()
```

---

## Data Input

### Accepting cobrabox Data

All visualization entry points must accept a `Data` object:

```python
from cobrabox.data import Data, SignalData

class MyVisualizer(pn.viewable.Viewer):
    def __init__(self, data: Data | SignalData, **params):
        # Validate
        if not isinstance(data, (Data, SignalData)):
            raise TypeError(f"Expected Data or SignalData, got {type(data).__name__}")
        
        # Check for required dimensions
        dims = list(data.data.dims)
        if "time" in dims and len(dims) < 2:
            raise ValueError("Time-series visualizations need at least one spatial dimension")
        
        self._state = YourState(data, ...)
```

### Accessing data properties

Use the properties and methods of the `Data` object to inform visualization:

```python
# From cobrabox Data object stored in VisualizationState
metadata = self._state.data
sampling_rate = metadata.sampling_rate  # May be None for non-time-series
subject_id = metadata.subjectID
group_id = metadata.groupID
condition = metadata.condition
history = metadata.history  # List of applied feature/pipeline names

# From underlying xarray.DataArray
xr_data = self._state.xr
dims = list(xr_data.dims)  # dimension names
sizes = dict(xr_data.sizes)  # {dim: size}
coords = xr_data.coords  # coordinate values
```

---

## Reactive State Management

### Using `param.Parameterized`

State is the **single source of truth** for all UI state. Use param types that match your data:

```python
import param

class MyState(param.Parameterized):
    # Selector for discrete choices
    sweep_dimension = param.Selector(
        default="time",
        objects=[],  # populated in __init__
        doc="Dimension mapped to X-axis"
    )
    
    # Integer for positions and counts
    position = param.Integer(
        default=0,
        bounds=(0, 1),  # dynamically updated in __init__
        doc="Current position in signal"
    )
    
    # Numbers for continuous values
    threshold = param.Number(
        default=0.5,
        bounds=(0.0, 1.0),
        doc="Threshold for signal detection"
    )
    
    # Boolean for toggles
    show_events = param.Boolean(
        default=True,
        doc="Show event markers on plots"
    )
    
    # Arrays for multi-value selectors or event indices
    event_indices = param.Array(
        default=None,
        doc="1-D array of detected event positions"
    )
    
    def __init__(self, data: Data, **params):
        # Pre-populate objects BEFORE super().__init__
        xr_data = data.data
        dims = list(xr_data.dims)
        self.param.sweep_dimension.objects = dims
        
        # Set bounds based on data shape
        n = xr_data.sizes[self.param.sweep_dimension.default]
        self.param.position.bounds = (0, n - 1)
        
        super().__init__(**params)
        
        # Store data as a non-param attribute (immutable reference)
        self._data = data
        self._xr = xr_data
```

**Why pre-populate before `super().__init__`?**

Param validation occurs during `super().__init__()`. If a Selector doesn't have `objects` yet, param validation fails when trying to set an initial value. Always populate `Selector.objects` before calling `super().__init__`.

### Reactive dependencies with `@param.depends`

Use `@param.depends` to mark plot methods that must re-render when parameters change:

```python
@param.depends("_state.sweep_dim", "_state.position", "_state.zoom_mode")
def _plot(self) -> hv.Element:
    """Re-renders whenever sweep_dim, position, or zoom_mode changes."""
    s = self._state
    # Compute plot using s.sweep_dim, s.position, s.zoom_mode
    return hv.Curve(...)
```

### Watching parameters for side effects

Use `@param.depends(watch=True)` or `param.watch()` for callbacks that update other state:

```python
@param.depends("sweep_dim", watch=True)
def _update_position_bounds(self) -> None:
    """When sweep_dim changes, update position bounds."""
    n = self._xr.sizes[self.sweep_dim]
    self.param.position.bounds = (0, n - 1)
    # Clamp position to new bounds
    self.position = min(self.position, n - 1)
```

Or use explicit watching:

```python
def __init__(self, ...):
    state.param.watch(self._on_zoom_toggle, ["zoom_window"])

def _on_zoom_toggle(self, event):
    # Handle change
    self._pane.linked_axes = not state.zoom_window
```

---

## Component Hierarchy

### 1. State Container

Holds all reactive parameters. Immutable after construction (except parameter updates).

```python
class MyState(param.Parameterized):
    """Shared reactive state for all components."""
    
    def __init__(self, data: Data, **params):
        # Validate data
        # Set up param bounds and objects
        super().__init__(**params)
        # Store immutable references
        self._data = data
        self._xr = data.data
    
    @property
    def data(self) -> Data:
        return self._data
    
    # Helper methods for common computations
    def get_window_bounds(self) -> tuple[int, int]:
        """Return (lo, hi) for the current detail window."""
        pass
```

### 2. Plot Components

Each plot extends `pn.viewable.Viewer` and renders one visualization element.

```python
class MyPlot(pn.viewable.Viewer):
    """Single plot in the visualization hierarchy."""
    
    def __init__(self, state: MyState, **params):
        super().__init__(**params)
        self._state = state
        
        # Initialize Panel pane to hold rendered output
        self._pane = pn.pane.HoloViews(
            self._plot,  # reactive-dependency-decorated method
            sizing_mode="stretch_width",
            min_height=300,
        )
    
    @param.depends("_state.param1", "_state.param2", ...)
    def _plot(self) -> hv.Element:
        """Render plot; re-called whenever depended-on params change."""
        s = self._state
        # Always create fresh HoloViews element; never reuse
        return hv.Curve((x, y), ...).opts(...)
    
    def __panel__(self):
        """Return the Panel pane for display."""
        return self._pane
```

**Key detail:** The `@param.depends` decorator on `_plot()` tells Panel to re-call the method whenever those parameters change, automatically re-rendering the pane.

### 3. Controls Panel

Builds interactive widgets from state parameters, plus metadata/info display.

```python
class ControlsPanel(pn.viewable.Viewer):
    """Interactive controls and metadata."""
    
    def __init__(self, state: MyState, **params):
        super().__init__(**params)
        self._state = state
        
        with pn.config.set(sizing_mode="stretch_width"):
            # Create widgets from state parameters
            self._dim_select = pn.widgets.Select.from_param(
                state.param.sweep_dimension,
                name="Sweep dimension"
            )
            self._pos_slider = pn.widgets.IntSlider.from_param(
                state.param.position,
                name="Position"
            )
            
            # Info displays with reactive content
            self._metadata = pn.pane.Markdown(
                self._metadata_text,
                sizing_mode="stretch_width"
            )
            
            # Assemble layout
            self._panel = pn.Column(
                pn.pane.Markdown("### Controls", disable_anchors=True),
                self._dim_select,
                self._pos_slider,
                pn.layout.Divider(),
                pn.pane.Markdown("### Metadata", disable_anchors=True),
                self._metadata,
            )
    
    @param.depends("_state.sweep_dimension", "_state.position")
    def _metadata_text(self) -> str:
        """Generate markdown for current state."""
        d = self._state.data
        return f"""
        **Subject:** {d.subjectID or '—'}
        
        **Sampling rate:** {d.sampling_rate or '—'} Hz
        
        **Dims:** {list(self._state.xr.dims)}
        """
    
    def __panel__(self):
        return self._panel
```

### 4. Top-Level Viewer

Composes all components and provides entry points for both Jupyter and served apps.

```python
class MyVisualizer(pn.viewable.Viewer):
    """Top-level interactive visualization."""
    
    def __init__(self, data: Data, **params):
        super().__init__(**params)
        
        # Build state
        self._state = MyState(data, **params)
        
        # Build all components
        with pn.config.set(sizing_mode="stretch_width"):
            self._controls = ControlsPanel(self._state)
            self._plot1 = MyPlot(self._state)
            self._plot2 = OtherPlot(self._state)
            
            # Assemble layout
            self._layout = pn.Row(
                pn.Column(self._controls, width=300),
                pn.Column(self._plot1, self._plot2, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            )
    
    def __panel__(self):
        """Entry point for Jupyter: return Panel layout."""
        return self._layout
    
    @classmethod
    def create_app(cls, data: Data, **params) -> pn.template.FastListTemplate:
        """Entry point for 'panel serve': return templated app."""
        instance = cls(data, **params)
        return pn.template.FastListTemplate(
            title="My Visualization",
            sidebar=[instance._controls],
            main=[pn.Column(instance._plot1, instance._plot2)],
            main_layout=None,
        )
```

---

## Jupyter Notebook Integration

### Using `__panel__`

Any class that implements `__panel__()` can be displayed in a Jupyter cell:

```python
from cobrabox.visualization import MyVisualizer

data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
viz = MyVisualizer(data)
viz  # Automatically calls __panel__() and displays
```

Under the hood, Jupyter's IPython display system calls `__panel__()` and renders the returned Panel object.

### Requirements

- The class must inherit from `pn.viewable.Viewer` (or any class with a `__panel__()` method)
- All subcomponents must also have `__panel__()` methods
- Use `pn.pane.HoloViews()` to wrap HoloViews elements (which automatically handles rendering)
- Panel handles communication with the Jupyter kernel via WebSocket/IPC

### Example notebook usage

```python
import cobrabox as cb
from cobrabox.visualization import MyVisualizer

# Load data
dataset = cb.dataset("dummy_noise")
data = dataset[0]

# Create visualization — displays automatically
viz = MyVisualizer(data, position=100, sweep_dimension="time")

# Interact: move slider, toggle options, click plots — all reactive
# Changes sync immediately via Jupyter kernel
```

---

## Panel Served App Integration

### Using `create_app` and `panel serve`

For larger-screen, standalone app deployments, use Panel's templating system:

```python
@classmethod
def create_app(cls, data: Data, **params) -> pn.template.FastListTemplate:
    """Build a served Panel app with template.
    
    Returns a FastListTemplate ready to call .servable().
    """
    instance = cls(data, **params)
    return pn.template.FastListTemplate(
        title="My Visualization",
        sidebar=[instance._controls],
        main=[pn.Column(instance._plot1, instance._plot2)],
        main_layout=None,
    )
```

### File entry point

At the end of your visualization file, include:

```python
if pn.state.served:
    from cobrabox.dataset_loader import load_noise_dummy
    
    # Load demo data
    demo_data = load_noise_dummy()[4]
    
    # Create and serve the app
    MyVisualizer.create_app(demo_data).servable()
```

The `if pn.state.served` guard ensures the entry point only runs when served via `panel serve`, not when the module is imported normally.

### Servicing the app

```bash
# From the project root
panel serve src/cobrabox/visualization/my_visualization.py --show --dev

# Open browser at http://localhost:5006/my_visualization
```

The `--dev` flag enables hot-reloading: edits to the .py file automatically restart the app.

### Why `FastListTemplate`?

- **Responsive:** Adapts to window resizing
- **Sidebar:** Perfect for narrow control panels (width=300)
- **Main area:** Stretches to fill remaining space
- **Clean:** Minimal chrome, focus on content
- **Mobile-friendly:** Responsive design works on tablets/phones

Other templates: `MaterialTemplate`, `BootstrapTemplate`, `VanillaTemplate` — choose based on aesthetic preference.

---

## Interactive Controls

### Click-based interaction with Bokeh

For Bokeh-based plots (HoloViews with `hv.extension("bokeh")`), you can register event handlers at the Bokeh figure level:

```python
from bokeh.events import Tap as BokehTap

class MyPlot(pn.viewable.Viewer):
    def __init__(self, state: MyState, ...):
        ...
        self._pane = pn.pane.HoloViews(self._plot)
    
    def _make_hook(self, state: MyState):
        """Bokeh post-render hook to attach click handlers."""
        def hook(plot_obj: object, element: object) -> None:
            fig = plot_obj.handles["plot"]  # Get Bokeh figure
            
            # Register one-time on_event handler
            if not getattr(fig, "_tap_registered", False):
                fig._tap_registered = True
                
                def on_tap(event) -> None:
                    # event.x, event.y contain click coordinates
                    # Update state (which triggers reactive re-renders)
                    new_index = compute_index_from_coordinates(event.x, state)
                    
                    # For Jupyter: use add_next_tick_callback for thread safety
                    doc = pn.state.curdoc
                    if doc and doc.session_context:
                        doc.add_next_tick_callback(
                            lambda: setattr(state, "position", new_index)
                        )
                    else:
                        state.position = new_index  # Direct for served apps
                
                fig.on_event(BokehTap, on_tap)
        
        return hook
    
    @param.depends("_state.position", "_state.sweep_dim")
    def _plot(self) -> hv.Element:
        return hv.Curve(...).opts(
            tools=["tap"],
            hooks=[self._make_hook(self._state)],
        )
```

**Key points:**

- Use `plot_obj.handles["plot"]` to access the underlying Bokeh figure from HoloViews
- Register handlers **once per figure** (check `_tap_registered` flag)
- For Jupyter, wrap state updates in `doc.add_next_tick_callback()` for thread safety
- For served apps, direct state updates work fine
- Updating state triggers reactive re-renders of all dependent plots

### Widget callbacks

Panel widgets automatically sync with param state, but you can also attach explicit callbacks:

```python
self._range_slider = pn.widgets.EditableRangeSlider(
    name="Color range",
    value=(state.clim_min, state.clim_max)
)

def _on_range_change(event):
    state.clim_min, state.clim_max = event.new

self._range_slider.param.watch(_on_range_change, ["value"])
```

---

## Common Patterns

### 1. Synchronizing zoom between plots

Use a shared "range linker" to keep horizontal axis ranges in sync:

```python
class _RangeLinker:
    """Shares a Bokeh x_range between two plots."""
    def __init__(self):
        self._x_range = None
    
    def reset(self):
        self._x_range = None
    
    def hook(self, plot_obj, element) -> None:
        fig = plot_obj.handles["plot"]
        if self._x_range is None:
            self._x_range = fig.x_range
        else:
            fig.x_range = self._x_range

# Then pass to both plots:
linker = _RangeLinker()
plot1 = MyPlot(state, linker=linker)
plot2 = OtherPlot(state, linker=linker)

# When sweep_dim changes, reset:
state.param.watch(lambda _e: linker.reset(), ["sweep_dim"])
```

### 2. Windowing and slicing

Always use xarray's `.isel()` for integer-based indexing along dimensions:

```python
def get_window_slice(self, lo: int, hi: int):
    """Return 2-D slice (trace_dim × sweep_dim) for a window."""
    window = self._xr.isel(**{self.sweep_dim: slice(lo, hi)})
    extra_dims = [d for d in self._xr.dims if d not in (self.sweep_dim,)]
    if extra_dims:
        window = window.mean(dim=extra_dims)
    return window
```

### 3. Computing axis coordinates

Always extract coordinates from xarray for plot axes:

```python
def sweep_coords(self) -> np.ndarray:
    """Get coordinate values for the current sweep dimension."""
    return self._xr.coords[self.sweep_dim].values

# Use in plotting:
coords = self._state.sweep_coords()
curve = hv.Curve((coords, values), kdims=[self._state.sweep_dim], ...)
```

This preserves physical units (e.g., time in seconds, frequency in Hz) from the xarray coordinates.

### 4. Handling extra dimensions

Data often has extra dimensions beyond the two being visualized. Average them out:

```python
def windowed_2d(self, lo: int, hi: int):
    """Return 2-D array by reducing extra dimensions."""
    window = self._xr.isel(**{self.sweep_dim: slice(lo, hi)})
    extra = [d for d in self._xr.dims if d not in (self.sweep_dim, self.trace_dim)]
    if extra:
        window = window.mean(dim=extra)
    return window
```

### 5. Colormap and clim controls

Provide user-adjustable colormaps and limits for scientific accuracy:

```python
class MyState(param.Parameterized):
    cmap = param.Selector(
        default="viridis",
        objects=["viridis", "RdBu_r", "plasma", ...],
        doc="Heatmap colormap"
    )
    clim_min = param.Number(default=0.0, doc="Colormap lower bound")
    clim_max = param.Number(default=1.0, doc="Colormap upper bound")

# In controls:
cmap_widget = pn.widgets.Select.from_param(state.param.cmap)
clim_slider = pn.widgets.EditableRangeSlider(
    name="Color limit",
    value=(state.clim_min, state.clim_max),
    ...
)

# In plot:
hv.Image(...).opts(
    cmap=state.cmap,
    clim=(state.clim_min, state.clim_max),
    colorbar=True,
)
```

### 6. Marking events/annotations

Show detected events on the signal:

```python
class MyState(param.Parameterized):
    event_indices = param.Array(default=None, doc="Event positions")

def _plot(self) -> hv.Overlay:
    s = self._state
    coords = s.sweep_coords()
    
    # Main curve
    curve = hv.Curve(...)
    
    # Event scatter if provided
    overlay = curve
    if s.event_indices is not None and len(s.event_indices) > 0:
        valid = s.event_indices[s.event_indices < len(coords)]
        if len(valid) > 0:
            scatter = hv.Scatter(
                (coords[valid], signal[valid]),
                kdims=[...], vdims=[...]
            ).opts(color="orange", size=8)
            overlay = overlay * scatter
    
    return overlay
```

---

## Implementation Checklist

When creating a new visualization, verify:

- [ ] **File located in** `src/cobrabox/visualization/`
- [ ] **Docstring** includes:
  - Brief description of the visualization type
  - `Classes:` section listing state + components + top-level viewer
  - `Example — notebook::` showing Jupyter usage
  - `Example — served::` showing panel serve command
- [ ] **Accepts cobrabox `Data`** as primary input
- [ ] **State class** inherits from `param.Parameterized`
  - Selectors pre-populated before `super().__init__`
  - Integer bounds set based on data shape before `super().__init__`
  - Non-param attributes `_data` and `_xr` stored immutably
- [ ] **Plot components** inherit from `pn.viewable.Viewer`
  - Each has `_pane` attribute (initialized in `__init__`)
  - Each has `_plot()` method decorated with `@param.depends`
  - Each implements `__panel__()` returning the pane
- [ ] **Controls panel** includes:
  - Widgets created via `.from_param()` for state synchronization
  - Metadata/info display (reactive via `@param.depends`)
- [ ] **Top-level viewer**:
  - Composes all components into a layout
  - Implements `__panel__()` for Jupyter
  - Implements `create_app()` class method for served apps
- [ ] **Served entry point** (at end of file):
  ```python
  if pn.state.served:
      from cobrabox.dataset_loader import load_example
      
      demo = load_example()
      TopLevelViewer.create_app(demo).servable()
  ```
- [ ] **Data validation** at entry point:
  - Check `isinstance(data, Data)`
  - Check minimum dimension count
  - Check for required dimensions (e.g., "time")
- [ ] **Neuroscience standards** applied:
  - Unit labels in axes (µV, Hz, ms, etc.)
  - Perceptually uniform colormaps by default
  - Proper coordinate systems and interpretations
- [ ] **No mutations** of input `Data` object
- [ ] **Handles edge cases**:
  - Empty dimensions
  - Single-sample windows
  - No events provided
  - NaN values in data
- [ ] **Tested**:
  - Works in Jupyter notebook
  - Works with `panel serve`
  - Responsive to all interactive controls
  - Handles user parameter changes gracefully

---

## Example: Minimal Visualization

Here's a complete, minimal visualization implementing all patterns:

```python
"""Minimal demonstration visualization."""

from __future__ import annotations

import numpy as np
import param

import holoviews as hv
import panel as pn

from cobrabox.data import Data

hv.extension("bokeh")


class MinimalState(param.Parameterized):
    """Minimal reactive state."""

    sweep_dim = param.Selector(default="time", objects=[], doc="X-axis dimension")
    trace_dim = param.Selector(default=None, objects=[], doc="Trace dimension")
    position = param.Integer(default=0, bounds=(0, 1), doc="Current position")

    def __init__(self, data: Data, **params):
        xr_data = data.data
        dims = list(xr_data.dims)
        if len(dims) < 2:
            raise ValueError("Need ≥2 dimensions")

        default_sweep = "time" if "time" in dims else dims[0]
        remaining = [d for d in dims if d != default_sweep]

        params.setdefault("sweep_dim", default_sweep)
        params.setdefault("trace_dim", remaining[0])

        self.param.sweep_dim.objects = dims
        self.param.trace_dim.objects = dims

        sweep = params.get("sweep_dim", default_sweep)
        n = xr_data.sizes[sweep]
        self.param.position.bounds = (0, n - 1)

        super().__init__(**params)

        self._data = data
        self._xr = xr_data

    @property
    def data(self) -> Data:
        return self._data


class MinimalPlot(pn.viewable.Viewer):
    """Minimal plot component."""

    def __init__(self, state: MinimalState, **params):
        super().__init__(**params)
        self._state = state
        self._pane = pn.pane.HoloViews(self._plot)

    @param.depends("_state.sweep_dim", "_state.trace_dim", "_state.position")
    def _plot(self) -> hv.Curve:
        s = self._state
        dims = list(s._xr.dims)
        # Average over all dims except sweep_dim
        reduced = s._xr.mean(dim=[d for d in dims if d != s.sweep_dim])
        coords = s._xr.coords[s.sweep_dim].values
        return hv.Curve((coords, reduced.values), kdims=[s.sweep_dim]).opts(
            responsive=True, title="Mean signal"
        )

    def __panel__(self):
        return self._pane


class MinimalVisualizer(pn.viewable.Viewer):
    """Minimal interactive visualization."""

    def __init__(self, data: Data, **params):
        super().__init__(**params)

        if not isinstance(data, Data):
            raise TypeError()
        if len(data.data.dims) < 2:
            raise ValueError()

        self._state = MinimalState(data, **params)

        with pn.config.set(sizing_mode="stretch_width"):
            self._plot = MinimalPlot(self._state)
            self._layout = pn.Column(self._plot, sizing_mode="stretch_both")

    def __panel__(self):
        return self._layout

    @classmethod
    def create_app(cls, data: Data, **params):
        instance = cls(data, **params)
        return pn.template.FastListTemplate(
            title="Minimal Visualization",
            main=[instance._layout],
            main_layout=None,
        )


if pn.state.served:
    from cobrabox.dataset_loader import load_noise_dummy

    demo = load_noise_dummy()[0]
    MinimalVisualizer.create_app(demo).servable()
```

---

## References

- **Panel documentation:** https://panel.holoviz.org/
- **HoloViews tutorial:** https://holoviews.org/
- **Param documentation:** https://param.holoviz.org/
- **Bokeh events:** https://docs.bokeh.org/en/latest/docs/user_guide/server.html
- **cobrabox Data API:** See `src/cobrabox/data.py`

