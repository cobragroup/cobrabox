"""Visualization module for cobrabox time-series data exploration.

Provides interactive plotting tools for exploring EEG/fMRI time-series data
with detailed time-window analysis and navigation.

Classes:
    InteractiveExplorer: Interactive visualization of time-series data.
    TimeSeriesPlotter: Time series visualization from Data objects.
    HeatmapPlotter: Heatmap visualization from Data objects.
    HistogramPlotter: Histogram visualization from Data objects.
    TopoPlotter: Topographical map visualization from Data objects.

Functions:
    time_window: Extract time window around position
    power_time_window: Calculate power in time window
    compute_feature_statistic: Compute feature summary in time window
    format_tick_value: Format values for axis labels
    plot_timeseries: Convenience function to create time series plots
    plot_heatmap: Convenience function to create heatmap plots
    plot_histogram: Convenience function to create histogram plots
    plot_topomap: Convenience function to create topographical maps
"""

from __future__ import annotations

from .interactive import InteractiveExplorer

__all__ = ["InteractiveExplorer"]
