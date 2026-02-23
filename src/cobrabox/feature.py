"""Feature module providing access to feature functions.

This module re-exports feature functions for convenient access via cb.feature.
"""

from .features import sliding_window, line_length, dummy

__all__ = ["sliding_window", "line_length", "dummy"]
