"""Cobrabox visualization framework.

Usage::

    import cobrabox as cb
    from cobrabox.visualization import explore

    data = cb.dataset("dummy_chain")[0]
    app = explore(data)          # returns CobraboxApp (displays in notebook)
    app.as_template()            # returns FastListTemplate for panel serve
"""

from __future__ import annotations

from typing import Any

from cobrabox.data import Data

from .app import CobraboxApp

__all__ = ["CobraboxApp", "explore"]


def explore(data: Data, **kwargs: Any) -> CobraboxApp:
    """Create an interactive visualization app for the given Data."""
    return CobraboxApp(data=data, **kwargs)
