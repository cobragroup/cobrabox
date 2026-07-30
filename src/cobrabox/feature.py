"""Feature module with automatic discovery across cobrabox domain packages.

Every subdirectory of ``src/cobrabox/`` except ``egg/`` and ``_*`` is treated as
a feature domain. Each ``.py`` file except ``__init__.py`` is imported, and any
callable with ``_is_cobrabox_feature = True`` is collected.

Implementation modules are private (``_autocorrelation.py``), following the
convention used by scipy and scikit-learn, so that the public lower-case name is
free for the functional API. Discovery therefore *includes* ``_*.py`` rather than
skipping it — private helpers such as ``connectivity/_mvar.py`` are imported
harmlessly and contribute nothing, since the filter is on
``_is_cobrabox_feature``, not on the filename.

Access paths:
    cb.LineLength                  # canonical
    cb.signalstats.LineLength      # domain-specific
    cb.feature.LineLength          # flat convenience namespace
"""

from __future__ import annotations

import importlib as _importlib
from pathlib import Path as _Path

_PACKAGE_ROOT = _Path(__file__).parent
_PACKAGE_NAME = __package__ or "cobrabox"

# Domains that should NOT be scanned for features.
_DOMAIN_BLOCKLIST = {"egg", "__pycache__"}


def _discover() -> dict[str, object]:
    """Import every domain module and collect classes flagged as features.

    Wrapped in a function so loop temporaries do not leak into the module
    namespace — only the discovered feature classes end up on ``cb.feature``.
    """
    discovered: dict[str, object] = {}
    for domain_dir in _PACKAGE_ROOT.iterdir():
        if not domain_dir.is_dir():
            continue
        if domain_dir.name in _DOMAIN_BLOCKLIST or domain_dir.name.startswith("_"):
            continue
        for module_path in domain_dir.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue
            rel_path = module_path.relative_to(_PACKAGE_ROOT)
            module_name = ".".join(rel_path.with_suffix("").parts)
            full_module_name = f"{_PACKAGE_NAME}.{module_name}"

            try:
                module = _importlib.import_module(full_module_name)
            except Exception:
                continue

            for name, obj in vars(module).items():
                # A feature class (`_is_cobrabox_feature`) or its one-shot function
                # (`_is_cobrabox_feature_function`, set by @functional). Both are
                # collected so `cb.feature` carries `Correlation` and `correlation`.
                is_feature = getattr(obj, "_is_cobrabox_feature", False) or getattr(
                    obj, "_is_cobrabox_feature_function", False
                )
                if (
                    callable(obj)
                    and is_feature
                    and getattr(obj, "__module__", "") == full_module_name
                ):
                    if name in discovered:  # pragma: no cover
                        raise ValueError(
                            f"Duplicate feature name '{name}' while importing "
                            f"module '{full_module_name}'."
                        )
                    discovered[name] = obj
    return discovered


_discovered = _discover()
globals().update(_discovered)
__all__ = [*sorted(_discovered.keys())]  # noqa: PLE0604
