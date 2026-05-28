"""Feature module with automatic discovery across cobrabox domain packages.

Every subdirectory of ``src/cobrabox/`` except ``egg/`` and ``_*`` is treated as
a feature domain. Each ``.py`` file (excluding ``__init__.py`` and ``_*.py``)
is imported, and any callable with ``_is_cobrabox_feature = True`` is collected.

Two access paths:
    cb.feature.LineLength          # flat convenience namespace
    cb.signalstats.LineLength      # domain-specific access
"""

from __future__ import annotations

import importlib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
_PACKAGE_NAME = __package__ or "cobrabox"

# Domains that should NOT be scanned for features.
_DOMAIN_BLOCKLIST = {"egg", "__pycache__"}

_discovered: dict[str, object] = {}

for domain_dir in PACKAGE_ROOT.iterdir():
    if not domain_dir.is_dir():
        continue
    if domain_dir.name in _DOMAIN_BLOCKLIST or domain_dir.name.startswith("_"):
        continue
    for module_path in domain_dir.rglob("*.py"):
        if module_path.name == "__init__.py" or module_path.name.startswith("_"):
            continue
        rel_path = module_path.relative_to(PACKAGE_ROOT)
        module_name = ".".join(rel_path.with_suffix("").parts)
        full_module_name = f"{_PACKAGE_NAME}.{module_name}"

        try:
            _module = importlib.import_module(full_module_name)
        except Exception:
            continue

        for _name, _obj in vars(_module).items():
            if (
                callable(_obj)
                and getattr(_obj, "_is_cobrabox_feature", False)
                and getattr(_obj, "__module__", "") == full_module_name
            ):
                if _name in _discovered:  # pragma: no cover
                    raise ValueError(
                        f"Duplicate feature name '{_name}' while importing "
                        f"module '{full_module_name}'."
                    )
                _discovered[_name] = _obj

globals().update(_discovered)
__all__ = [*sorted(_discovered.keys())]  # noqa: PLE0604
