from __future__ import annotations

import importlib
import pkgutil

_discovered: dict[str, object] = {}

for _mod in pkgutil.iter_modules(__path__):
    _module = importlib.import_module(f"{__name__}.{_mod.name}")
    for _name, _obj in vars(_module).items():
        if callable(_obj) and getattr(_obj, "_is_cobrabox_feature", False):
            if _name in _discovered:
                raise ValueError(
                    f"Duplicate feature function name '{_name}' found while importing "
                    f"module '{_module.__name__}'."
                )
            _discovered[_name] = _obj

globals().update(_discovered)
__all__ = [*sorted(_discovered.keys())]  # noqa: PLE0604
