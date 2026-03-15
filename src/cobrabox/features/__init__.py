from __future__ import annotations

import importlib
from pathlib import Path

# Auto-discover features
_discovered: dict[str, object] = {}

# Recursively scan all subdirectories and files
FEATURES_DIR = Path(__file__).parent
for module_path in FEATURES_DIR.rglob("*.py"):
    if module_path.name == "__init__.py" or module_path.name.startswith("_"):
        continue

    rel_path = module_path.relative_to(FEATURES_DIR)
    module_name = ".".join(rel_path.with_suffix("").parts)
    full_module_name = f"{__name__}.{module_name}"

    try:
        _module = importlib.import_module(full_module_name)
        for _name, _obj in vars(_module).items():
            if (
                callable(_obj)
                and getattr(_obj, "_is_cobrabox_feature", False)
                and getattr(_obj, "__module__", "") == full_module_name
            ):
                if _name in _discovered:  # pragma: no cover
                    raise ValueError(
                        f"Duplicate feature function name '{_name}' found while importing "
                        f"module '{full_module_name}'."
                    )
                _discovered[_name] = _obj
    except Exception:
        pass

globals().update(_discovered)
__all__ = [*sorted(_discovered.keys())]  # noqa: PLE0604
