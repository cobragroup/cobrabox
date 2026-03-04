# Feature Auto-Discovery: Static Analysis Problem

**Date:** 2026-02-27
**Updated:** 2026-03-04
**Status:** Discussion / Decision needed

## What's happening

`src/cobrabox/features/__init__.py` auto-discovers all feature classes at import time by scanning every module in the `features/` package:

```python
for _mod in pkgutil.iter_modules(__path__):
    _module = importlib.import_module(f"{__name__}.{_mod.name}")
    for _name, _obj in vars(_module).items():
        if (
            callable(_obj)
            and getattr(_obj, "_is_cobrabox_feature", False)
            and getattr(_obj, "__module__", "") == _module.__name__
        ):
            _discovered[_name] = _obj

globals().update(_discovered)
__all__ = [*sorted(_discovered.keys())]  # noqa: PLE0604
```

`_is_cobrabox_feature = True` is a `ClassVar` on `BaseFeature`, `SplitterFeature`, and `AggregatorFeature`, so every subclass inherits it automatically. The `__module__` guard ensures base classes imported into a feature file are not re-registered as duplicates.

**At runtime this works perfectly.** Adding a new file in `features/` is enough — no manual registration needed.

## The problems

### 1. Lint suppression required

Ruff's PLE0604 checks that names in `__all__` are statically defined in the module. Because `globals().update()` is a runtime operation, ruff can't see those names and flags them as undefined. Hence the `# noqa: PLE0604`.

### 2. IDEs and type checkers are blind

Pyright, mypy, and IDE language servers do static analysis. They cannot follow `globals().update()`, so `from cobrabox.features import LineLength` appears undefined to them — red squiggles everywhere.

### 3. `cobrabox/__init__.py` has hardcoded imports as a workaround

```python
from .features.line_length import LineLength
from .features.mean_aggregate import MeanAggregate
from .features.sliding_window import SlidingWindow
```

Only three of the 14+ feature classes are re-exported from the top-level package. This:

- Defeats the purpose of auto-discovery (new features must be added here too)
- Is still broken at the `features` level for anyone importing from there directly

## Options

### Option A — `features/__init__.pyi` stub (auto-generated)

Write a script (`scripts/gen_stubs.py`) that inspects the `features/` directory and generates a stub file:

```python
# features/__init__.pyi  (auto-generated — do not edit)
from .bandpower import Bandpower as Bandpower
from .coherence import Coherence as Coherence
from .line_length import LineLength as LineLength
from .mean_aggregate import MeanAggregate as MeanAggregate
from .sliding_window import SlidingWindow as SlidingWindow
# ... etc
```

Run it as a pre-commit hook so it stays in sync automatically.

**Pros:** Fully automatic after setup; IDEs and type checkers happy; `noqa` can be removed.
**Cons:** Adds tooling complexity; stub file is a generated artifact (commit it or gitignore it?).

### Option B — `features/__init__.pyi` stub (manually maintained)

Same stub file, but updated by hand when a feature is added. Document this in `CLAUDE.md`.

**Pros:** No extra tooling; IDEs and type checkers happy; `noqa` can be removed.
**Cons:** Easy to forget; contributor friction; slightly defeats "no registration needed" goal.

### Option C — Explicit star imports per module

Replace the dynamic discovery with explicit star imports in `features/__init__.py`:

```python
from .bandpower import *
from .coherence import *
from .line_length import *
from .mean_aggregate import *
from .sliding_window import *
# ... etc
```

Each feature module owns its `__all__`.

**Pros:** Fully static; no stubs needed; no `noqa`; IDE support for free.
**Cons:** Requires touching `features/__init__.py` when adding a new feature. Effectively removes auto-discovery.

### Option D — Accept the current state

Keep `# noqa: PLE0604`, keep hardcoded imports in `cobrabox/__init__.py`, document the pattern.

**Pros:** Zero changes needed now.
**Cons:** Ongoing contributor confusion; lint suppression feels like a code smell; IDE experience stays broken; only 3 of 14+ features are accessible via `cb.*`.

## Recommendation (for discussion)

**Option A** is the cleanest long-term. A small `gen_stubs.py` + pre-commit hook keeps everything automatic while making the codebase fully understood by tools.

Regardless of which option is chosen, `cobrabox/__init__.py` should expose all feature classes — either via `from .features import *` (if the stub/static approach is adopted) or by keeping the hardcoded list complete and updated.

## Files involved

- `src/cobrabox/features/__init__.py` — dynamic discovery lives here
- `src/cobrabox/__init__.py` — top-level API, currently has hardcoded imports for only 3 features
- `src/cobrabox/features/__init__.pyi` — does not exist yet; would be created by Options A or B
- `scripts/gen_stubs.py` — does not exist yet; would be created by Option A
