# Feature Auto-Discovery: Static Analysis Problem

**Date:** 2026-02-27
**Status:** Discussion / Decision needed

## What's happening

`src/cobrabox/features/__init__.py` auto-discovers all `@feature`-decorated functions at import time:

```python
for _mod in pkgutil.iter_modules(__path__):
    _module = importlib.import_module(f"{__name__}.{_mod.name}")
    for _name, _obj in vars(_module).items():
        if callable(_obj) and getattr(_obj, "_is_cobrabox_feature", False):
            _discovered[_name] = _obj

globals().update(_discovered)
__all__ = [*sorted(_discovered.keys())]  # noqa: PLE0604
```

**At runtime this works perfectly.** Adding a new file in `features/` is enough — no manual registration needed.

## The problems

### 1. Lint suppression required

Ruff's PLE0604 checks that names in `__all__` are statically defined in the module. Because `globals().update()` is a runtime operation, ruff can't see those names and flags them as undefined. Hence the `# noqa: PLE0604`.

### 2. IDEs and type checkers are blind

Pyright, mypy, and IDE language servers do static analysis. They cannot follow `globals().update()`, so `from cobrabox.features import line_length` appears undefined to them — red squiggles everywhere.

### 3. `cobrabox/__init__.py` has hardcoded imports as a workaround

```python
from .features import line_length, sliding_window
```

This was added to make IDEs happy at the top-level, but it:
- Defeats the purpose of auto-discovery (new features must be added here too)
- Is still broken at the `features` level for anyone importing from there directly

## Options

### Option A — `features/__init__.pyi` stub (auto-generated)

Write a script (`scripts/gen_stubs.py`) that inspects the `features/` directory and generates a stub file:

```python
# features/__init__.pyi  (auto-generated — do not edit)
from .line_length import line_length as line_length
from .mean import mean as mean
from .sliding_window import sliding_window as sliding_window
```

Run it as a pre-commit hook so it stays in sync automatically.

**Pros:** Fully automatic after setup; IDEs and type checkers happy; `noqa` can be removed.
**Cons:** Adds tooling complexity; stub file is a generated artifact (commit it or gitignore it?).

### Option B — `features/__init__.pyi` stub (manually maintained)

Same stub file, but updated by hand when a feature is added. Document this in `CLAUDE.md` / `contributing_feature.md`.

**Pros:** No extra tooling; IDEs and type checkers happy; `noqa` can be removed.
**Cons:** Easy to forget; contributor friction; slightly defeats "no registration needed" goal.

### Option C — Explicit star imports per module

Replace the dynamic discovery with explicit star imports in `features/__init__.py`:

```python
from .line_length import *
from .mean import *
from .sliding_window import *
```

Each feature module owns its `__all__`.

**Pros:** Fully static; no stubs needed; no `noqa`; IDE support for free.
**Cons:** Requires touching `features/__init__.py` when adding a new feature (same friction as before, just in a different file). Effectively removes auto-discovery.

### Option D — Accept the current state

Keep `# noqa: PLE0604`, keep hardcoded imports in `cobrabox/__init__.py`, document the pattern.

**Pros:** Zero changes needed now.
**Cons:** Ongoing contributor confusion; lint suppression feels like a code smell; IDE experience stays broken.

## Recommendation (for discussion)

**Option A** is the cleanest long-term. A small `gen_stubs.py` + pre-commit hook keeps everything automatic while making the codebase fully understood by tools.

As a quick interim fix regardless of which option is chosen: replace the hardcoded imports in `cobrabox/__init__.py` with `from .features import *` so at least the top-level API stays automatically up to date.

## Files involved

- `src/cobrabox/features/__init__.py` — dynamic discovery lives here
- `src/cobrabox/__init__.py` — top-level API, currently has hardcoded feature imports
- `src/cobrabox/features/__init__.pyi` — does not exist yet; would be created by Options A or B
- `scripts/gen_stubs.py` — does not exist yet; would be created by Option A
