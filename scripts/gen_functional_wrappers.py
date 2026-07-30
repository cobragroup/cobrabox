#!/usr/bin/env python3
"""Seed the one-shot functional wrapper for any feature that lacks one (GH #116).

For each feature class it writes a module-level ``def feature_name(data, ...)``
at the end of the class's own file, decorated with ``@functional(TheClass)``, and
re-exports it from the domain ``__init__.py``. The wrapper's docstring is seeded
from the class docstring, with the ``Example:`` call rewritten to the functional
form — a full starting point that maintainers then tune by hand.

**One-shot, non-destructive.** A file that already defines its wrapper is left
untouched, so running this again only scaffolds *new* features and never
clobbers a hand-tuned docstring. It is therefore NOT part of pre-commit; run it
when you add a feature, then edit the seeded wrapper like any other source.

    uv run python scripts/gen_functional_wrappers.py            # apply
    uv run python scripts/gen_functional_wrappers.py --check    # list what's missing
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import subprocess
import sys
import typing
from pathlib import Path

import cobrabox as cb
from cobrabox._functional import function_name, has_functional_form, public_fields
from cobrabox.base_feature import SplitterFeature

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "cobrabox"
DOMAINS = [
    "connectivity",
    "decompositions",
    "infometrics",
    "signalstats",
    "spectral",
    "surrogates",
    "transforms",
    "windowing",
]


def _data_bound(cls: type) -> str:
    """The generic bound a feature declares, e.g. ``BaseFeature[SignalData]`` -> ``SignalData``."""
    for base in getattr(cls, "__orig_bases__", ()):
        args = typing.get_args(base)
        if args:
            return args[0].__name__
    return "Data"


def _read_balanced(text: str, k: int) -> tuple[str, int]:
    """Read up to the paren matching the one just opened at ``text[k-1]``.

    Returns the inner content (without the outer parens) and the index past the
    closing paren.
    """
    depth, start = 1, k
    while k < len(text) and depth:
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
        k += 1
    return text[start : k - 1], k


def _rewrite_calls(text: str, class_name: str, func_name: str) -> str:
    """Rewrite ``cb.Class(args).apply(x)`` / ``cb.Class(args)(x)`` -> ``cb.func(x, args)``."""
    needle = f"cb.{class_name}("
    out: list[str] = []
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        args, k = _read_balanced(text, j + len(needle))
        if text.startswith(".apply(", k):
            x, k = _read_balanced(text, k + len(".apply("))
        elif text.startswith("(", k):
            x, k = _read_balanced(text, k + 1)
        else:
            x = None  # bare construction, no call to fold data into
        if x is None:
            new_args = args
        else:
            new_args = x + (", " + args if args.strip() else "")
        out.append(f"cb.{func_name}({new_args})")
        i = k


def _signature(cls: type, func_name: str) -> str:
    parts = [f"data: {_data_bound(cls)}"]
    for f in public_fields(cls):
        ann = f.type if isinstance(f.type, str) else typing.get_type_hints(cls).get(f.name, "Any")
        if f.default is not dataclasses.MISSING:
            parts.append(f"{f.name}: {ann} = {f.default!r}")
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            # No single default object to show; accept None and let the class
            # apply its own factory (see body).
            optional = ann if "None" in ann else f"{ann} | None"
            parts.append(f"{f.name}: {optional} = None")
        else:
            parts.append(f"{f.name}: {ann}")
    returns = "Iterator[Data]" if issubclass(cls, SplitterFeature) else "Data"
    return f"def {func_name}({', '.join(parts)}) -> {returns}:"


def _body(cls: type) -> str:
    fields = public_fields(cls)
    factory = [f for f in fields if f.default_factory is not dataclasses.MISSING]  # type: ignore[misc]
    passthrough = [f for f in fields if f not in factory]
    call = ", ".join(f"{f.name}={f.name}" for f in passthrough)
    tail = "(data)" if issubclass(cls, SplitterFeature) else ".apply(data)"
    ctor = cls.__name__
    if not factory:
        return f"    return {ctor}({call}){tail}"
    if len(factory) > 1:  # pragma: no cover - no feature has two factory fields
        raise NotImplementedError(f"{ctor} has multiple default_factory fields")
    # A ``None`` for the factory field means "use the class default", so branch
    # rather than unpack a loosely-typed dict — keeps the wrapper fully type-checked.
    ff = factory[0].name
    with_ff = (call + ", " if call else "") + f"{ff}={ff}"
    return (
        f"    if {ff} is None:\n"
        f"        return {ctor}({call}){tail}\n"
        f"    return {ctor}({with_ff}){tail}"
    )


def _docstring(cls: type, func_name: str) -> str:
    doc = inspect.getdoc(cls) or f"Apply :class:`~cobrabox.{cls.__name__}` to ``data``."
    doc = _rewrite_calls(doc, cls.__name__, func_name)
    lines = doc.split("\n")
    body = "\n".join((f"    {ln}").rstrip() if ln else "" for ln in lines[1:])
    return f'    """{lines[0]}\n{body}\n    """' if len(lines) > 1 else f'    """{lines[0]}"""'


def _wrapper_source(cls: type) -> str:
    func_name = function_name(cls)
    return (
        f"@functional({cls.__name__})\n"
        f"{_signature(cls, func_name)}\n"
        f"{_docstring(cls, func_name)}\n"
        f"{_body(cls)}\n"
    )


def _ensure_import(text: str) -> str:
    # Feature files import their base class either relatively (`from ..base_feature`)
    # or absolutely (`from cobrabox.base_feature`); match the file's own style so the
    # injected import is consistent and isort-clean.
    anchor = re.search(r"^from (\.\.|cobrabox\.)base_feature import .*$", text, re.M)
    if anchor is None:  # pragma: no cover - every feature imports a base class
        raise RuntimeError("no base_feature import to anchor the functional import to")
    imp = f"from {anchor.group(1)}_functional import functional\n"
    if imp.strip() in text:
        return text
    return text[: anchor.start()] + imp + text[anchor.start() :]


def _features_by_domain() -> dict[str, list[type]]:
    # cb.feature.__all__ carries both classes and functions; the wrappers derive
    # from the classes.
    by_domain: dict[str, list[type]] = {d: [] for d in DOMAINS}
    for name in cb.feature.__all__:
        # Read from the discovered registry, not the root namespace, so a brand-new
        # feature works before it has been re-exported from cobrabox/__init__.py.
        obj = getattr(cb.feature, name)
        if isinstance(obj, type):
            by_domain[obj.__module__.split(".")[1]].append(obj)
    return by_domain


def _regen_init(domain: str, classes: list[type]) -> str:
    path = PACKAGE_ROOT / domain / "__init__.py"
    original = path.read_text(encoding="utf-8")
    # Preserve the leading comment block that names the domain.
    lead = []
    for line in original.splitlines():
        if line.startswith(("from ", "import ")) or (line and not line.startswith("#")):
            break
        lead.append(line)
    imports, names = [], []
    for cls in sorted(classes, key=lambda c: c.__module__):
        module = cls.__module__.rsplit(".", 1)[-1]
        if has_functional_form(cls):
            fn = function_name(cls)
            imports.append(f"from .{module} import {cls.__name__}, {fn}")
            names += [cls.__name__, fn]
        else:
            imports.append(f"from .{module} import {cls.__name__}")
            names.append(cls.__name__)
    all_block = "__all__ = [\n" + "".join(f'    "{n}",\n' for n in sorted(names)) + "]\n"
    return "\n".join([*lead, *imports, "", all_block]).lstrip("\n")


def main(check: bool) -> int:
    by_domain = _features_by_domain()
    seeded, skipped = [], []
    for domain, classes in by_domain.items():
        for cls in classes:
            if not has_functional_form(cls):
                continue
            path = Path(inspect.getsourcefile(cls))  # type: ignore[arg-type]
            text = path.read_text(encoding="utf-8")
            if "@functional(" in text:
                skipped.append(function_name(cls))
                continue
            if check:
                seeded.append(function_name(cls))
                continue
            text = _ensure_import(text)
            path.write_text(text.rstrip() + "\n\n\n" + _wrapper_source(cls), encoding="utf-8")
            seeded.append(function_name(cls))
        if not check:
            (PACKAGE_ROOT / domain / "__init__.py").write_text(
                _regen_init(domain, classes), encoding="utf-8"
            )

    if check:
        print(f"{len(seeded)} feature(s) missing a wrapper: {sorted(seeded)}", file=sys.stderr)
        return 1 if seeded else 0

    print(f"Seeded {len(seeded)}, skipped {len(skipped)} already-present.", file=sys.stderr)
    if seeded:
        subprocess.run(["ruff", "check", "--fix", "--quiet", str(PACKAGE_ROOT)], check=False)
        subprocess.run(["ruff", "format", "--quiet", str(PACKAGE_ROOT)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
