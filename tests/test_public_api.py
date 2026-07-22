"""Guard the three ways a feature can be reached.

Since GH #116 the canonical form is ``cb.Correlation()``. ``cb.connectivity.Correlation()``
and ``cb.feature.Correlation()`` stay valid aliases, but all three must resolve to the
same class. A new feature file is picked up automatically by ``cb.feature`` and by its
domain ``__init__``, but the root namespace is written by hand in
``src/cobrabox/__init__.py`` — these tests are what catches a forgotten re-export.

The last test closes the loop that #116 opened: docstring examples are not doctested,
so a rename can leave ``>>> cb.OldName()`` behind with nothing failing.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re
import types

import pytest

import cobrabox as cb

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

FEATURE_NAMES = sorted(cb.feature.__all__)


def test_features_were_discovered() -> None:
    """A discovery bug that finds nothing would make every other test here vacuous."""
    assert len(FEATURE_NAMES) > 30


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_feature_is_on_root_namespace(name: str) -> None:
    assert hasattr(cb, name), (
        f"{name} is missing from the root namespace. Add it to the domain re-export "
        f"block in src/cobrabox/__init__.py."
    )
    assert name in cb.__all__, f"{name} is importable as cb.{name} but missing from cb.__all__."


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_root_and_flat_namespace_agree(name: str) -> None:
    assert getattr(cb, name) is getattr(cb.feature, name)


def test_every_feature_belongs_to_exactly_one_domain() -> None:
    """`cb.<domain>.X` must cover the whole catalog, with no name claimed twice."""
    owners: dict[str, list[str]] = {}
    for domain in DOMAINS:
        module = importlib.import_module(f"cobrabox.{domain}")
        for name in module.__all__:
            # `__all__` also carries the functional wrappers (GH #116); this test is
            # about the classes. test_functional_api.py covers the function side.
            if not isinstance(getattr(module, name, None), type):
                continue
            owners.setdefault(name, []).append(domain)

    duplicates = {n: ds for n, ds in owners.items() if len(ds) > 1}
    assert not duplicates, f"features exported by more than one domain: {duplicates}"
    assert sorted(owners) == FEATURE_NAMES


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_domain_namespace_agrees_with_root(name: str) -> None:
    domain = next(d for d in DOMAINS if name in importlib.import_module(f"cobrabox.{d}").__all__)
    module = importlib.import_module(f"cobrabox.{domain}")
    assert getattr(module, name) is getattr(cb, name)


def test_root_namespace_adds_nothing_unexported() -> None:
    """`cb.__all__` must not promise names that are not actually there."""
    missing = [name for name in cb.__all__ if not hasattr(cb, name)]
    assert not missing, f"cb.__all__ lists names that do not exist: {missing}"


# --- docstring examples ----------------------------------------------------

SRC_ROOT = pathlib.Path(cb.__file__).parent

# `_dummy.py` is the deliberate negative reference: it is excluded from discovery,
# so its `>>> cb.Dummy(...)` example is expected not to resolve.
DOCSTRING_SKIP = {"_dummy.py"}

# Captures the whole dotted path, so `cb.feature.OldName` and `cb.connectivity.OldName`
# are checked too — not just the first segment. Trailing `.data` / `.apply` on the result
# of a call are not matched, because the `(` ends the run.
_CB_REFERENCE = re.compile(r"\bcb((?:\.[A-Za-z_][A-Za-z0-9_]*)+)")


def _docstring_references() -> list[tuple[str, str]]:
    """[(relative_path, dotted_path), ...] for every `cb.X[.Y]` inside a docstring."""
    found: list[tuple[str, str]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name in DOCSTRING_SKIP or "egg" in path.relative_to(SRC_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            rel = str(path.relative_to(SRC_ROOT))
            found.extend((rel, dotted.lstrip(".")) for dotted in _CB_REFERENCE.findall(doc))
    return found


DOCSTRING_REFERENCES = sorted(set(_docstring_references()))


def test_docstrings_actually_reference_the_api() -> None:
    """Sanity check: if this finds nothing, the parametrised test below is vacuous."""
    assert len(DOCSTRING_REFERENCES) > 40


@pytest.mark.parametrize(("source_file", "dotted"), DOCSTRING_REFERENCES)
def test_docstring_reference_resolves(source_file: str, dotted: str) -> None:
    target: object = cb
    for part in dotted.split("."):
        assert hasattr(target, part), (
            f"{source_file} has a docstring example using `cb.{dotted}`, which does not "
            f"resolve on the public API. Docstring examples are not doctested, so a "
            f"rename leaves these behind silently (GH #116)."
        )
        target = getattr(target, part)


# --- namespace cleanliness (GH #116) ---------------------------------------

# Implementation modules are private (`_line_length.py`), and the top-level
# implementation modules are dropped from `cobrabox/__init__.py`, so no module
# object should be reachable as a public attribute. These are the namespaces we
# publish on purpose.
PUBLIC_MODULE_ATTRS = {"feature", "serialization", *DOMAINS}


def _public_module_attrs(namespace: object) -> list[str]:
    return [
        name
        for name in dir(namespace)
        if not name.startswith("_") and isinstance(getattr(namespace, name), types.ModuleType)
    ]


def test_root_namespace_exposes_only_intended_modules() -> None:
    assert sorted(_public_module_attrs(cb)) == sorted(PUBLIC_MODULE_ATTRS)


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_namespace_exposes_no_modules(domain: str) -> None:
    """A feature file must not squat on the lower-case name (GH #116).

    `Autocorrelation` lives in `_autocorrelation.py`, so `cb.signalstats` exposes
    the class and leaves `autocorrelation` free for the functional API. A new
    feature added as `autocorrelation.py` would fail here.
    """
    module = importlib.import_module(f"cobrabox.{domain}")
    assert _public_module_attrs(module) == []


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_feature_lives_in_a_private_module(name: str) -> None:
    module_name = getattr(cb, name).__module__.rsplit(".", 1)[-1]
    assert module_name.startswith("_"), (
        f"{name} lives in public module {module_name!r}; implementation modules must "
        f"be private so the lower-case name stays free."
    )


def test_dataset_is_not_a_module() -> None:
    """`cb.dataset` was a module, making `cb.dataset(...)` fail as 'not callable'.

    The loader is `cb.load_dataset`. `cb.dataset` must be absent so the mistake
    surfaces as a plain AttributeError.
    """
    assert not hasattr(cb, "dataset")
    assert callable(cb.load_dataset)
