"""The one-shot functional API (GH #116).

    o = cb.autocorrelation(d, "time", 250.0, 10)
    o = cb.Autocorrelation("time", 250.0, 10).apply(d)

Both forms are permanent: functions for a single call, classes for composition
(``|``), serialization and ``Chord``. These tests pin the contract between them.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from typing import Any

import numpy as np
import pytest

import cobrabox as cb
from cobrabox._functional import function_name, has_functional_form, public_fields

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
FUNCTIONAL = [n for n in FEATURE_NAMES if has_functional_form(getattr(cb, n))]
CLASS_ONLY = [n for n in FEATURE_NAMES if not has_functional_form(getattr(cb, n))]


@pytest.fixture
def data() -> cb.SignalData:
    """Multi-channel signal with a sampling rate — enough for most features to run."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(256, 4))
    return cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=128.0, subjectID="sub-01"
    )


def test_the_split_is_what_we_expect() -> None:
    """Aggregators are the deliberate gap; everything else gets a function."""
    assert sorted(CLASS_ONLY) == ["ConcatAggregate", "MeanAggregate"]
    assert len(FUNCTIONAL) == 39


@pytest.mark.parametrize("name", FUNCTIONAL)
def test_function_exists_at_root_and_domain(name: str) -> None:
    cls = getattr(cb, name)
    fn_name = function_name(cls)
    domain = cls.__module__.split(".")[1]
    module = importlib.import_module(f"cobrabox.{domain}")

    assert hasattr(cb, fn_name), f"{fn_name} missing from the root namespace"
    assert fn_name in cb.__all__, f"{fn_name} missing from cb.__all__"
    assert getattr(cb, fn_name) is getattr(module, fn_name)
    assert fn_name in module.__all__


@pytest.mark.parametrize("name", CLASS_ONLY)
def test_aggregators_have_no_function(name: str) -> None:
    """Aggregators fold a splitter's stream; standalone they invite misuse."""
    cls = getattr(cb, name)
    assert not hasattr(cb, function_name(cls))


@pytest.mark.parametrize("name", FUNCTIONAL)
def test_function_name_comes_from_the_module(name: str) -> None:
    """Not from the class: CamelCase splitting gives `e_m_d` / `s_v_d`."""
    cls = getattr(cb, name)
    module_name = cls.__module__.rsplit(".", 1)[-1]
    assert function_name(cls) == module_name.removeprefix("_")


def test_acronym_features_are_named_sensibly() -> None:
    """The reason names come from filenames rather than the class name."""
    assert cb.emd.__name__ == "emd"
    assert cb.svd.__name__ == "svd"


@pytest.mark.parametrize("name", FUNCTIONAL)
def test_signature_mirrors_the_dataclass(name: str) -> None:
    """`(data, *fields)` — so positional calls work, as the issue asked."""
    cls = getattr(cb, name)
    params = list(inspect.signature(getattr(cb, function_name(cls))).parameters)
    assert params == ["data", *[f.name for f in public_fields(cls)]]


def _outcome(call: Any) -> tuple[str, Any]:
    """Run a call, returning either its value or the exception it raised."""
    try:
        return ("ok", call())
    except Exception as exc:
        return ("raised", (type(exc), str(exc)))


@pytest.mark.parametrize("name", FUNCTIONAL)
def test_function_matches_class_form(name: str, data: cb.SignalData) -> None:
    """`cb.f(d)` and `cb.F().apply(d)` must be indistinguishable.

    Compared by outcome rather than by value so features that reject this fixture
    still assert something meaningful: both forms must fail the same way. Features
    with required parameters are skipped — there is no default call to compare.
    """
    cls = getattr(cb, name)
    if any(
        f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        for f in public_fields(cls)
    ):
        pytest.skip(f"{name} has required parameters; no default call to compare")

    fn = getattr(cb, function_name(cls))
    fn_kind, fn_result = _outcome(lambda: fn(data))
    cls_kind, cls_result = _outcome(
        lambda: cls()(data) if issubclass(cls, cb.SplitterFeature) else cls().apply(data)
    )

    assert fn_kind == cls_kind, f"{name}: function {fn_kind}, class {cls_kind}"
    if fn_kind == "raised":
        assert fn_result == cls_result
        return

    if issubclass(cls, cb.SplitterFeature):
        fn_windows = [w.data.values for w in fn_result]
        cls_windows = [w.data.values for w in cls_result]
        assert len(fn_windows) == len(cls_windows)
        for a, b in zip(fn_windows, cls_windows, strict=True):
            np.testing.assert_allclose(a, b)
    else:
        np.testing.assert_allclose(fn_result.data.values, cls_result.data.values, equal_nan=True)
        assert fn_result.history == cls_result.history


@pytest.mark.parametrize("name", FUNCTIONAL)
def test_function_is_bound_to_its_feature_class(name: str) -> None:
    cls = getattr(cb, name)
    fn = getattr(cb, function_name(cls))
    assert fn.__wrapped_feature__ is cls


def test_positional_parameters_work(data: cb.SignalData) -> None:
    """The issue asked for `o = cb.function(a, b, c)`, not keyword-only."""
    positional = cb.correlation(data, "time", "spearman")
    keyword = cb.Correlation(dim="time", method="spearman").apply(data)
    np.testing.assert_allclose(positional.data.values, keyword.data.values)


def test_parameters_actually_reach_the_feature(data: cb.SignalData) -> None:
    """A non-default parameter must change the result, not be silently dropped."""
    pearson = cb.correlation(data, method="pearson")
    spearman = cb.correlation(data, method="spearman")
    assert not np.allclose(pearson.data.values, spearman.data.values)
    np.testing.assert_allclose(
        spearman.data.values, cb.Correlation(method="spearman").apply(data).data.values
    )


def test_factory_default_is_not_passed_through(data: cb.SignalData) -> None:
    """`<factory>` is signature decoration; handing it back must not reach the dataclass."""
    sentinel = inspect.signature(cb.bandpass_filter).parameters["bands"].default
    assert repr(sentinel) == "<factory>"
    np.testing.assert_allclose(
        cb.bandpass_filter(data, bands=sentinel).data.values,
        cb.BandpassFilter().apply(data).data.values,
    )


def test_splitter_returns_a_stream(data: cb.SignalData) -> None:
    windows = list(cb.sliding_window(data, window_size=64, step_size=32))
    assert len(windows) > 1
    assert all(isinstance(w, cb.Data) for w in windows)


def test_docstring_shows_the_functional_form() -> None:
    """Composed, not copied — the class docstring's example shows the class form."""
    doc = cb.correlation.__doc__ or ""
    assert ">>> result = cb.correlation(" in doc
    assert "cb.Correlation(...).apply(data)" in doc


def test_no_function_name_collides_with_anything_public() -> None:
    names = [function_name(getattr(cb, n)) for n in FUNCTIONAL]
    assert len(names) == len(set(names)), "two features derive the same function name"
    assert not set(names) & set(FEATURE_NAMES), "a function name shadows a feature class"


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_all_lists_both_forms(domain: str) -> None:
    module = importlib.import_module(f"cobrabox.{domain}")
    for exported in module.__all__:
        assert hasattr(module, exported)
