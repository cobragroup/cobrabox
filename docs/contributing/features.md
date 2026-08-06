# Contributing a Feature

Everything you need to add a feature to CobraBox, in order.

## The checklist

Copy this into your PR description and tick it off.

- [ ] **1.** Branch off `main`
- [ ] **2.** Pick a domain → [table below](#1-pick-a-domain)
- [ ] **3.** Create `src/cobrabox/<domain>/_my_feature.py` — **note the leading underscore**
- [ ] **4.** Write the `@dataclass`, inheriting the right base class
- [ ] **5.** Add `_tags` so it appears in the docs' tag filter
- [ ] **6.** Seed the one-shot function: `uv run python scripts/gen_functional_wrappers.py`
- [ ] **7.** Tune the seeded function's docstring
- [ ] **8.** Re-export the class **and** function from `src/cobrabox/__init__.py`
- [ ] **9.** Write tests in `tests/features/<domain>/test_feature_my_feature.py`
- [ ] **10.** Regenerate stubs and docs
- [ ] **11.** Lint, format, run the suite, open a PR

Step 8 is the one people forget — the domain `__init__.py` is written for you by
step 6, but the root re-export is by hand. `tests/test_public_api.py` and
`tests/test_functional_api.py` fail loudly if you skip it.

---

## 1. Pick a domain

Each domain answers one question about the signal.

| Domain | Question it answers | Examples |
| ------ | ------------------- | -------- |
| `signalstats/` | What are the basic properties of my signal? | `LineLength`, `Mean`, `Autocorrelation`, `SpikeCount` |
| `infometrics/` | How complex or irregular is my signal? | `SampleEntropy`, `LempelZiv`, `FractalDimension` |
| `spectral/` | What is happening in frequency space? | `BandPower`, `Spectrogram`, `Cordance` |
| `connectivity/` | Which channels interact? | `Correlation`, `Coherence`, `GrangerCausality` |
| `transforms/` | Convert to another representation | `AnalyticSignal`, `FourierTransform` |
| `decompositions/` | Break into components | `BandpassFilter`, `EMD`, `SVD` |
| `surrogates/` | Test statistical significance | `FourierTransformSurrogates` |
| `windowing/` | Analyse temporal dynamics | `SlidingWindow`, `MeanAggregate` |

## 2. Pick a base class

| You want to… | Inherit from | Returns |
| ------------ | ------------ | ------- |
| Transform any `Data` → `Data` | `BaseFeature[Data]` | `xr.DataArray` or `Data` |
| Transform time-series → `Data` | `BaseFeature[SignalData]` | `xr.DataArray` or `Data` |
| Split into a stream of windows | `SplitterFeature[SignalData]` | `Iterator[Data]` |
| Fold a stream back into one `Data` | `AggregatorFeature` | `Data` |

Use `[SignalData]` when the algorithm **inherently** needs time-series structure —
it reads `sampling_rate`, or applies an FFT or Hilbert transform along time. Use
`[Data]` when the dimension is just a parameter the caller chooses, even if every
caller happens to pass time-series.

## 3. Write the file

The filename is **private** (leading underscore) and the class is public:

```python
# src/cobrabox/signalstats/_variance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data


@dataclass
class Variance(BaseFeature[Data]):
    """Compute variance over a dimension.

    Args:
        dim: Dimension to reduce over. Defaults to ``"time"``.

    Returns:
        :class:`~cobrabox.Data` with ``dim`` removed.

    Example:
        >>> result = cb.Variance(dim="time").apply(data)
    """

    _tags: ClassVar[list[str]] = ["variability", "eeg", "io:scalar-per-channel"]
    output_type: ClassVar[type[Data]] = Data

    dim: str = "time"

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim {self.dim!r} not found in {data.data.dims}")
        return data.data.var(dim=self.dim)
```

### Why the underscore

`Variance` in `variance.py` would make `cb.signalstats.variance` resolve to the
*module*, squatting on the name the one-shot function needs. Private
implementation modules are the convention scipy and scikit-learn use
(`scipy.stats.entropy` lives in `scipy.stats._entropy`). See
[GH #116](https://github.com/cobragroup/cobrabox/issues/116).

### Things worth getting right

- **`output_type`** — set `ClassVar[type[Data]] = Data` when your feature *removes*
  the time dimension (scalars, matrices, frequency-only output). Without it
  `apply()` tries to keep the input container type and will fail.
- **`_tags`** — drives the tag filter on the docs home page. Reuse existing tags
  where you can; see [Features by Tag](../tags.md). `io:*` describes the output
  shape, `req:*` an input requirement.
- **Validate and raise `ValueError`** with a message naming the offending value.
- **Docstring** — Google style with an `Example:`. It is the single source for the
  API docs *and* the domain pages, so write it for a reader.
- **Never mutate** — `Data` is immutable; return new objects.

## 4. Seed the one-shot function

Every feature has a companion function — `cb.variance(data, dim="time")` beside
`cb.Variance(dim="time").apply(data)`. You don't hand-write it; a generator seeds
it into your feature file, below the class:

```bash
uv run python scripts/gen_functional_wrappers.py
```

This appends a `@functional(Variance)`-decorated `def variance(data, dim="time")`
to `_variance.py` and re-exports it from the domain `__init__.py`. The name is
your filename minus the underscore, so `_variance.py` → `cb.variance`. The
generator is **non-destructive** — it only seeds features that lack a wrapper, so
re-running it never touches a wrapper you've since edited. Aggregators get no
function (they fold a splitter's stream, so a standalone call is meaningless).

**Then tune the seeded docstring.** It starts as a copy of your class docstring
with the `Example:` rewritten to the functional call — a full starting point, but
give it a read and adjust anything that reads oddly out of the class's context.

Because the wrapper is ordinary source in your file, `cb.variance?` in IPython
points straight at `_variance.py`, and type-checkers see its real signature.

## 5. Re-export from the root

Auto-discovery already gives you `cb.feature.Variance` / `cb.feature.variance`,
and step 4 wrote the domain `__init__.py`. The **root** namespace is by hand — add
the class *and* function to `src/cobrabox/__init__.py`:

```python
from .signalstats import (
    ...,
    Variance,
    ...,
    variance,
)

__all__ = [..., "Variance", ..., "variance", ...]
```

## 6. Write tests

Location mirrors the source: `tests/features/<domain>/test_feature_<name>.py`.

```python
# tests/features/signalstats/test_feature_variance.py
from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_variance_reduces_the_time_dimension() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.Variance(dim="time").apply(data)

    assert "time" not in out.data.dims
    np.testing.assert_allclose(out.to_numpy(), np.var(arr, axis=0))
    assert out.history == ["Variance"]


def test_variance_rejects_an_unknown_dimension() -> None:
    data = cb.SignalData.from_numpy(np.ones((5, 3)), dims=["time", "space"])
    with pytest.raises(ValueError, match="band_index"):
        cb.Variance(dim="band_index").apply(data)
```

Cover at minimum:

- correct values on input you can verify by hand
- `history` contains the class name
- metadata (`subjectID`, `sampling_rate`) survives
- `ValueError` on a bad dimension or parameter

You do **not** need to test that `cb.variance(data)` matches
`cb.Variance().apply(data)` — `tests/test_functional_api.py` already does that for
every feature.

## 7. Regenerate the generated files

```bash
uv run python scripts/gen_stubs.py         # .pyi for IDEs and type-checkers
uv run python scripts/gen_feature_docs.py  # domain pages, tag page, API page
```

Both are idempotent and exit `0` when nothing changed. `gen_stubs.py` also runs
from pre-commit. Commit whatever they touch.

## 8. Check it

```bash
uv run pytest -q
uvx ruff check --fix . && uvx ruff format .
```

Then confirm all the access paths work:

```python
import cobrabox as cb

cb.variance(data)  # one-shot function
cb.Variance().apply(data)  # class
cb.signalstats.Variance  # domain
cb.feature.Variance  # flat registry
```

## 9. Open a PR

```bash
git add src/cobrabox/signalstats/_variance.py \
        src/cobrabox/signalstats/__init__.py \
        src/cobrabox/__init__.py \
        tests/features/signalstats/test_feature_variance.py \
        src/cobrabox/**/*.pyi docs/
git commit -m "Add Variance feature"
```

See [Pull Requests](pr.md) for the rest.

## When something goes wrong

These are the actual failures, checked by making each mistake on purpose.

| Failure | Cause |
| ------- | ----- |
| `test_feature_lives_in_a_private_module` | Your file is missing its leading underscore. Note nothing *breaks* — the generated function quietly overwrites the module attribute — so this test is the only thing that tells you |
| `test_every_feature_belongs_to_exactly_one_domain` | Same cause; fires alongside the above |
| `test_feature_is_on_root_namespace` | Class not re-exported from `src/cobrabox/__init__.py` (step 7). Expect `test_root_and_flat_namespace_agree` and `test_domain_namespace_agrees_with_root` to fail with it |
| `test_function_exists_at_root_and_domain` | The *function* is missing from a re-export, even though the class is there |
| `test_domain_namespace_exposes_no_modules` | A public `.py` in a domain is exposed as a module attribute — a helper without an underscore, rather than a feature |
| `test_prose_reference_resolves` | A guide or example mentions a `cb.X` that no longer exists |
| `apply()` raises about dimensions | You removed a dimension without setting `output_type` |
| Feature absent from `cb.feature.__all__` | Discovery could not import your module. Import it directly to see the real error — discovery swallows it |

## Reference implementations

Read one close to what you are building.

| Pattern | File |
| ------- | ---- |
| Simplest `BaseFeature[SignalData]` | `src/cobrabox/signalstats/_line_length.py` |
| Parameters and validation | `src/cobrabox/spectral/_band_power.py` |
| Generic, caller picks the dim | `src/cobrabox/signalstats/_mean.py` |
| Shared private helpers | `src/cobrabox/connectivity/_coherence.py` (uses `_mvar.py`) |
| Splitter | `src/cobrabox/windowing/_sliding_window.py` |
| Aggregator (builds history by hand) | `src/cobrabox/windowing/_mean_aggregate.py` |

`src/cobrabox/_dummy.py` is a deliberate **negative** example — no useful
docstring, no validation. Do not model anything on it.
