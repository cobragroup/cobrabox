"""Generate domain and tag documentation pages from the live feature registry.

Run with:  uv run python scripts/gen_feature_docs.py

Produces, under ``docs/``:
    domain/<domain>.md   one page per domain package, listing its features
    tags.md              all tags grouped by category, linking to features
    api/features.md      mkdocstrings directives, one per feature
    index.md             injects the interactive feature explorer widget

Everything is derived from the auto-discovered features (their module path,
docstring summary and ``_tags``), so the pages cannot drift out of sync with the
code. The feature catalog restructure proposal (§8) calls for cross-cutting tag
discovery; this is their generator.
"""

from __future__ import annotations

import html
import inspect
from pathlib import Path

import cobrabox as cb
from cobrabox._functional import function_name, has_functional_form

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# Human-readable domain titles and one-line descriptions (the research question
# each domain answers, per the proposal's domain principle).
DOMAIN_META: dict[str, tuple[str, str]] = {
    "windowing": ("Windowing & Aggregation", "How do I analyze temporal dynamics?"),
    "surrogates": ("Surrogates", "How do I test statistical significance?"),
    "transforms": ("Transforms", "How do I transform my signal into another representation?"),
    "signalstats": ("Signal Statistics", "What are the basic properties of my signal?"),
    "infometrics": ("Infometrics", "How complex/irregular is my signal?"),
    "spectral": ("Spectral", "What's happening in frequency space?"),
    "connectivity": ("Connectivity", "Which regions are synchronized/interacting?"),
    "decompositions": ("Decompositions", "How do I decompose my signal into components?"),
}


def _summary(cls: type) -> str:
    """First non-empty line of the class docstring."""
    doc = inspect.getdoc(cls) or ""
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _domain_of(cls: type) -> str:
    """Domain package name from ``cobrabox.<domain>.<module>``."""
    parts = cls.__module__.split(".")
    return parts[1] if len(parts) > 2 else "misc"


def _collect() -> dict[str, list[tuple[str, type]]]:
    """Map domain -> sorted list of (feature_name, class)."""
    by_domain: dict[str, list[tuple[str, type]]] = {}
    for name in cb.feature.__all__:
        cls = getattr(cb.feature, name)
        # cb.feature carries the one-shot functions too; the docs are built per class.
        if not isinstance(cls, type):
            continue
        by_domain.setdefault(_domain_of(cls), []).append((name, cls))
    for features in by_domain.values():
        features.sort(key=lambda item: item[0])
    return by_domain


# Tag taxonomy (proposal §8). Every non-prefixed tag maps to exactly one
# category; io:* and req:* are handled by prefix. Order here is display order.
_TAG_CATEGORIES: list[tuple[str, frozenset[str]]] = [
    ("Modality", frozenset({"eeg", "fmri", "meg", "ecg", "intracranial-eeg"})),
    (
        "Application domain",
        frozenset(
            {
                "epilepsy",
                "depression",
                "anesthesia",
                "sleep",
                "aging",
                "dementia",
                "consciousness",
                "heart-rate-variability",
                "treatment-response",
                "seizure-detection",
                "seizure-onset",
                "seizure-onset-zone",
                "seizure-propagation",
                "spike-detection",
                "artifact",
                "qeeg",
            }
        ),
    ),
    (
        "Research paradigm",
        frozenset({"resting-state", "event-related", "erps", "default-mode", "nonstationarity"}),
    ),
    (
        "Connectivity",
        frozenset(
            {
                "directed",
                "undirected",
                "direct-causality",
                "total-causality",
                "phase-synchrony",
                "cross-spectral",
                "conditional-independence",
                "volume-conduction",
                "causality",
                "functional-connectivity",
                "sink-source",
                "aec",
            }
        ),
    ),
    (
        "Method",
        frozenset(
            {
                "entropy",
                "shannon-entropy",
                "kl-divergence",
                "compression",
                "fractal",
                "higuchi",
                "katz",
                "box-counting",
                "mvar",
                "var",
                "fft",
                "ifft",
                "wavelet",
                "hilbert",
                "welch",
                "pearson",
                "spearman",
                "butterworth",
                "stft",
                "page-hinkley",
                "histogram",
                "precision-matrix",
                "phase-randomization",
                "surrogate-significance",
                "autocorrelation-preserving",
                "nonlinearity-test",
                "energy-ratio",
                "entropy-production",
                "prediction-error",
                "orthogonalization",
            }
        ),
    ),
    (
        "Signal representation",
        frozenset(
            {
                "alpha",
                "theta",
                "delta",
                "beta",
                "gamma",
                "beta-gamma",
                "frequency-band",
                "frequency-domain",
                "power-spectrum",
                "absolute-power",
                "relative-power",
                "sub-band",
                "dyadic",
                "scale-adaptive",
                "scalogram",
                "time-frequency",
                "time-domain",
                "instantaneous-frequency",
                "instantaneous-phase",
                "envelope",
            }
        ),
    ),
    (
        "Signal property",
        frozenset(
            {
                "nonlinear",
                "linear",
                "data-driven",
                "nonstationary",
                "stationarity",
                "dynamical-systems",
                "nonlinear-dynamics",
                "time-irreversibility",
                "self-similarity",
                "algorithmic-complexity",
                "predictability",
                "regularity",
                "signal-complexity",
                "variability",
                "standard-deviation",
                "temporal-dependence",
                "temporal-dynamics",
                "lag",
                "binary",
                "state-space",
                "patterns",
                "intrinsic-mode-functions",
                "probability-distribution",
            }
        ),
    ),
    (
        "Operation",
        frozenset(
            {
                "filtering",
                "preprocessing",
                "denoising",
                "segmentation",
                "aggregation",
                "reduction",
                "decomposition",
                "post-processing",
                "outlier-detection",
                "dimensionality-reduction",
                "null-hypothesis",
                "source-localization",
            }
        ),
    ),
]

# Full display order, including the prefix-derived categories.
_CATEGORY_ORDER: list[str] = [name for name, _ in _TAG_CATEGORIES] + [
    "IO shape",
    "Requirement",
    "Other",
]


def _tag_category(tag: str) -> str:
    if tag.startswith("io:"):
        return "IO shape"
    if tag.startswith("req:"):
        return "Requirement"
    for name, members in _TAG_CATEGORIES:
        if tag in members:
            return name
    return "Other"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def gen_domain_pages(by_domain: dict[str, list[tuple[str, type]]]) -> list[str]:
    written = []
    for domain, features in sorted(by_domain.items()):
        title, question = DOMAIN_META.get(domain, (domain.title(), ""))
        lines = [f"# {title}", ""]
        if question:
            lines += [f"*{question}*", ""]
        lines += [
            f"Features in the `cobrabox.{domain}` domain. "
            f"Access them as `cb.<Feature>` (canonical), or as "
            f"`cb.{domain}.<Feature>` / `cb.feature.<Feature>`.",
            "",
            "Each feature has two forms: a **class** for building pipelines, and a "
            "one-shot **function** for a single call.",
            "",
        ]
        for name, cls in features:
            tags = getattr(cls, "_tags", [])
            lines.append(f"### {name}")
            summary = _summary(cls)
            if summary:
                lines.append(summary)
            if has_functional_form(cls):
                lines.append("")
                one_shot = f"cb.{function_name(cls)}(data, ...)"
                composable = f"cb.{name}(...).apply(data)"
                width = max(len(one_shot), len(composable))
                lines.append(
                    "```python\n"
                    f"{one_shot:<{width}}  # one-shot\n"
                    f"{composable:<{width}}  # composable, for pipelines\n"
                    "```"
                )
            if tags:
                tag_links = ", ".join(f"[`{t}`](../tags.md#tag-{_tag_slug(t)})" for t in tags)
                lines.append("")
                lines.append(f"**Tags:** {tag_links}")
            lines.append("")
        _write(DOCS / "domain" / f"{domain}.md", "\n".join(lines).rstrip() + "\n")
        written.append(domain)
    return written


def _call_cell(name: str, fn: str) -> str:
    """How to call the feature, rendered under its name in the explorer table.

    Folded into the Feature cell rather than given its own column: a fifth column
    pushed Tags and Summary out of view, and browsing those matters more.

    Aggregators have no functional form — they fold a splitter's stream — so they
    show only the class.
    """
    if not fn:
        return f'<span class="cbfx-call">cb.{html.escape(name)}()</span>'
    return (
        f'<span class="cbfx-call">cb.{html.escape(fn)}(data)</span>'
        f'<span class="cbfx-call cbfx-alt">cb.{html.escape(name)}().apply(data)</span>'
    )


def _tag_slug(tag: str) -> str:
    return tag.replace(":", "-")


def gen_tags_page(by_domain: dict[str, list[tuple[str, type]]]) -> int:
    """Write a single docs/tags.md: every tag, grouped by category, linking to
    the features that carry it. One page instead of one file per tag.

    Each tag gets an explicit HTML anchor (``tag-<slug>``) so the per-feature
    tag links on the domain pages resolve to the right spot.
    """
    # tag -> sorted list of (feature_name, domain)
    by_tag: dict[str, list[tuple[str, str]]] = {}
    for domain, features in by_domain.items():
        for name, cls in features:
            for tag in getattr(cls, "_tags", []):
                by_tag.setdefault(tag, []).append((name, domain))
    for entries in by_tag.values():
        entries.sort()

    categories: dict[str, list[str]] = {}
    for tag in by_tag:
        categories.setdefault(_tag_category(tag), []).append(tag)

    lines = [
        "# Tags",
        "",
        "Cross-cutting discovery across domain boundaries. Every feature is tagged "
        "by method, modality, application, IO shape and requirements. Each tag below "
        "lists the features that carry it.",
        "",
    ]
    for category in _CATEGORY_ORDER:
        if category not in categories:
            continue
        lines.append(f"## {category}")
        lines.append("")
        for tag in sorted(categories[category]):
            entries = by_tag[tag]
            feats = ", ".join(f"[{name}](domain/{domain}.md)" for name, domain in entries)
            lines.append(f'<a id="tag-{_tag_slug(tag)}"></a>')
            lines.append(f"`{tag}` ({len(entries)}) — {feats}")
            lines.append("")
    _write(DOCS / "tags.md", "\n".join(lines).rstrip() + "\n")
    return len(by_tag)


def gen_api_features_page(by_domain: dict[str, list[tuple[str, type]]]) -> int:
    """Regenerate docs/api/features.md as mkdocstrings directives, one per feature.

    Paths come from each class's real ``__module__``, so they track the code.
    """
    features = sorted(
        ((name, cls) for feats in by_domain.values() for name, cls in feats),
        key=lambda item: item[0],
    )
    lines = [
        "# Features API",
        "",
        "Auto-generated documentation for all feature classes.",
        "",
        "!!! note",
        "    This page is generated by `scripts/gen_feature_docs.py` from the "
        "auto-discovered feature registry. Do not edit by hand.",
        "",
    ]
    for name, cls in features:
        # Address the feature by its public re-export (`cobrabox.signalstats.LineLength`)
        # rather than its implementation module (`..._line_length`), which is private
        # since #116. mkdocstrings resolves re-exported names, and this keeps the
        # private path out of the published docs.
        lines.append(f"::: cobrabox.{_domain_of(cls)}.{name}")
        lines.append("    options:")
        lines.append("        show_root_heading: true")
        lines.append("        show_source: true")
        lines.append("")
    _write(DOCS / "api" / "features.md", "\n".join(lines).rstrip() + "\n")
    return len(features)


# Modality + Clinical are always shown; the rest tuck into one collapsible box,
# in this order (each keeps its own headline inside the box).
_OPEN_CATEGORIES = ["Modality", "Application domain"]
_BOX_CATEGORY_ORDER = [
    "Signal representation",
    "Signal property",
    "Connectivity",
    "Research paradigm",
    "Method",
    "Operation",
    "IO shape",
    "Requirement",
]

_EXPLORER_CSS = """
<style>
#cb-fx{--fx-fg:var(--md-typeset-color,#1f2328);--fx-muted:var(--md-default-fg-color--light,#57606a);
--fx-bg:var(--md-default-bg-color,#fff);--fx-border:var(--md-default-fg-color--lightest,#d0d7de);
--fx-accent:var(--md-accent-fg-color,#7c3aed);--fx-chip:var(--md-code-bg-color,#f3f4f6);
font-size:.8rem;color:var(--fx-fg);margin:1rem 0}
#cb-fx .cbfx-controls{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin-bottom:.6rem}
#cb-fx .cbfx-search{flex:1 1 12rem;min-width:8rem;padding:.35rem .6rem;
border:1px solid var(--fx-border);border-radius:.4rem;
background:var(--fx-bg);color:var(--fx-fg);font-size:.8rem}
#cb-fx .cbfx-mode{color:var(--fx-muted);font-size:.75rem}
#cb-fx .cbfx-mode label{margin-left:.3rem;cursor:pointer}
#cb-fx .cbfx-clear{padding:.35rem .7rem;border:1px solid var(--fx-border);border-radius:.4rem;
background:var(--fx-bg);color:var(--fx-fg);cursor:pointer;font-size:.75rem}
#cb-fx .cbfx-count{color:var(--fx-muted);font-size:.75rem;margin-left:auto}
#cb-fx .cbfx-clabel{font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;
color:var(--fx-muted);font-weight:700;margin:.55rem 0 .3rem}
#cb-fx .cbfx-primary>.cbfx-clabel:first-child{margin-top:.1rem}
#cb-fx .cbfx-primary .cbfx-clabel{color:var(--fx-accent)}
#cb-fx .cbfx-catrow{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.15rem}
#cb-fx .cbfx-more{border:1px solid var(--fx-border);border-radius:.5rem;
margin:.65rem 0 .2rem;background:var(--fx-bg);overflow:hidden}
#cb-fx .cbfx-more>summary{cursor:pointer;padding:.55rem .8rem;display:flex;align-items:center;
gap:.5rem;list-style:none;font-size:.8rem;user-select:none;background:var(--fx-chip)}
#cb-fx .cbfx-more>summary:hover{color:var(--fx-accent)}
#cb-fx .cbfx-more>summary::-webkit-details-marker{display:none}
#cb-fx .cbfx-more>summary::before{content:"\\25B8";color:var(--fx-accent);font-size:.75rem}
#cb-fx .cbfx-more[open]>summary::before{content:"\\25BE"}
#cb-fx .cbfx-more-label{font-weight:700}
#cb-fx .cbfx-more-meta{color:var(--fx-muted);font-size:.72rem}
#cb-fx .cbfx-more-active{color:var(--fx-accent);font-size:.72rem;font-weight:600;margin-left:auto}
#cb-fx .cbfx-more-body{padding:0 .8rem .6rem}
#cb-fx .cbfx-more-body .cbfx-clabel{margin-top:.6rem}
#cb-fx button[data-tag]{font-family:inherit;font-size:.72rem;padding:.15rem .5rem;
border-radius:1rem;border:1px solid var(--fx-border);background:var(--fx-chip);
color:var(--fx-fg);cursor:pointer;line-height:1.4}
#cb-fx button[data-tag]:hover{border-color:var(--fx-accent)}
#cb-fx button[data-tag].is-active{background:var(--fx-accent);
border-color:var(--fx-accent);color:#fff}
#cb-fx button[data-tag] small{opacity:.6;margin-left:.25rem}
#cb-fx .cbfx-wrap{overflow-x:auto;border:1px solid var(--fx-border);border-radius:.5rem}
#cb-fx table{width:100%;border-collapse:collapse;font-size:.78rem;margin:0}
#cb-fx thead th{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--fx-border);
position:sticky;top:0;background:var(--fx-bg);font-weight:600}
#cb-fx tbody td{padding:.45rem .6rem;border-bottom:1px solid var(--fx-border);vertical-align:top}
#cb-fx tbody tr:last-child td{border-bottom:0}
#cb-fx .cbfx-name{font-weight:600;white-space:nowrap}
#cb-fx .cbfx-call{display:block;font-family:var(--md-code-font-family,monospace);
 font-size:.78em;font-weight:400;color:var(--fx-muted);margin-top:.15rem;white-space:nowrap}
#cb-fx .cbfx-call.cbfx-alt{opacity:.6}
#cb-fx .cbfx-dom{color:var(--fx-muted);white-space:nowrap}
#cb-fx .cbfx-rowtags{display:flex;flex-wrap:wrap;gap:.25rem;max-width:24rem}
#cb-fx .cbfx-sum{color:var(--fx-muted);min-width:12rem}
</style>
""".strip()

_EXPLORER_JS = """
<script>
(function(){var root=document.getElementById('cb-fx');if(!root)return;
var rows=[].slice.call(root.querySelectorAll('tbody tr'));
var chips=[].slice.call(root.querySelectorAll('button[data-tag]'));
var search=root.querySelector('.cbfx-search');var count=root.querySelector('.cbfx-count');
var clearBtn=root.querySelector('.cbfx-clear');var active=new Set();
function mode(){var m=root.querySelector('input[name=cbfx-mode]:checked');return m?m.value:'and';}
function apply(){var q=(search.value||'').trim().toLowerCase();var sel=Array.from(active);
var m=mode();var shown=0;
rows.forEach(function(r){var tags=(r.getAttribute('data-tags')||'').split(' ');
var tagOk=sel.length===0?true:(m==='and'
?sel.every(function(t){return tags.indexOf(t)>-1;})
:sel.some(function(t){return tags.indexOf(t)>-1;}));
var sOk=q===''?true:(r.getAttribute('data-search')||'').indexOf(q)>-1;
var show=tagOk&&sOk;r.style.display=show?'':'none';if(show)shown++;});
count.textContent=shown+' of '+rows.length+' features';
chips.forEach(function(c){c.classList.toggle('is-active',active.has(c.getAttribute('data-tag')));});
var more=root.querySelector('.cbfx-more');
if(more){var a=more.querySelectorAll('button[data-tag].is-active').length;
var ma=more.querySelector('.cbfx-more-active');if(ma)ma.textContent=a?a+' selected':'';}}
function toggle(t){if(active.has(t))active.delete(t);else active.add(t);apply();}
chips.forEach(function(c){c.addEventListener('click',function(){toggle(c.getAttribute('data-tag'));});});
search.addEventListener('input',apply);
[].slice.call(root.querySelectorAll('input[name=cbfx-mode]')).forEach(function(i){i.addEventListener('change',apply);});
clearBtn.addEventListener('click',function(){active.clear();search.value='';apply();});
apply();})();
</script>
""".strip()


def build_explorer_html(by_domain: dict[str, list[tuple[str, type]]]) -> str:
    """Self-contained tag-filterable feature table (HTML + scoped CSS + JS).

    All tags are shown, grouped into collapsible category dropdowns (collapsed by
    default). Reads Material CSS variables (with light fallbacks) so it themes
    natively in MkDocs and standalone. No blank lines: kept as one raw HTML block
    so python-markdown passes it through untouched.
    """
    features = sorted(
        ((name, cls, _domain_of(cls)) for feats in by_domain.values() for name, cls in feats),
        key=lambda item: item[0].lower(),
    )
    tag_counts: dict[str, int] = {}
    for _, cls, _ in features:
        for tag in getattr(cls, "_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Group tags by category.
    cats: dict[str, list[str]] = {}
    for tag in tag_counts:
        cats.setdefault(_tag_category(tag), []).append(tag)

    def _cat_block(cat: str) -> str:
        cat_tags = sorted(cats.get(cat, []), key=lambda t: (-tag_counts[t], t))
        if not cat_tags:
            return ""
        btns = "".join(
            f'<button type="button" data-tag="{html.escape(t)}">'
            f"{html.escape(t)}<small>{tag_counts[t]}</small></button>"
            for t in cat_tags
        )
        return (
            f'<div class="cbfx-clabel">{html.escape(cat)}</div>'
            f'<div class="cbfx-catrow">{btns}</div>'
        )

    open_blocks = "".join(_cat_block(c) for c in _OPEN_CATEGORIES)
    box_blocks = "".join(_cat_block(c) for c in _BOX_CATEGORY_ORDER)

    rows = []
    for name, cls, domain in features:
        tags = list(getattr(cls, "_tags", []))
        summary = _summary(cls)
        # The one-shot call is the form most people want; show it in the table and
        # make it searchable, so looking up "line_length" finds LineLength (GH #116).
        fn = function_name(cls) if has_functional_form(cls) else ""
        search = " ".join([name, fn, domain, *tags, summary]).lower()
        rowtags = "".join(
            f'<button type="button" data-tag="{html.escape(t)}">{html.escape(t)}</button>'
            for t in tags
        )
        rows.append(
            f'<tr data-tags="{html.escape(" ".join(tags))}" data-search="{html.escape(search)}">'
            f'<td class="cbfx-name">{html.escape(name)}'
            f"{_call_cell(name, fn)}</td>"
            f'<td class="cbfx-dom">{html.escape(domain)}</td>'
            f'<td><div class="cbfx-rowtags">{rowtags}</div></td>'
            f'<td class="cbfx-sum">{html.escape(summary)}</td></tr>'
        )

    parts = [
        _EXPLORER_CSS,
        '<div id="cb-fx">',
        '<div class="cbfx-controls">',
        '<input type="text" class="cbfx-search" placeholder="Search features, tags, summaries…">',
        '<span class="cbfx-mode">Match: '
        '<label><input type="radio" name="cbfx-mode" value="and" checked> all</label> '
        '<label><input type="radio" name="cbfx-mode" value="or"> any</label></span>',
        '<button type="button" class="cbfx-clear">Clear</button>',
        '<span class="cbfx-count"></span>',
        "</div>",
        '<div class="cbfx-primary">' + open_blocks + "</div>",
        '<details class="cbfx-more"><summary>'
        '<span class="cbfx-more-label">More filters</span>'
        '<span class="cbfx-more-meta">connectivity &middot; output &middot; method '
        "&middot; signal type &amp; more</span>"
        '<span class="cbfx-more-active"></span></summary>'
        '<div class="cbfx-more-body">' + box_blocks + "</div></details>",
        '<div class="cbfx-wrap"><table><thead><tr>'
        "<th>Feature</th><th>Domain</th><th>Tags</th><th>Summary</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
        "</div>",
        _EXPLORER_JS,
    ]
    return "\n".join(parts)


def inject_explorer(html_block: str) -> None:
    """Replace the marked region of docs/index.md with the explorer widget."""
    start, end = "<!-- feature-explorer:start -->", "<!-- feature-explorer:end -->"
    index = DOCS / "index.md"
    text = index.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise RuntimeError(f"feature-explorer markers not found in {index}")
    head = text[: text.index(start) + len(start)]
    tail = text[text.index(end) :]
    index.write_text(f"{head}\n{html_block}\n{tail}", encoding="utf-8")


def main() -> None:
    by_domain = _collect()
    domains = gen_domain_pages(by_domain)
    n_tags = gen_tags_page(by_domain)
    n_api = gen_api_features_page(by_domain)
    inject_explorer(build_explorer_html(by_domain))
    print(
        f"Wrote {len(domains)} domain pages, tags.md ({n_tags} tags), "
        f"and api/features.md ({n_api} features)."
    )


if __name__ == "__main__":
    main()
