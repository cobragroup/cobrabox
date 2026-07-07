"""Generate domain and tag documentation pages from the live feature registry.

Run with:  uv run python scripts/gen_feature_docs.py

Produces, under ``docs/``:
    domain/<domain>.md   one page per domain package, listing its features
    tags/index.md        a tag index grouped by category, with counts
    tags/<tag>.md        one page per tag, listing every feature carrying it

Everything is derived from the auto-discovered features (their module path,
docstring summary and ``_tags``), so the pages cannot drift out of sync with the
code. The feature catalog restructure proposal (§8) calls for cross-cutting tag
pages; this is their generator.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import cobrabox as cb

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
        by_domain.setdefault(_domain_of(cls), []).append((name, cls))
    for features in by_domain.values():
        features.sort(key=lambda item: item[0])
    return by_domain


def _tag_category(tag: str) -> str:
    if tag.startswith("io:"):
        return "IO shape"
    if tag.startswith("req:"):
        return "Requirement"
    return "Descriptor"


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
            f"Access them as `cb.{domain}.<Feature>` or `cb.feature.<Feature>`.",
            "",
        ]
        for name, cls in features:
            tags = getattr(cls, "_tags", [])
            lines.append(f"### {name}")
            summary = _summary(cls)
            if summary:
                lines.append(summary)
            if tags:
                tag_links = ", ".join(f"[`{t}`](../tags/{_tag_slug(t)}.md)" for t in tags)
                lines.append("")
                lines.append(f"**Tags:** {tag_links}")
            lines.append("")
        _write(DOCS / "domain" / f"{domain}.md", "\n".join(lines).rstrip() + "\n")
        written.append(domain)
    return written


def _tag_slug(tag: str) -> str:
    return tag.replace(":", "-")


def gen_tag_pages(by_domain: dict[str, list[tuple[str, type]]]) -> list[str]:
    # tag -> list of (feature_name, domain)
    by_tag: dict[str, list[tuple[str, str]]] = {}
    for domain, features in by_domain.items():
        for name, cls in features:
            for tag in getattr(cls, "_tags", []):
                by_tag.setdefault(tag, []).append((name, domain))

    for entries in by_tag.values():
        entries.sort()

    # Per-tag pages
    for tag, entries in by_tag.items():
        lines = [f"# Tag: `{tag}`", ""]
        lines.append(f"Category: **{_tag_category(tag)}**")
        lines.append("")
        lines.append(f"{len(entries)} feature(s) carry this tag:")
        lines.append("")
        for name, domain in entries:
            lines.append(f"- **{name}** — [`cobrabox.{domain}`](../domain/{domain}.md)")
        lines.append("")
        _write(DOCS / "tags" / f"{_tag_slug(tag)}.md", "\n".join(lines).rstrip() + "\n")

    # Index page grouped by category
    categories: dict[str, list[str]] = {}
    for tag in by_tag:
        categories.setdefault(_tag_category(tag), []).append(tag)

    lines = [
        "# Tags",
        "",
        "Cross-cutting discovery across domain boundaries. Every feature is tagged "
        "by method, modality, application, IO shape and requirements; each tag below "
        "links to the features that carry it.",
        "",
    ]
    for category in sorted(categories):
        lines.append(f"## {category}")
        lines.append("")
        for tag in sorted(categories[category]):
            count = len(by_tag[tag])
            lines.append(f"- [`{tag}`]({_tag_slug(tag)}.md) ({count})")
        lines.append("")
    _write(DOCS / "tags" / "index.md", "\n".join(lines).rstrip() + "\n")
    return sorted(by_tag)


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
        lines.append(f"::: {cls.__module__}.{name}")
        lines.append("    options:")
        lines.append("        show_root_heading: true")
        lines.append("        show_source: true")
        lines.append("")
    _write(DOCS / "api" / "features.md", "\n".join(lines).rstrip() + "\n")
    return len(features)


def main() -> None:
    by_domain = _collect()
    domains = gen_domain_pages(by_domain)
    tags = gen_tag_pages(by_domain)
    n_api = gen_api_features_page(by_domain)
    print(
        f"Wrote {len(domains)} domain pages, {len(tags)} tag pages + index, "
        f"and api/features.md ({n_api} features)."
    )


if __name__ == "__main__":
    main()
