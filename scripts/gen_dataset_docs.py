"""Generate dataset documentation tables from the live dataset registry.

Run with:  uv run python scripts/gen_dataset_docs.py

Produces, under ``docs/``:
    guide/datasets.md   dataset table (marked region)
    index.md            compact dataset summary table on the front page

Everything is derived from ``cobrabox.downloader.REMOTE_DATASETS``, so the tables
cannot drift out of sync with the code (e.g. a new dataset added to the registry
shows up here automatically).
"""

from __future__ import annotations

from pathlib import Path

from cobrabox.downloader import REMOTE_DATASETS, RemoteDatasetSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"


def _replace_marked_region(path: Path, start: str, end: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise RuntimeError(f"markers {start!r}/{end!r} not found in {path}")
    head = text[: text.index(start) + len(start)]
    tail = text[text.index(end) :]
    path.write_text(f"{head}\n{body}\n{tail}", encoding="utf-8")


def _subset_count(spec: RemoteDatasetSpec) -> str:
    keys = spec.subset_keys()
    if not keys:
        return "—"
    name = spec.subset_key_name or "subsets"
    if "(" in name and ")" in name:
        name = name.split("(", 1)[1].rstrip(")")
    return f"{len(keys)} {name}"


def gen_remote_table() -> int:
    lines = [
        "| Identifier | Description | Subsets | Size | License |",
        "| ---------- | ----------- | ------- | ---- | ------- |",
    ]
    for ident in sorted(REMOTE_DATASETS):
        spec = REMOTE_DATASETS[ident]
        desc = spec.description or "—"
        if spec.info_url:
            desc = f"[{desc}]({spec.info_url})"
        subsets = _subset_count(spec)
        size = spec.size_hint or "—"
        lic = spec.license or "—"
        lines.append(f"| `{ident}` | {desc} | {subsets} | {size} | {lic} |")
    _replace_marked_region(
        DOCS / "guide" / "datasets.md",
        "<!-- remote-dataset-table:start -->",
        "<!-- remote-dataset-table:end -->",
        "\n".join(lines),
    )
    return len(REMOTE_DATASETS)


_SUMMARY_MAX_LEN = 140


def _short_summary(text: str) -> str:
    """Truncate a description so the front-page table stays scannable."""
    text = text.replace("|", "/")
    if len(text) <= _SUMMARY_MAX_LEN:
        return text
    cut = text[:_SUMMARY_MAX_LEN].rsplit(" ", 1)[0]
    return cut.rstrip(",;: ") + "…"


def gen_index_summary() -> None:
    lines = ["| Identifier | Summary | Size |", "| ---------- | ------- | ---- |"]
    for ident in sorted(REMOTE_DATASETS):
        spec = REMOTE_DATASETS[ident]
        desc = _short_summary(spec.description or "—")
        size = spec.size_hint or "—"
        lines.append(f"| `{ident}` | {desc} | {size} |")
    _replace_marked_region(
        DOCS / "index.md",
        "<!-- dataset-table:start -->",
        "<!-- dataset-table:end -->",
        "\n".join(lines),
    )


def main() -> None:
    n_datasets = gen_remote_table()
    gen_index_summary()
    print(f"Wrote guide/datasets.md ({n_datasets} datasets) and index.md summary.")


if __name__ == "__main__":
    main()
