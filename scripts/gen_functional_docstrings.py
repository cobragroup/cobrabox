#!/usr/bin/env python3
"""Sync functional-wrapper docstrings from the class docstring (GH #135).

The class docstring is the single source of truth.  This script regenerates
every ``@functional``-decorated function's docstring by:

1. Copying the class docstring verbatim.
2. Inserting a ``data:`` entry as the first item in the ``Args:`` section.
3. Rewriting ``cb.Class(args).apply(x)`` examples to ``cb.func(x, args)``.

Only the docstring is replaced — the function's signature, body, imports and
all surrounding code are left untouched.

Exit codes:
  0 — all docstrings are already in sync
  1 — one or more files were updated (pre-commit will re-stage them)
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
import typing
from pathlib import Path

from gen_functional_wrappers import _rewrite_calls

import cobrabox as cb
from cobrabox._functional import function_name, has_functional_form

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "cobrabox"

# ---------------------------------------------------------------------------
# Data-argument text, chosen by generic bound
# ---------------------------------------------------------------------------

_DATA_ARG_SIGNAL = (
    "data: The input time-series signal to process, as a\n"
    "        :class:`~cobrabox.SignalData` (or any :class:`~cobrabox.Data`\n"
    "        carrying a ``time`` dimension)."
)

_DATA_ARG_GENERIC = "data: The input data to process, as a :class:`~cobrabox.Data`."


def _data_bound(cls: type) -> str:
    for base in getattr(cls, "__orig_bases__", ()):
        args = typing.get_args(base)
        if args:
            return args[0].__name__
    return "Data"


def _data_arg_text(cls: type) -> str:
    return _DATA_ARG_SIGNAL if _data_bound(cls) == "SignalData" else _DATA_ARG_GENERIC


# ---------------------------------------------------------------------------
# Docstring transformation
# ---------------------------------------------------------------------------

_ARGS_RE = re.compile(r"^(Args:\s*\n)", re.MULTILINE)
_ARGS_NONE_RE = re.compile(r"^(Args:\s*\n)[^\S\n]*None[^\S\n]*$", re.MULTILINE)


def _insert_data_arg(doc: str, data_text: str) -> str:
    """Insert ``data:`` as the first entry in the ``Args:`` section."""
    # Case 1: Args:\n    None  →  replace None with the data arg
    m = _ARGS_NONE_RE.search(doc)
    if m:
        return doc[: m.start()] + "Args:\n    " + data_text + doc[m.end() :]

    # Case 2: Args:\n    <field>:  →  prepend data arg before the first field
    m = _ARGS_RE.search(doc)
    if m:
        return doc[: m.end()] + "    " + data_text + "\n" + doc[m.end() :]

    # Case 3: No Args section at all  →  insert one after the description block
    # Find the first section header (Returns:, Raises:, Example:, Note:)
    section = re.search(r"^(?:Returns|Raises|Example|Note):", doc, re.MULTILINE)
    if section:
        insert_at = section.start()
        return doc[:insert_at] + "Args:\n    " + data_text + "\n\n" + doc[insert_at:]

    # Fallback: append at the end
    return doc.rstrip() + "\n\nArgs:\n    " + data_text


def _func_docstring(cls: type, func_name: str) -> str:
    """Build the function docstring from the class docstring."""
    doc = inspect.getdoc(cls) or f"Apply :class:`~cobrabox.{cls.__name__}` to ``data``."
    doc = _rewrite_calls(doc, cls.__name__, func_name)
    return _insert_data_arg(doc, _data_arg_text(cls))


# ---------------------------------------------------------------------------
# AST-based docstring replacement
# ---------------------------------------------------------------------------


def _find_functional_func(source: str) -> ast.FunctionDef | None:
    """Find the ``@functional(...)`` decorated function in ``source``."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Name) and target.id == "functional":
                return node
    return None


def _extract_docstring_range(source: str, func_node: ast.FunctionDef) -> tuple[int, int] | None:
    """Return (start_offset, end_offset) of the function's docstring in ``source``.

    Covers from the opening ``\"\"\"`` to the closing ``\"\"\"`` inclusive.
    """
    if not func_node.body:
        return None
    first_stmt = func_node.body[0]
    if not isinstance(first_stmt, ast.Expr) or not isinstance(first_stmt.value, ast.Constant):
        return None
    if not isinstance(first_stmt.value.value, str):
        return None

    # ast gives 1-based line numbers; convert to 0-based offsets.
    lines = source.splitlines(keepends=True)
    start_line = first_stmt.lineno - 1  # 0-based
    end_line = first_stmt.end_lineno - 1  # type: ignore[operator]

    # Find the opening triple-quote on start_line
    line = lines[start_line]
    col = line.find('"""')
    if col < 0:
        col = line.find("'''")
    if col < 0:
        return None
    start_offset = sum(len(lines[i]) for i in range(start_line)) + col

    # Find the closing triple-quote on end_line
    end_line_text = lines[end_line]
    # The closing triple-quote is the LAST occurrence on end_line (could share
    # the line with the opening triple-quote for one-liners).
    if start_line == end_line:
        # One-liner: """..."""  — find the second triple-quote
        second = end_line_text.find('"""', col + 3)
        if second < 0:
            second = end_line_text.find("'''", col + 3)
        close_col = second
    else:
        close_col = end_line_text.find('"""')
        if close_col < 0:
            close_col = end_line_text.find("'''")
    if close_col < 0:
        return None
    end_offset = sum(len(lines[i]) for i in range(end_line)) + close_col + 3

    return start_offset, end_offset


def _format_docstring(doc: str, indent: str = "    ") -> str:
    """Wrap ``doc`` in triple quotes with proper indentation.

    The first line has NO indent prefix because the caller inserts
    the formatted text at the position where the ``\"\"\"`` already
    sits — the leading whitespace is part of ``source[:start]``.
    """
    lines = doc.split("\n")
    if len(lines) == 1:
        return f'"""{lines[0]}"""'
    body = "\n".join((f"{indent}{ln}").rstrip() if ln.strip() else "" for ln in lines[1:])
    return f'"""{lines[0]}\n{body}\n{indent}"""'


def _replace_docstring(source: str, func_node: ast.FunctionDef, new_doc: str) -> str:
    """Replace the function's docstring in ``source`` with ``new_doc``."""
    rng = _extract_docstring_range(source, func_node)
    if rng is None:
        return source
    start, end = rng

    # Detect indentation from the function body
    lines = source.splitlines(keepends=True)
    body_line = lines[func_node.body[0].lineno - 1]
    indent = body_line[: len(body_line) - len(body_line.lstrip())]

    formatted = _format_docstring(new_doc, indent)
    return source[:start] + formatted + source[end:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_if_changed(path: Path, content: str) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if content == current:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    updated: list[str] = []

    for name in sorted(cb.feature.__all__):
        obj = getattr(cb.feature, name)
        if not isinstance(obj, type) or not has_functional_form(obj):
            continue

        cls = obj
        func_name = function_name(cls)
        path = Path(inspect.getsourcefile(cls))  # type: ignore[arg-type]
        source = path.read_text(encoding="utf-8")

        func_node = _find_functional_func(source)
        if func_node is None:
            continue

        new_doc = _func_docstring(cls, func_name)
        new_source = _replace_docstring(source, func_node, new_doc)

        if _write_if_changed(path, new_source):
            updated.append(func_name)
            print(f"  Updated {path.name}::{func_name}", file=sys.stderr)

    if updated:
        print(f"Synced {len(updated)} docstring(s).", file=sys.stderr)
    return 1 if updated else 0


if __name__ == "__main__":
    sys.exit(main())
