# Design: `review-feature` skill

**Date**: 2026-02-25

## Purpose

A Claude Code skill that reviews a single feature file in `src/cobrabox/features/` for
code quality. Primary use: audit existing features. Secondary use: writing guide when
authoring a new feature.

## Location

```text
.claude/skills/review-feature/
├── SKILL.md
└── references/
    └── criteria.md
```

## Trigger

Invoked manually via `/review-feature <path>` or auto-triggered by phrases such as
"review this feature", "check feature quality", "audit feature file", "I just wrote a
feature, check it".

## Review procedure

1. Accept feature file path as `$ARGUMENTS`
2. Read the file
3. Run `uvx ruff check <file>` and `uvx ruff format --check <file>`, capture output
4. Load `references/criteria.md` and check file against each criterion
5. Write review to `docs/agent-reviews/<feature-name>.md` (create dir if needed)
6. Print one-line summary to conversation: verdict + issue count

## Review output format (`docs/agent-reviews/<feature-name>.md`)

- **Header**: feature name, file path, date
- **Verdict**: `PASS` or `NEEDS WORK` + one-sentence summary
- **Ruff findings**: raw output from both ruff commands, or "Clean" if none
- **Criteria findings**: one section per category, narrative with line references
- **Action list**: numbered, concrete, ordered by severity

## Criteria categories (detail in `references/criteria.md`)

1. **Signature & structure** — `@feature` decorator, `data: Data` first param, return
   type annotation, `from __future__ import annotations`, no unused imports
2. **Docstring** — Google style; required sections: one-liner, Args, Returns, Example
3. **Typing** — all params typed, return type matches actual return, no bare `Any`
4. **Safety & style** — no `print()`, `ValueError` on bad input, no mutation of `data`

## Reference examples

- `line_length.py` — positive reference (compliant)
- `dummy.py` — negative reference (missing docstring sections, has `print`, no validation)
