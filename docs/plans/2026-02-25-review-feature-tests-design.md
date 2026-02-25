# Design: `review-feature-tests` skill

**Date**: 2026-02-25

## Purpose

A Claude Code skill that reviews and/or plans tests for a single feature in
`src/cobrabox/features/`. Two modes depending on whether a test file exists:

- **Scratch mode**: no test file found → plan a complete test file from scratch
- **Gap-fill mode**: test file exists → review it, identify gaps, plan additions

Output is always a plan document for a coding agent to implement.

## Location

```text
.claude/skills/review-feature-tests/
├── SKILL.md
└── references/
    ├── criteria.md      ← what to check when reviewing existing tests
    └── test-patterns.md ← concrete pytest snippets for each required scenario
```

## Trigger

Invoked manually via `/review-feature-tests <path>` or auto-triggered by phrases such as
"review tests for this feature", "check if feature X has tests", "write tests for X",
"generate tests for X", "audit test coverage for feature".

## Procedure

1. Accept feature file path as `$ARGUMENTS`
2. Derive test path: `tests/test_<feature_name>.py`
3. Read the feature file
4. Branch on test file existence:
   - **No test file** → load `references/test-patterns.md`, plan full file from scratch
   - **Test file exists** → read it, load `references/criteria.md`, identify gaps;
     load `references/test-patterns.md` for any gaps needing new tests planned
5. Write plan to `docs/agent-reviews/<feature-name>-tests.md`
6. Print one-line summary: `MISSING` / `NEEDS WORK` / `PASS` + issue count

## Output format

### Scratch mode

- Header + verdict (`MISSING`)
- Summary of feature's behaviour that informs what to test
- Full proposed test file as a fenced code block, ready to copy

### Gap-fill mode

- Header + verdict (`NEEDS WORK` or `PASS`)
- Summary of what exists and overall quality
- **Keep** section: tests that are fine as-is
- **Fix** section: tests needing changes (with corrected code)
- **Add** section: missing scenarios (with new test code)
- Action list ordered by severity

## Test criteria (detail in `references/criteria.md`)

- File naming: `test_<feature_name>.py`
- Each function has a docstring and `-> None` annotation
- Naming: `test_<feature>_<scenario>`
- Required scenarios: happy path, history updated, metadata preserved,
  invalid dims, invalid params (if any), output type is `Data`, no mutation

## Test patterns (detail in `references/test-patterns.md`)

Concrete pytest snippets for each required scenario using `line_length` and
`sliding_window` as examples, ready to adapt by substituting the target feature name.
