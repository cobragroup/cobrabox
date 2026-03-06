---
name: review-feature-tests
description: Use when the user wants to review, audit, generate, or check tests for a feature in src/cobrabox/features/. Triggers on "review tests for feature X", "check if feature X has tests", "write tests for X", "generate tests for X", "audit test coverage for feature", "does X have good tests".
---

# Review Feature Tests

Review or plan pytest tests for a single cobrabox feature. Produces an actionable
plan document for a coding agent or developer to implement.

## Usage

```text
/review-feature-tests src/cobrabox/features/my_feature.py
```

## Procedure

### 1. Read inputs and check coverage

- Read the feature file at `$ARGUMENTS`
- Derive the expected test path: `tests/test_feature_<feature_name>.py`
- **Check per-file coverage**: Run `uv run pytest tests/test_feature_<feature_name>.py --cov=src/cobrabox/features/<feature_name>.py --cov-report=term-missing` and note the coverage percentage. If <95%, flag as HIGH severity issue.

### 2. Branch on test file existence

#### If test file does not exist — Scratch mode

Load `references/test-patterns.md`. Using the feature's signature, parameters, and
behaviour, plan a complete test file covering all required scenarios.

#### If test file exists — Gap-fill mode

Read the test file. Load `references/criteria.md` and evaluate against all criteria.
Load `references/test-patterns.md` for any missing scenarios that need planned code.

### 3. Write the plan

Derive feature name from filename (e.g. `my_feature.py` → `my_feature`).
Create `docs/agent-reviews/` if it does not exist.
Write plan to `docs/agent-reviews/<feature-name>-tests.md` using the output format below.

### 4. Validate review file with markdownlint

Run markdownlint on the generated review file to ensure it has no markdown formatting issues:

```bash
markdownlint docs/agent-reviews/<feature-name>-tests.md
```

If markdownlint reports any errors, fix them in the review file before proceeding.

Common issues to fix:

- Missing blank lines around headings (MD022)
- Missing blank lines around lists (MD032)
- Missing blank lines around fenced code blocks (MD031)
- Missing language specifier on code blocks (MD040)

**Important exception for MD050 (strong-style)**: Do NOT convert `__` to `**` when it appears inside code blocks or inline code. Patterns like `__future__`, `__call__`, `__init__`, etc. are Python dunder names and must remain as `__name__`. Only fix bold text that uses `__text__` outside of code.

### 5. Report to conversation

Print a single line:

```text
Review written to docs/agent-reviews/<feature-name>-tests.md — <MISSING|NEEDS WORK|PASS> (<N> issues)
```

---

## Output format

### Scratch mode

```markdown
# Test Plan: <feature_name>

**Feature**: `src/cobrabox/features/<feature>.py`
**Test file**: `tests/test_feature_<feature_name>.py` (does not exist)
**Date**: YYYY-MM-DD
**Verdict**: MISSING

## Summary

Brief description of what the feature does and what the tests will verify.

## Proposed test file

\`\`\`python
<complete proposed test file>
\`\`\`
```

### Gap-fill mode

```markdown
# Test Review: <feature_name>

**Feature**: `src/cobrabox/features/<feature>.py`
**Test file**: `tests/test_feature_<feature_name>.py`
**Date**: YYYY-MM-DD
**Verdict**: PASS | NEEDS WORK

## Coverage

Report the per-file coverage percentage and any uncovered lines:

```text
SlidingWindowReduce: 100% (34 statements, 0 missing)
```

If coverage is <95%, list the missing lines and flag as HIGH severity in Action List.

## Summary

One paragraph on overall test quality and key gaps.

## Keep

Tests that are correct and complete — no changes needed:

- `test_<name>` — reason it passes

## Fix

Tests that exist but need changes:

### `test_<name>`

Issue: <what's wrong>

```python
# corrected version
```

## Add

Missing scenarios — new tests to add:

### `test_<feature>_<scenario>`

```python
# new test
```

## Action List

1. [Severity: HIGH/MEDIUM/LOW] What to do (file, line if applicable)
2. ...

Severity guide:

- **HIGH** — missing required scenario, no docstring, missing metadata preservation test, **coverage <95%**
- **MEDIUM** — incomplete assertion, wrong naming convention, missing `-> None`
- **LOW** — style, minor improvements

If all required scenarios are covered and criteria pass, write `## Action List\n\nNone.`
and set verdict to `PASS`.
