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

### 1. Read inputs

- Read the feature file at `$ARGUMENTS`
- Derive the expected test path: `tests/test_feature_<feature_name>.py`

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

### 4. Report to conversation

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

```text
```

Severity guide:

- **HIGH** — missing required scenario, no docstring, missing metadata preservation test
- **MEDIUM** — incomplete assertion, wrong naming convention, missing `-> None`
- **LOW** — style, minor improvements

If all required scenarios are covered and criteria pass, write `## Action List\n\nNone.`
and set verdict to `PASS`.
