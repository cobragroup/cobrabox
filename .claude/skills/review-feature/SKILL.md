---
name: review-feature
description: This skill should be used when the user asks to "review this feature", "check feature quality", "audit a feature file", "I just wrote a feature check it", or wants to verify a feature in src/cobrabox/features/ meets code quality standards before merging.
---

# Review Feature

Review a single feature file in `src/cobrabox/features/` for code quality and produce
an actionable report for a coding agent or developer to implement.

## Usage

Invoke with the feature file path as the argument:

```text
/review-feature src/cobrabox/features/my_feature.py
```

## Procedure

Follow these steps in order:

### 1. Read the feature file

Read `$ARGUMENTS` in full.

**Special case**: If the file is `src/cobrabox/features/dummy.py`, skip the review and report:
```text
Review skipped — dummy.py is an intentional negative reference (not to be fixed)
```

### 2. Run ruff

Run both commands and capture full output:

```bash
uvx ruff check $ARGUMENTS
uvx ruff format --check $ARGUMENTS
```

### 3. Load criteria

Load `references/criteria.md` from this skill's directory. Use it to evaluate the
feature file against all four criterion categories.

### 4. Write the review file

Determine the feature name from the filename (e.g. `my_feature.py` → `my_feature`).
Create `docs/agent-reviews/` if it does not exist. Write the review to
`docs/agent-reviews/<feature-name>.md` using the output format below.

### 5. Report to conversation

After writing the file, print a single line to the conversation:

```text
Review written to docs/agent-reviews/<feature-name>.md — <PASS|NEEDS WORK> (<N> issues)
```

---

## Output format

```markdown
# Feature Review: <feature_name>

**File**: `src/cobrabox/features/<feature>.py`
**Date**: YYYY-MM-DD
**Verdict**: PASS | NEEDS WORK

## Summary

One paragraph. State overall quality, highlight the most important issues or confirm
the feature is clean. Written for a coding agent that will act on this review.

## Ruff

### `uvx ruff check`
<raw output, or "Clean — no issues found.">

### `uvx ruff format --check`
<raw output, or "Clean — no formatting issues.">

## Signature & Structure
<Narrative. Reference line numbers for any issues.>

## Docstring
<Narrative. Note missing or incomplete sections.>

## Typing
<Narrative. Flag missing annotations or bare Any.>

## Safety & Style
<Narrative. Call out print statements, missing validation, mutation of input.>

## Action List

1. [Severity: HIGH/MEDIUM/LOW] Description of what to fix and where (line N).
2. ...
```

Severity guide:

- **HIGH** — missing type annotations, missing docstring, ruff errors, `print` statements
- **MEDIUM** — incomplete docstring sections, missing input validation
- **LOW** — style suggestions, minor improvements

If the feature passes all criteria and ruff is clean, write `## Action List\n\nNone.`
and set verdict to `PASS`.
