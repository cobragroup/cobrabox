# Making a Pull Request

This guide walks you through creating a pull request (PR) on GitHub.

## Prerequisites

- A feature branch with your changes
- Tests passing locally
- Code formatted with ruff

## Step 1: Push Your Branch

```bash
git push -u origin <your-branch-name>
```

Example:

```bash
git push -u origin feature/add-line-length
```

## Step 2: Open GitHub

1. Navigate to the repository on GitHub
2. You should see a banner suggesting to create a PR for your recently pushed branch
3. Click **"Compare & pull request"**

Alternatively:

1. Go to the **Pull requests** tab
2. Click **"New pull request"**
3. Select:
   - **base**: `main`
   - **compare**: your feature branch

## Step 3: Fill in PR Details

### Title

Use a clear, descriptive title:

```text
feat: add line_length feature for time-series complexity
```

### Description

Include:

- **What** changed (brief summary)
- **Why** the change was made (motivation)
- **How** you tested it (test coverage)

Example:

```markdown
## Summary

Adds the `line_length` feature function that computes the sum of absolute
differences between consecutive timepoints.

## Changes

- New feature: `cobrabox.features.line_length`
- Tests: `tests/test_feature_line_length.py`

## Testing

- Unit tests for basic functionality
- Tests for edge cases (single timepoint, NaN handling)
- Coverage: 95%+
```

## Step 4: Create the PR

Click **"Create pull request"**

## Step 5: Request Review

Add reviewers from the team (if applicable).

## What Happens Next

1. **Automated checks** - CI runs tests and linting
2. **Code review** - Team members review your code
3. **Feedback** - Address any comments or requested changes
4. **Approval** - Once approved, the PR can be merged
5. **Merge** - PR is merged into `main`

## Responding to Review Feedback

If reviewers request changes:

1. Make the changes in your local branch
2. Commit and push:

```bash
git add <files>
git commit -m "address review comments"
git push
```

Your PR automatically updates with the new commits.

## PR Checklist

Before submitting:

- [ ] Tests pass locally (`uv run pytest`)
- [ ] Code is formatted (`uvx ruff format`)
- [ ] No linting errors (`uvx ruff check`)
- [ ] Docstrings are complete
- [ ] Examples work as documented

## Common PR Mistakes

- **Large PRs** - Keep PRs focused and small (< 400 lines ideal)
- **Missing tests** - Always include tests for new features
- **Unclear descriptions** - Explain what and why, not just what
- **Skipping local testing** - Run tests before pushing

## Resources

- [GitHub: Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
- [GitHub: About pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
