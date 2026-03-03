# Setup Guide

This guide walks you through setting up your development environment for CobraBox.

## Prerequisites

1. **Python 3.11+** - Check with `python --version`
2. **git** - Version control
3. **git-lfs** - Large file storage (for dummy datasets)
4. **uv** - Package manager (recommended)

## Step 1: Install git-lfs

### macOS

```bash
brew install git-lfs
git lfs install
```

### Linux

```bash
apt-get install git-lfs
git lfs install
```

### Windows

```bash
winget install git-lfs
git lfs install
```

## Step 2: Clone the Repository

```bash
git clone git@github.com:yourusername/cobrabox.git
cd cobrabox
```

## Step 3: Set Up Python Environment

```bash
# Install dependencies
uv sync

# Or with dev dependencies
uv sync --group dev
```

## Step 4: Install Pre-commit Hooks

```bash
uvx pre-commit install
```

Pre-commit hooks will automatically run ruff linting on every commit.

## Step 5: Verify Setup

Run the test suite:

```bash
uv run pytest -q
```

Try the demo:

```bash
uv run python examples/feature_pipeline_demo.py
```

## Development Workflow

1. **Create a branch** for your feature
2. **Write tests** for new functionality
3. **Run linting** - `uvx ruff check`
4. **Run tests** - `uv run pytest`
5. **Commit** - pre-commit hooks run automatically
6. **Push** and create a pull request

## Common Issues

### git-lfs not working

If you see placeholder files:

```bash
git lfs pull
```

### Pre-commit fails

Fix the reported issues and stage the changes again:

```bash
# Fix issues, then:
git add <files>
git commit
```

### Tests fail

Check the error output and fix the failing tests. Run with more detail:

```bash
uv run pytest -v
```
