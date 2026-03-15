# Installation

## Prerequisites

- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`
- `git-lfs` (for accessing dummy datasets)

## Quick Install

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/cobrabox.git
cd cobrabox

# Install git-lfs and pull large files
git lfs install
git lfs pull

# Install dependencies and set up the environment
uv sync
```

### Using pip

```bash
pip install -e .
```

## Development Setup

For development, install with dev dependencies:

```bash
uv sync --group dev
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Verify Installation

Run the test suite:

```bash
uv run pytest -q
```

Try a quick example:

```bash
uv run python examples/feature_pipeline_demo.py
```

## Optional: Documentation Dependencies

To build the documentation locally:

```bash
uv sync --group docs
```

Then serve the docs:

```bash
uv run mkdocs serve
```

## Troubleshooting

### git-lfs issues

If you see placeholder files instead of actual data:

```bash
git lfs install
git lfs pull
```

### Python version

Check your Python version:

```bash
python --version
```

CobraBox requires Python 3.11+.
