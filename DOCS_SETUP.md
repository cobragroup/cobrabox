# Documentation Setup Summary

## What Was Created

A complete MkDocs-based documentation system for CobraBox with:

### 1. Configuration

- **`mkdocs.yml`** - MkDocs configuration with:
  - Material theme (dark/light mode)
  - mkdocstrings for auto-generated API docs
  - mkdocs-jupyter for notebook support
  - Search, code highlighting, and navigation features

### 2. Documentation Structure

```
docs/
├── index.md                 # Home page
├── installation.md          # Installation guide
├── quickstart.md            # Quick start guide
├── guide/
│   ├── core-concepts.md    # Immutability, metadata, history
│   ├── data-containers.md  # Data, EEG, FMRI classes
│   ├── features.md         # Using and creating features
│   ├── datasets.md         # Working with datasets
│   └── pipelines.md        # Building pipelines
├── api/
│   ├── index.md            # API reference overview
│   ├── data.md             # Data class API (auto-generated)
│   ├── features.md         # Features API (auto-generated)
│   └── datasets.md         # Datasets API (auto-generated)
└── contributing/
    ├── setup.md            # Development setup
    ├── features.md         # Contributing features
    └── pr.md               # Making pull requests
```

### 3. Dependencies Added to `pyproject.toml`

```toml
[dependency-groups]
docs = [
    "mkdocs-material>=9.6.0",
    "mkdocstrings[python]>=0.28.0",
    "mkdocs-jupyter>=0.25.0",
]
```

### 4. Makefile Commands

```bash
make docs-serve    # Start dev server with auto-reload
make docs-build    # Build static site
make docs-deploy   # Deploy to GitHub Pages
```

## Usage

### Install Dependencies

```bash
uv sync --group docs
```

### Serve Locally (Development)

```bash
uv run mkdocs serve
# or
make docs-serve
```

Then open <http://127.0.0.1:8000> in your browser.

The server auto-reloads when you edit markdown files.

### Build Static Site

```bash
uv run mkdocs build
# or
make docs-build
```

Output goes to `site/` directory.

### Deploy to GitHub Pages

```bash
uv run mkdocs gh-deploy --force
# or
make docs-deploy
```

## Key Features

### Auto-Generated API Docs

API documentation is automatically extracted from docstrings using mkdocstrings.

**Example:** In `docs/api/data.md`:

```markdown
::: cobrabox.data.Data
    options:
        show_root_heading: true
        show_source: true
        show_signature_annotations: true
        merge_init_into_class: true
```

This generates full API documentation for the `Data` class including:

- Class docstring
- All methods with signatures
- Type hints
- Source code links

### Google-Style Docstrings

The system is configured for Google-style docstrings:

```python
def my_function(data: Data, param: float) -> xr.DataArray:
    """Compute custom feature.
    
    Args:
        data: Input Data with 'time' and 'space' dimensions
        param: Custom parameter
        
    Returns:
        xarray DataArray with computed feature values
        
    Example:
        >>> result = cb.feature.my_feature(data, param=0.5)
    """
```

### Material Theme Features

- **Dark/light mode toggle**
- **Tabbed navigation**
- **Instant search**
- **Code highlighting with copy buttons**
- **Responsive design**
- **Table of contents sidebar**

## Next Steps

### Immediate

1. **Test the docs** - Run `uv run mkdocs serve` and browse <http://127.0.0.1:8000>
2. **Review content** - Check that all guides are accurate
3. **Improve docstrings** - Enhance docstrings in source files for better API docs

### Future Enhancements

1. **Add tutorials** - Step-by-step guides for common tasks
2. **Add examples** - More code examples in guides
3. **Add diagrams** - Architecture diagrams, data flow charts
4. **Add changelog** - Version history
5. **Configure deployment** - Set up automatic deployment on push

## Troubleshooting

### API Docs Not Showing

Make sure the module is importable:

```bash
uv run python -c "import cobrabox; print(cobrabox.__version__)"
```

### Build Errors

Check the mkdocstrings syntax:

```markdown
::: module.path.Class
    options:
        show_root_heading: true
```

### Theme Issues

Clear browser cache or try incognito mode.

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [mkdocstrings Python handler](https://mkdocstrings.github.io/python/)
