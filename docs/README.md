# Documentation - ONBOARDING

If you have any questions or problems with any of the guides. Please try to ask your favourite AI for assitance, or google or whatever. If you still have any issues, then please ask me.
~ Joakim

## Before the outing

1. First follow [`setup_repo.md`](setup_repo.md). To set up the repository locally.
2. If you have never made a pull request before. Follow [`your_first_pull_request.md`](your_first_pull_request.md).

## During the outing

### If feature dev

- Follow [`contributing/features.md`](contributing/features.md) for every feature
- For a quick reference on how to do a pr: [`how_to_make_a_pr.md`](how_to_make_a_pr.md)

## New Documentation Site

A new MkDocs-based documentation site is now available with:

- User guides and tutorials
- Auto-generated API reference from docstrings
- Better navigation and search

To view it locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

Then open <http://127.0.0.1:8000> in your browser.
