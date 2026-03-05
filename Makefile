setup:
	git lfs install && uv sync && uvx pre-commit install

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

docs-deploy:
	uv run mkdocs gh-deploy --force

run-examples:
	@for f in examples/*.py; do echo "Running $$f..."; uv run python "$$f"; done
