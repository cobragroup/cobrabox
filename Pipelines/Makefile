.PHONY: setup sync kernel

setup: sync kernel

sync:
	uv sync

# Register the venv as a Jupyter kernel named "cobrabox-pipeline"
kernel:
	uv run python -m ipykernel install --user --name cobrabox-pipeline --display-name "CobraBox Pipeline"
