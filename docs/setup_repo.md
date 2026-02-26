# Repository Setup Guide

This is intentionally a MINIMAL guide on how to set it up. If you run into trouble, please just ask your favourite AI for help.

## 1) Clone the repo

This command is an option on how to clone it. If you use some GUI, then feel free to use it.

```bash
git clone git@github.com:cobragroup/cobrabox.git
cd cobrabox
```

If you get a permission error, your SSH key may not be set up for GitHub yet.
Use this guide:
`docs/setup_github_ssh_key.md`

## 2) Install `uv` (unless you already have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3) Set up the environment

```bash
uv sync && uv run pre-commit install
```

## 4) Verify that it works

```bash
uv run python examples/feature_pipeline_demo.py
```

## 5) Contribute a feature

When you are ready to add a new feature, follow:
`docs/contributing_feature.md`