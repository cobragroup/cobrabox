# Repository Setup Guide

This is intentionally a MINIMAL guide on how to set it up. If you run into trouble, please just ask your favourite AI for help.

## 1) Clone the repo

This command is an option on how to clone it. If you use some GUI, then feel free to use it.

```bash
git clone git@github.com:cobragroup/cobrabox.git
```


If you get an error "Add permision denied (public key)", your SSH key is not set up for GitHub yet.
Use this guide:
[`docs/setup_github_ssh_key.md`](setup_github_ssh_key.md)

Once you have an ssh key set up, come back here and try to clone again.

## 2) Install `uv` (unless you already have it)
Enter the repository (the rest of the guide assumes you'll be running commands from here)
```bash
cd cobrabox
```

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
[`docs/contributing_feature.md`](contributing_feature.md)