import subprocess
from typing import Any


def on_config(config: dict[str, Any], **kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], text=True
        ).strip()
        version = version.lstrip("v")
        config["extra"]["cobrabox_version"] = version
    except Exception:
        config["extra"]["cobrabox_version"] = "unknown"
    return config


def on_page_markdown(markdown: str, page: Any, **kwargs: dict[str, Any]) -> str | None:
    version = kwargs.get("config", {}).get("extra", {}).get("cobrabox_version")
    if version and "{{ cobrabox_version }}" in markdown:
        markdown = markdown.replace("{{ cobrabox_version }}", version)
    return markdown
