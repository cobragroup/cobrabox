"""Tests for downloader helpers and verify prompt logic."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cobrabox.downloader import (
    RemoteDatasetSpec,
    RemoteFile,
    _format_bytes,
    _head_size,
    _prompt_download_verify,
    ensure_remote_files,
)

# ---------------------------------------------------------------------------
# _format_bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0.0 B"),
        (512, "512.0 B"),
        (1023, "1023.0 B"),
        (1024, "1.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1024 * 1024 * 1024, "1.0 GB"),
        (int(1.5 * 1024 * 1024), "1.5 MB"),
    ],
)
def test_format_bytes(n: int, expected: str) -> None:
    assert _format_bytes(n) == expected


# ---------------------------------------------------------------------------
# _head_size
# ---------------------------------------------------------------------------


def test_head_size_returns_content_length() -> None:
    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = "1234"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = _head_size("http://example.com/file.zip")

    assert result == 1234


def test_head_size_returns_none_when_no_content_length() -> None:
    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = None
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = _head_size("http://example.com/file.zip")

    assert result is None


def test_head_size_returns_none_on_network_error() -> None:
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        result = _head_size("http://example.com/file.zip")

    assert result is None


# ---------------------------------------------------------------------------
# _prompt_download_verify
# ---------------------------------------------------------------------------


def _make_spec(tmp_path: Path, files: list[RemoteFile]) -> RemoteDatasetSpec:
    return RemoteDatasetSpec(
        identifier="test_dataset",
        local_rel_dir=tmp_path,
        files=files,
        loader=lambda d, s: [],  # type: ignore[arg-type]
        description="A test dataset.",
    )


def test_prompt_verify_returns_true_on_yes(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files)

    with (
        patch("cobrabox.downloader._head_size", return_value=1024 * 1024),
        patch("builtins.input", return_value="y"),
    ):
        assert _prompt_download_verify(spec, files) is True


def test_prompt_verify_returns_true_on_yes_uppercase(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files)

    with (
        patch("cobrabox.downloader._head_size", return_value=512),
        patch("builtins.input", return_value="YES"),
    ):
        assert _prompt_download_verify(spec, files) is True


def test_prompt_verify_returns_false_on_no(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files)

    with (
        patch("cobrabox.downloader._head_size", return_value=1024),
        patch("builtins.input", return_value="n"),
    ):
        assert _prompt_download_verify(spec, files) is False


def test_prompt_verify_returns_false_on_empty_input(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files)

    with (
        patch("cobrabox.downloader._head_size", return_value=1024),
        patch("builtins.input", return_value=""),
    ):
        assert _prompt_download_verify(spec, files) is False


def test_prompt_verify_shows_unknown_size_when_head_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files)

    with (
        patch("cobrabox.downloader._head_size", return_value=None),
        patch("builtins.input", return_value="n"),
    ):
        _prompt_download_verify(spec, files)

    out = capsys.readouterr().out
    assert "unknown" in out


def test_prompt_verify_extrapolates_size_for_large_datasets(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """When there are more files than the HEAD sample cap, size is extrapolated."""
    # 20 files but HEAD is capped at 16; each sampled file reports 1 MB
    files = [RemoteFile(url=f"http://example.com/{i}.zip", filename=f"{i}.zip") for i in range(20)]
    spec = _make_spec(tmp_path, files)

    with (
        patch("cobrabox.downloader._head_size", return_value=1024 * 1024),
        patch("builtins.input", return_value="n"),
    ):
        _prompt_download_verify(spec, files)

    out = capsys.readouterr().out
    assert "estimated" in out


# ---------------------------------------------------------------------------
# ensure_remote_files — verify integration
# ---------------------------------------------------------------------------


def test_ensure_remote_files_verify_false_skips_prompt(tmp_path: Path) -> None:
    """verify=False should never call _prompt_download_verify."""
    remote = RemoteFile(url="http://example.com/a.zip", filename="a.zip")
    spec = RemoteDatasetSpec(
        identifier="test_dataset",
        local_rel_dir=tmp_path,
        files=[remote],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    with (
        patch("cobrabox.downloader._prompt_download_verify") as mock_prompt,
        patch("cobrabox.downloader.ensure_remote_files.__wrapped__", create=True),
        # Prevent actual network call by making _has_valid_local_copy return True
        patch("urllib.request.urlopen", side_effect=AssertionError("should not download")),
    ):
        # Pre-create the file so nothing needs downloading
        (tmp_path / "a.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        ensure_remote_files(spec, repo_root=tmp_path.parent, verify=False)
        mock_prompt.assert_not_called()


def test_ensure_remote_files_verify_true_cancels_on_no(tmp_path: Path) -> None:
    """User declining the prompt raises RuntimeError."""
    remote = RemoteFile(url="http://example.com/a.zip", filename="a.zip")
    # Build a dataset dir that doesn't have the file yet
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    spec = RemoteDatasetSpec(
        identifier="test_dataset",
        local_rel_dir=Path("test_dataset"),
        files=[remote],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    with (
        patch("cobrabox.downloader._prompt_download_verify", return_value=False),
        patch("cobrabox.downloader._default_repo_root", return_value=tmp_path),
    ):
        with pytest.raises(RuntimeError, match="cancelled by user"):
            ensure_remote_files(spec, verify=True)


def test_ensure_remote_files_verify_true_proceeds_on_yes(tmp_path: Path) -> None:
    """User confirming the prompt allows the download to proceed."""
    remote = RemoteFile(url="http://example.com/a.zip", filename="a.zip")
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    spec = RemoteDatasetSpec(
        identifier="test_dataset",
        local_rel_dir=Path("test_dataset"),
        files=[remote],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    downloaded: list[str] = []

    def _fake_download(url: str, *args: object, **kwargs: object) -> MagicMock:
        downloaded.append(url)
        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = None
        mock_resp.read.side_effect = [b"PK\x05\x06" + b"\x00" * 18, b""]
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    with (
        patch("cobrabox.downloader._prompt_download_verify", return_value=True),
        patch("cobrabox.downloader._default_repo_root", return_value=tmp_path),
        patch("urllib.request.urlopen", side_effect=_fake_download),
    ):
        result = ensure_remote_files(spec, verify=True)

    assert result == dataset_dir
    assert downloaded == ["http://example.com/a.zip"]


def test_ensure_remote_files_no_prompt_when_all_cached(tmp_path: Path) -> None:
    """When all files are already present, verify prompt is never shown."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    # Create a valid (empty) zip so _has_valid_local_copy returns True
    zip_path = dataset_dir / "a.zip"
    with zipfile.ZipFile(zip_path, "w"):
        pass

    remote = RemoteFile(url="http://example.com/a.zip", filename="a.zip")
    spec = RemoteDatasetSpec(
        identifier="test_dataset",
        local_rel_dir=Path("test_dataset"),
        files=[remote],
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    with (
        patch("cobrabox.downloader._prompt_download_verify") as mock_prompt,
        patch("cobrabox.downloader._default_repo_root", return_value=tmp_path),
    ):
        ensure_remote_files(spec, verify=True)
        mock_prompt.assert_not_called()
