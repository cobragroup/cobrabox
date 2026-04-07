"""Tests for downloader helpers and verify prompt logic."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cobrabox.downloader import (
    DownloadCancelled,
    RemoteDatasetSpec,
    RemoteFile,
    _format_bytes,
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
# _prompt_download_verify
# ---------------------------------------------------------------------------


def _make_spec(
    tmp_path: Path, files: list[RemoteFile], size_hint: str | None = None
) -> RemoteDatasetSpec:
    return RemoteDatasetSpec(
        identifier="test_dataset",
        local_rel_dir=tmp_path,
        files=files,
        loader=lambda d, s: [],  # type: ignore[arg-type]
        description="A test dataset.",
        size_hint=size_hint,
    )


def test_prompt_verify_returns_true_on_yes(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~1 MB")

    with patch("builtins.input", return_value="y"):
        assert _prompt_download_verify(spec, files, tmp_path) is True


def test_prompt_verify_returns_true_on_yes_uppercase(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~512 B")

    with patch("builtins.input", return_value="YES"):
        assert _prompt_download_verify(spec, files, tmp_path) is True


def test_prompt_verify_returns_false_on_no(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~1 MB")

    with patch("builtins.input", return_value="n"):
        assert _prompt_download_verify(spec, files, tmp_path) is False


def test_prompt_verify_returns_false_on_empty_input(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~1 MB")

    with patch("builtins.input", return_value=""):
        assert _prompt_download_verify(spec, files, tmp_path) is False


def test_prompt_verify_shows_unknown_size_when_no_size_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint=None)

    with patch("builtins.input", return_value="n"):
        _prompt_download_verify(spec, files, tmp_path)

    out = capsys.readouterr().out
    assert "unknown" in out


def test_prompt_verify_shows_size_hint(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~10 GB")

    with patch("builtins.input", return_value="n"):
        _prompt_download_verify(spec, files, tmp_path)

    out = capsys.readouterr().out
    assert "~10 GB" in out


# ---------------------------------------------------------------------------
# ensure_remote_files — verify integration
# ---------------------------------------------------------------------------


def test_ensure_remote_files_accept_true_skips_prompt(tmp_path: Path) -> None:
    """accept=True should never call _prompt_download_verify."""
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
        ensure_remote_files(spec, data_dir=tmp_path.parent, accept=True)
        mock_prompt.assert_not_called()


def test_ensure_remote_files_accept_false_cancels_on_no(tmp_path: Path) -> None:
    """User declining the prompt raises DownloadCancelled."""
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
        patch("cobrabox.downloader.get_dataset_dir", return_value=tmp_path),
    ):
        with pytest.raises(DownloadCancelled):
            ensure_remote_files(spec, accept=False)


def test_ensure_remote_files_accept_false_proceeds_on_yes(tmp_path: Path) -> None:
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

    def _fake_download(url: object, *args: object, **kwargs: object) -> MagicMock:
        downloaded.append(getattr(url, "full_url", url))
        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = None
        mock_resp.read.side_effect = [b"PK\x05\x06" + b"\x00" * 18, b""]
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    with (
        patch("cobrabox.downloader._prompt_download_verify", return_value=True),
        patch("cobrabox.downloader.get_dataset_dir", return_value=tmp_path),
        patch("urllib.request.urlopen", side_effect=_fake_download),
    ):
        result = ensure_remote_files(spec, accept=False)

    assert result == dataset_dir
    assert downloaded == ["http://example.com/a.zip"]


def test_ensure_remote_files_no_prompt_when_all_cached(tmp_path: Path) -> None:
    """When all files are already present, accept prompt is never shown."""
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
        patch("cobrabox.downloader.get_dataset_dir", return_value=tmp_path),
    ):
        ensure_remote_files(spec, accept=False)
        mock_prompt.assert_not_called()


class _NoOpBar:
    """No-op tqdm progress bar for use in downloader tests.

    Can be used as a drop-in patch for the ``tqdm`` class itself (not just
    instances), because it also exposes ``tqdm.write`` as a no-op staticmethod.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _NoOpBar:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def update(self, n: int) -> None:
        pass

    def close(self) -> None:
        pass

    @staticmethod
    def write(msg: str, *args: object, **kwargs: object) -> None:
        pass


# ---------------------------------------------------------------------------
# subset_keys — known_subset_keys fast path
# ---------------------------------------------------------------------------


def test_subset_keys_returns_list_from_known_subset_keys() -> None:
    """subset_keys() returns the static list without inspecting files."""
    from cobrabox.downloader import RemoteDatasetSpec, RemoteFile

    spec = RemoteDatasetSpec(
        identifier="test",
        local_rel_dir=Path("data/test"),
        files=[RemoteFile(url="http://x.com/a.zip", filename="a.zip", subset_key="A")],
        loader=lambda _p, _s=None: [],  # type: ignore[arg-type]
        subset_key_name="sets",
        known_subset_keys=("X", "Y", "Z"),
    )
    assert spec.subset_keys() == ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# get_dataset_dir
# ---------------------------------------------------------------------------


def test_get_dataset_dir_returns_path() -> None:
    """get_dataset_dir returns a Path object."""
    from cobrabox.downloader import get_dataset_dir

    d = get_dataset_dir()
    assert isinstance(d, Path)


# ---------------------------------------------------------------------------
# ensure_remote_files — corrupt zip triggers re-download
# ---------------------------------------------------------------------------


def test_ensure_remote_files_redownloads_corrupt_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .zip file that fails ZipFile validation is treated as missing."""
    import io

    import cobrabox.downloader as downloader

    dataset_dir = tmp_path / "data" / "remote" / "test_corrupt"
    dataset_dir.mkdir(parents=True)
    corrupt_zip = dataset_dir / "a.zip"
    corrupt_zip.write_bytes(b"not a zip")

    remote = RemoteFile(url="http://example.com/a.zip", filename="a.zip")
    spec = RemoteDatasetSpec(
        identifier="test_corrupt",
        local_rel_dir=Path("data") / "remote" / "test_corrupt",
        files=[remote],
        loader=lambda _p, _s=None: [],  # type: ignore[arg-type]
    )

    downloaded: list[str] = []

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return default

    class _FakeResponse(io.BytesIO):
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:  # type: ignore[override]
            self.close()

    def _fake_urlopen(url: object, *a: object, **kw: object) -> _FakeResponse:
        downloaded.append(getattr(url, "full_url", url))
        # Return a minimal valid zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass
        return _FakeResponse(buf.getvalue())

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(downloader, "tqdm", lambda *a, **kw: _NoOpBar())

    ensure_remote_files(spec, data_dir=tmp_path, accept=True)
    assert downloaded == ["http://example.com/a.zip"]


# ---------------------------------------------------------------------------
# ensure_remote_files — timeout and ENOSPC errors
# ---------------------------------------------------------------------------


def test_ensure_remote_files_raises_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A socket timeout during download raises RuntimeError with a clear message."""
    import cobrabox.downloader as downloader

    remote = RemoteFile(url="http://example.com/a.bin", filename="a.bin")
    spec = RemoteDatasetSpec(
        identifier="test_timeout",
        local_rel_dir=Path("data") / "remote" / "test_timeout",
        files=[remote],
        loader=lambda _p, _s=None: [],  # type: ignore[arg-type]
    )

    def _timeout(url: str, *a: object, **kw: object) -> None:
        raise downloader.urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _timeout)

    with pytest.raises(RuntimeError, match="timed out"):
        ensure_remote_files(spec, data_dir=tmp_path, accept=True)


def test_ensure_remote_files_raises_on_enospc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ENOSPC OSError during download raises RuntimeError with a clear message."""
    import errno as errno_mod

    import cobrabox.downloader as downloader

    remote = RemoteFile(url="http://example.com/a.bin", filename="a.bin")
    spec = RemoteDatasetSpec(
        identifier="test_enospc",
        local_rel_dir=Path("data") / "remote" / "test_enospc",
        files=[remote],
        loader=lambda _p, _s=None: [],  # type: ignore[arg-type]
    )

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return default

    class _FakeResponse:
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

        def read(self, n: int) -> bytes:
            raise OSError(errno_mod.ENOSPC, "No space left on device")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda url, *a, **kw: _FakeResponse())

    with pytest.raises(RuntimeError, match="No space left"):
        ensure_remote_files(spec, data_dir=tmp_path, accept=True)


# ---------------------------------------------------------------------------
# KeyboardInterrupt cancellation
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_during_download_raises_and_leaves_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt during download re-raises and leaves .part file for resumption."""
    import io

    import cobrabox.downloader as downloader

    files = [RemoteFile(url="http://example.com/big.bin", filename="big.bin")]
    spec = _make_spec(tmp_path, files)

    chunk_count = 0

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return None

    class _FakeResponse(io.RawIOBase):
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

        def read(self, n: int = -1) -> bytes:
            nonlocal chunk_count
            chunk_count += 1
            if chunk_count == 2:
                raise KeyboardInterrupt
            return b"x" * 65536

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda url, *a, **kw: _FakeResponse())
    monkeypatch.setattr(downloader, "tqdm", lambda *a, **kw: _NoOpBar())

    with pytest.raises(KeyboardInterrupt):
        ensure_remote_files(spec, data_dir=tmp_path, accept=True)

    # .part file should exist (resumable), final file should not
    assert (tmp_path / "big.bin.part").exists()
    assert not (tmp_path / "big.bin").exists()


# ---------------------------------------------------------------------------
# Retry on transient network errors
# ---------------------------------------------------------------------------


def test_download_retries_on_network_error_and_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient URLError is retried; download succeeds on the third attempt."""
    import io

    import cobrabox.downloader as downloader

    files = [RemoteFile(url="http://example.com/a.bin", filename="a.bin")]
    spec = _make_spec(tmp_path, files)

    attempt = 0

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return None

    class _GoodResponse(io.RawIOBase):
        headers = _FakeHeaders()

        def __enter__(self) -> _GoodResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

        def read(self, n: int = -1) -> bytes:
            return b""  # EOF immediately → empty but valid file

    def _urlopen(url: object, *a: object, **kw: object) -> object:
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise downloader.urllib.error.URLError("connection reset")
        return _GoodResponse()

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(downloader, "tqdm", _NoOpBar)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)  # instant retries

    ensure_remote_files(spec, data_dir=tmp_path, accept=True)

    assert attempt == 3
    assert (tmp_path / "a.bin").exists()


def test_download_raises_after_all_retries_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After all retry attempts fail the final RuntimeError is raised."""
    import cobrabox.downloader as downloader

    files = [RemoteFile(url="http://example.com/a.bin", filename="a.bin")]
    spec = _make_spec(tmp_path, files)

    attempt = 0

    def _urlopen(url: object, *a: object, **kw: object) -> object:
        nonlocal attempt
        attempt += 1
        raise downloader.urllib.error.URLError("network unavailable")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Network error"):
        ensure_remote_files(spec, data_dir=tmp_path, accept=True)

    # One initial attempt + one per retry delay = 4 total
    assert attempt == len(downloader._RETRY_DELAYS) + 1


def test_download_retry_preserves_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The .part file is kept between retry attempts so the download can resume."""
    import io

    import cobrabox.downloader as downloader

    files = [RemoteFile(url="http://example.com/a.bin", filename="a.bin")]
    spec = _make_spec(tmp_path, files)

    part_path = tmp_path / "a.bin.part"
    attempt = 0

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return None

    class _PartialResponse(io.RawIOBase):
        """Writes one chunk then raises a network error."""

        headers = _FakeHeaders()

        def __enter__(self) -> _PartialResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

        def read(self, n: int = -1) -> bytes:
            raise downloader.urllib.error.URLError("dropped")

    class _GoodResponse(io.RawIOBase):
        headers = _FakeHeaders()

        def __enter__(self) -> _GoodResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

        def read(self, n: int = -1) -> bytes:
            return b""

    def _urlopen(url: object, *a: object, **kw: object) -> object:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            part_path.write_bytes(b"x" * 100)  # simulate partial progress
            return _PartialResponse()
        return _GoodResponse()

    monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(downloader, "tqdm", _NoOpBar)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)

    ensure_remote_files(spec, data_dir=tmp_path, accept=True)

    assert attempt == 2
    assert (tmp_path / "a.bin").exists()


# ---------------------------------------------------------------------------
# dry_run mode
# ---------------------------------------------------------------------------


def test_dry_run_prints_summary_and_does_not_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """dry_run=True prints a summary and returns without downloading anything."""
    import cobrabox.downloader as downloader

    files = [
        RemoteFile(url="http://example.com/a.bin", filename="a.bin", subset_key="A"),
        RemoteFile(url="http://example.com/b.bin", filename="b.bin", subset_key="B"),
    ]
    spec = RemoteDatasetSpec(
        identifier="test_dry",
        local_rel_dir=tmp_path,
        files=files,
        loader=lambda d, s: [],  # type: ignore[arg-type]
        size_hint="~5 MB",
        subset_key_name="subsets",
    )
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not download")),
    )

    result = ensure_remote_files(spec, data_dir=tmp_path.parent, accept=True, dry_run=True)

    out = capsys.readouterr().out
    assert "Dry run" in out
    assert (
        "a.bin" not in (tmp_path.parent / spec.local_rel_dir).parts
        or not (tmp_path.parent / "a.bin").exists()
    )
    assert "~5 MB" in out
    assert "A" in out
    assert "B" in out
    # Should return the would-be dataset dir path
    assert result == tmp_path.parent / spec.local_rel_dir


def test_dry_run_shows_cached_count(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """dry_run reports how many files are already cached vs. to be downloaded."""
    files = [
        RemoteFile(url="http://example.com/a.bin", filename="a.bin", subset_key="A"),
        RemoteFile(url="http://example.com/b.bin", filename="b.bin", subset_key="B"),
    ]
    spec = _make_spec(tmp_path, files)

    # Pre-cache one file
    (tmp_path / "a.bin").write_bytes(b"data")

    ensure_remote_files(spec, data_dir=tmp_path.parent, accept=True, dry_run=True)

    out = capsys.readouterr().out
    assert "1 already cached" in out
    assert "1 file" in out  # one to download


def test_dry_run_when_all_cached(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """dry_run reports nothing to download when all files are already cached."""
    files = [RemoteFile(url="http://example.com/a.bin", filename="a.bin")]
    spec = _make_spec(tmp_path, files)
    (tmp_path / "a.bin").write_bytes(b"data")

    ensure_remote_files(spec, data_dir=tmp_path.parent, accept=True, dry_run=True)

    out = capsys.readouterr().out
    assert "already cached" in out
    assert "Would download" not in out


def test_dry_run_does_not_create_dataset_directory(tmp_path: Path) -> None:
    """dry_run=True does not create the dataset directory."""
    files = [RemoteFile(url="http://example.com/a.bin", filename="a.bin")]
    new_dir = tmp_path / "new_dataset_dir"
    spec = RemoteDatasetSpec(
        identifier="test_dry_nodir",
        local_rel_dir=new_dir,
        files=files,
        loader=lambda d, s: [],  # type: ignore[arg-type]
    )

    ensure_remote_files(spec, accept=True, dry_run=True)

    assert not new_dir.exists()
