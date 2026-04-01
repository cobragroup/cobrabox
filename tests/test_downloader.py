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
        assert _prompt_download_verify(spec, files) is True


def test_prompt_verify_returns_true_on_yes_uppercase(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~512 B")

    with patch("builtins.input", return_value="YES"):
        assert _prompt_download_verify(spec, files) is True


def test_prompt_verify_returns_false_on_no(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~1 MB")

    with patch("builtins.input", return_value="n"):
        assert _prompt_download_verify(spec, files) is False


def test_prompt_verify_returns_false_on_empty_input(tmp_path: Path) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~1 MB")

    with patch("builtins.input", return_value=""):
        assert _prompt_download_verify(spec, files) is False


def test_prompt_verify_shows_unknown_size_when_no_size_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint=None)

    with patch("builtins.input", return_value="n"):
        _prompt_download_verify(spec, files)

    out = capsys.readouterr().out
    assert "unknown" in out


def test_prompt_verify_shows_size_hint(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    files = [RemoteFile(url="http://example.com/a.zip", filename="a.zip")]
    spec = _make_spec(tmp_path, files, size_hint="~10 GB")

    with patch("builtins.input", return_value="n"):
        _prompt_download_verify(spec, files)

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
        patch("cobrabox.downloader.get_data_dir", return_value=tmp_path),
    ):
        with pytest.raises(RuntimeError, match="cancelled by user"):
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

    def _fake_download(req: object, *args: object, **kwargs: object) -> MagicMock:
        url = req.full_url if hasattr(req, "full_url") else req
        downloaded.append(url)
        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = None
        mock_resp.read.side_effect = [b"PK\x05\x06" + b"\x00" * 18, b""]
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    with (
        patch("cobrabox.downloader._prompt_download_verify", return_value=True),
        patch("cobrabox.downloader.get_data_dir", return_value=tmp_path),
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
        patch("cobrabox.downloader.get_data_dir", return_value=tmp_path),
    ):
        ensure_remote_files(spec, accept=False)
        mock_prompt.assert_not_called()


class _NoOpBar:
    """No-op tqdm progress bar for use in downloader tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _NoOpBar:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def update(self, n: int) -> None:
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
# get_data_dir
# ---------------------------------------------------------------------------


def test_get_data_dir_returns_path() -> None:
    """get_data_dir returns a Path object."""
    from cobrabox.downloader import get_data_dir

    d = get_data_dir()
    assert isinstance(d, Path)


# ---------------------------------------------------------------------------
# ensure_remote_files — file_index_fn branch
# ---------------------------------------------------------------------------


def test_ensure_remote_files_uses_file_index_fn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_remote_files resolves files via file_index_fn when files is None."""
    import io

    import cobrabox.downloader as downloader

    resolved = [RemoteFile(url="http://example.com/a.bin", filename="a.bin")]

    spec = RemoteDatasetSpec(
        identifier="test_fn",
        local_rel_dir=Path("data") / "remote" / "test_fn",
        files=None,
        loader=lambda _p, _s=None: [],  # type: ignore[arg-type]
        file_index_fn=lambda: resolved,
    )

    class _FakeHeaders:
        def get(self, key: str, default: str | None = None) -> str | None:
            return default

    class _FakeResponse(io.BytesIO):
        headers = _FakeHeaders()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *exc_info: object) -> None:  # type: ignore[override]
            self.close()

    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda url, *a, **kw: _FakeResponse(b"DATA")
    )
    monkeypatch.setattr(downloader, "tqdm", lambda *a, **kw: _NoOpBar())

    dataset_dir = ensure_remote_files(spec, data_dir=tmp_path, accept=True)
    assert (dataset_dir / "a.bin").read_bytes() == b"DATA"
    # file_index_fn result should be cached on the spec
    assert spec.files is not None


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

    def _fake_urlopen(req: object, *a: object, **kw: object) -> _FakeResponse:
        url = req.full_url if hasattr(req, "full_url") else req
        downloaded.append(url)
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
# _chb_mit_file_index — error paths
# ---------------------------------------------------------------------------


def test_chb_mit_file_index_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_chb_mit_file_index raises RuntimeError when the RECORDS URL returns HTTP error."""
    import cobrabox.downloader as downloader
    from cobrabox.downloader import _chb_mit_file_index

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda url, **kw: (_ for _ in ()).throw(
            downloader.urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        ),
    )
    with pytest.raises(RuntimeError, match="HTTP 404"):
        _chb_mit_file_index()


def test_chb_mit_file_index_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_chb_mit_file_index raises RuntimeError on network error."""
    import cobrabox.downloader as downloader
    from cobrabox.downloader import _chb_mit_file_index

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda url, **kw: (_ for _ in ()).throw(
            downloader.urllib.error.URLError("connection refused")
        ),
    )
    with pytest.raises(RuntimeError, match="Network error"):
        _chb_mit_file_index()


# ---------------------------------------------------------------------------
# _siena_file_index — error paths
# ---------------------------------------------------------------------------


def test_siena_file_index_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_siena_file_index raises RuntimeError when the RECORDS URL returns HTTP error."""
    import cobrabox.downloader as downloader
    from cobrabox.downloader import _siena_file_index

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda url, **kw: (_ for _ in ()).throw(
            downloader.urllib.error.HTTPError(url, 503, "Unavailable", {}, None)
        ),
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        _siena_file_index()


def test_siena_file_index_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_siena_file_index raises RuntimeError on network error."""
    import cobrabox.downloader as downloader
    from cobrabox.downloader import _siena_file_index

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda url, **kw: (_ for _ in ()).throw(downloader.urllib.error.URLError("timeout")),
    )
    with pytest.raises(RuntimeError, match="Network error"):
        _siena_file_index()


# ---------------------------------------------------------------------------
# _sleep_ieeg_file_index — error paths
# ---------------------------------------------------------------------------


def test_sleep_ieeg_file_index_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_sleep_ieeg_file_index raises RuntimeError on network error."""
    import cobrabox.downloader as downloader
    from cobrabox.downloader import _sleep_ieeg_file_index

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda url, **kw: (_ for _ in ()).throw(downloader.urllib.error.URLError("timeout")),
    )
    with pytest.raises(RuntimeError, match="Network error"):
        _sleep_ieeg_file_index()


def test_sleep_ieeg_file_index_raises_when_no_subjects(monkeypatch: pytest.MonkeyPatch) -> None:
    """_sleep_ieeg_file_index raises RuntimeError when participants.tsv has no subjects."""
    import io

    import cobrabox.downloader as downloader
    from cobrabox.downloader import _sleep_ieeg_file_index

    empty_tsv = b"participant_id\tage\n"  # header only, no data rows

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc_info: object) -> None:
            self.close()

    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda url, **kw: _FakeResp(empty_tsv)
    )
    with pytest.raises(RuntimeError, match="no valid subjects"):
        _sleep_ieeg_file_index()


# ---------------------------------------------------------------------------
# _zurich_ieeg_file_index — error paths and happy path
# ---------------------------------------------------------------------------


def test_zurich_ieeg_file_index_builds_three_files_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """_zurich_ieeg_file_index returns .vhdr, .eeg, .vmrk for each run."""
    import io

    import cobrabox.downloader as downloader
    from cobrabox.downloader import _zurich_ieeg_file_index

    participants_tsv = b"participant_id\tage\nsub-01\t30\nsub-02\t25\n"
    scans_tsv = (
        b"filename\tacq_time\n"
        b"ieeg/sub-XX_ses-interictalsleep_run-01_ieeg.vhdr\t2013-01-01T00:00:00Z\n"
        b"ieeg/sub-XX_ses-interictalsleep_run-02_ieeg.vhdr\t2013-01-01T00:00:00Z\n"
    )

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc_info: object) -> None:
            self.close()

    responses = iter([participants_tsv, scans_tsv, scans_tsv])
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda url, **kw: _FakeResp(next(responses))
    )

    files = _zurich_ieeg_file_index()

    # 2 subjects x 2 runs x 3 extensions = 12 files
    assert len(files) == 12
    exts = {f.filename.rsplit(".", 1)[-1] for f in files}
    assert exts == {"vhdr", "eeg", "vmrk"}
    subjects = {f.subset_key for f in files}
    assert subjects == {"sub-01", "sub-02"}


def test_zurich_ieeg_file_index_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """_zurich_ieeg_file_index raises RuntimeError on network error."""
    import cobrabox.downloader as downloader
    from cobrabox.downloader import _zurich_ieeg_file_index

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda url, **kw: (_ for _ in ()).throw(downloader.urllib.error.URLError("timeout")),
    )
    with pytest.raises(RuntimeError, match="Network error"):
        _zurich_ieeg_file_index()


def test_zurich_ieeg_file_index_raises_when_no_subjects(monkeypatch: pytest.MonkeyPatch) -> None:
    """_zurich_ieeg_file_index raises RuntimeError when participants.tsv has no subjects."""
    import io

    import cobrabox.downloader as downloader
    from cobrabox.downloader import _zurich_ieeg_file_index

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc_info: object) -> None:
            self.close()

    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda url, **kw: _FakeResp(b"participant_id\tage\n")
    )
    with pytest.raises(RuntimeError, match="no valid subjects"):
        _zurich_ieeg_file_index()
