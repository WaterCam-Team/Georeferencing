"""Tests for scripts/odm_process.py — no NodeODM server required."""
import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from odm_process import _api, parse_options, download_asset, _DEFAULT_OPTIONS


# ---------------------------------------------------------------------------
# _api — URL construction
# ---------------------------------------------------------------------------

def test_api_joins_base_and_path():
    assert _api("http://localhost:3000", "/task/new/init") == \
        "http://localhost:3000/task/new/init"


def test_api_strips_trailing_slash_from_base():
    assert _api("http://localhost:3000/", "/task/new/init") == \
        "http://localhost:3000/task/new/init"


# ---------------------------------------------------------------------------
# parse_options
# ---------------------------------------------------------------------------

def test_parse_options_empty_returns_defaults():
    opts = parse_options([])
    for k, v in _DEFAULT_OPTIONS.items():
        assert opts[k] == v


def test_parse_options_override_one_key():
    opts = parse_options(["dsm=false"])
    assert opts["dsm"] == "false"
    assert opts["orthophoto-resolution"] == _DEFAULT_OPTIONS["orthophoto-resolution"]


def test_parse_options_add_new_key():
    opts = parse_options(["fast-orthophoto=true"])
    assert opts["fast-orthophoto"] == "true"


def test_parse_options_bad_format_raises_value_error():
    with pytest.raises(ValueError, match="key=value"):
        parse_options(["badoption"])


def test_parse_options_value_with_equals_keeps_full_value():
    # e.g. a key whose value contains "=" (unlikely but safe to handle)
    opts = parse_options(["proj=+proj=utm"])
    assert opts["proj"] == "+proj=utm"


# ---------------------------------------------------------------------------
# download_asset — ZIP response
# ---------------------------------------------------------------------------

def _make_zip_bytes(filename: str = "dsm.tif", content: bytes = b"TIFDATA") -> bytes:
    """Return bytes of a valid ZIP archive containing one file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def _mock_get(content: bytes):
    """Return a mock for requests.get(stream=True) that yields content."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.iter_content.return_value = [content]
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=cm)


def test_download_asset_extracts_tif_from_zip(tmp_path):
    zip_bytes = _make_zip_bytes("dsm.tif", b"FAKE_TIF")
    out_path = tmp_path / "dsm.tif"

    with patch("odm_process.requests.get", _mock_get(zip_bytes)):
        result = download_asset("http://localhost:3000", "abc123", "odm_dem", out_path)

    assert result is True
    assert out_path.exists()
    assert out_path.read_bytes() == b"FAKE_TIF"


def test_download_asset_handles_nested_zip_path(tmp_path):
    """ZIP entries with directory prefix (e.g. odm_dem/dsm.tif) are found correctly."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("odm_dem/dsm.tif", b"NESTED")
    zip_bytes = buf.getvalue()
    out_path = tmp_path / "dsm.tif"

    with patch("odm_process.requests.get", _mock_get(zip_bytes)):
        result = download_asset("http://localhost:3000", "abc123", "odm_dem", out_path)

    assert result is True
    assert out_path.read_bytes() == b"NESTED"


def test_download_asset_direct_file_not_zip(tmp_path):
    """Non-ZIP response is written directly to out_path."""
    raw_tif = b"\x49\x49\x2a\x00" + b"\x00" * 100  # TIFF magic bytes
    out_path = tmp_path / "orthophoto.tif"

    with patch("odm_process.requests.get", _mock_get(raw_tif)):
        result = download_asset("http://localhost:3000", "abc123", "orthophoto", out_path)

    assert result is True
    assert out_path.exists()
    assert out_path.read_bytes() == raw_tif


def test_download_asset_zip_with_no_tif_returns_false(tmp_path):
    """ZIP containing no .tif file should return False."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", b"no tif here")
    out_path = tmp_path / "out.tif"

    with patch("odm_process.requests.get", _mock_get(buf.getvalue())):
        result = download_asset("http://localhost:3000", "abc123", "odm_dem", out_path)

    assert result is False
    assert not out_path.exists()


def test_download_asset_http_error_returns_false(tmp_path):
    """HTTP error during download returns False without crashing."""
    resp = MagicMock()
    resp.raise_for_status.side_effect = Exception("404 Not Found")
    resp.iter_content.return_value = []
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    mock_get = MagicMock(return_value=cm)

    out_path = tmp_path / "out.tif"
    with patch("odm_process.requests.get", mock_get):
        result = download_asset("http://localhost:3000", "abc123", "odm_dem", out_path)

    assert result is False
    assert not out_path.exists()
