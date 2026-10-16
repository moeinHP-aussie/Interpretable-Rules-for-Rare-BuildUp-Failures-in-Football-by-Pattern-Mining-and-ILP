from __future__ import annotations

import requests
import pytest

from tactical_ilp.data import load_raw_data


def test_load_frames_caches_empty_result_for_404(monkeypatch, tmp_path):
    response = requests.Response()
    response.status_code = 404
    error = requests.HTTPError("not found", response=response)
    monkeypatch.setattr(load_raw_data, "cache_path", lambda kind, mid: tmp_path / f"{kind}_{mid}.parquet")
    monkeypatch.setattr("statsbombpy.sb.frames", lambda **kwargs: (_ for _ in ()).throw(error))

    frames = load_raw_data.load_frames(123)

    assert frames.empty
    assert list(frames.columns) == load_raw_data.FRAME_COLUMNS
    assert (tmp_path / "frames_123.parquet").exists()


@pytest.mark.parametrize("status_code", [None, 429, 500])
def test_load_frames_does_not_cache_transient_http_errors(monkeypatch, tmp_path, status_code):
    response = None
    if status_code is not None:
        response = requests.Response()
        response.status_code = status_code
    error = requests.HTTPError("temporary failure", response=response)
    monkeypatch.setattr(load_raw_data, "cache_path", lambda kind, mid: tmp_path / f"{kind}_{mid}.parquet")
    monkeypatch.setattr("statsbombpy.sb.frames", lambda **kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(requests.HTTPError):
        load_raw_data.load_frames(123)

    assert not (tmp_path / "frames_123.parquet").exists()


def test_load_frames_does_not_swallow_non_http_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(load_raw_data, "cache_path", lambda kind, mid: tmp_path / f"{kind}_{mid}.parquet")
    monkeypatch.setattr(
        "statsbombpy.sb.frames",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("malformed response")),
    )

    with pytest.raises(ValueError, match="malformed response"):
        load_raw_data.load_frames(123)

    assert not (tmp_path / "frames_123.parquet").exists()
