from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fpl_forecast.ingest.snapshots import (
    latest_snapshot_path,
    read_metadata,
    sha256_bytes,
    write_raw_snapshot,
)


def test_write_raw_snapshot_checksum_metadata_and_latest_path(tmp_path):
    content = b'{"ok": true}'
    first_time = datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)
    first = write_raw_snapshot(
        tmp_path,
        season="2026-27",
        endpoint_name="bootstrap_static",
        content=content,
        source_url="https://example.test/bootstrap-static/",
        http_status=200,
        response_headers={"content-type": "application/json", "etag": "abc"},
        source="fpl_api",
        retrieved_at=first_time,
    )
    second = write_raw_snapshot(
        tmp_path,
        season="2026-27",
        endpoint_name="bootstrap_static",
        content=content,
        source_url="https://example.test/bootstrap-static/",
        http_status=200,
        response_headers={},
        source="fpl_api",
        retrieved_at=first_time + timedelta(microseconds=1),
    )

    assert first.raw_path != second.raw_path
    assert first.checksum_sha256 == sha256_bytes(content)
    metadata = read_metadata(first.raw_path)
    assert metadata["sha256"] == sha256_bytes(content)
    assert metadata["content_length"] == len(content)
    assert metadata["response_headers"]["etag"] == "abc"
    assert latest_snapshot_path(
        tmp_path,
        season="2026-27",
        endpoint_name="bootstrap_static",
        content_type="json",
    ) == second.raw_path


def test_write_raw_snapshot_refuses_same_path_overwrite(tmp_path):
    retrieved_at = datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)
    kwargs = {
        "season": "2026-27",
        "endpoint_name": "fixtures",
        "content": b"[]",
        "source_url": "https://example.test/fixtures/",
        "http_status": 200,
        "response_headers": {},
        "source": "fpl_api",
        "retrieved_at": retrieved_at,
    }
    write_raw_snapshot(tmp_path, **kwargs)

    with pytest.raises(FileExistsError):
        write_raw_snapshot(tmp_path, **kwargs)
