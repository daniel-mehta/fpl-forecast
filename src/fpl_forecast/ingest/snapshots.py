from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RELEVANT_RESPONSE_HEADERS = (
    "cache-control",
    "content-encoding",
    "content-length",
    "content-type",
    "date",
    "etag",
    "last-modified",
)


@dataclass(frozen=True)
class SnapshotRecord:
    endpoint_name: str
    season: str
    raw_path: Path
    metadata_path: Path
    retrieved_at: str
    source_url: str
    source_version: str | None
    checksum_sha256: str
    content_length: int


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_retrieved_at(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def timestamp_for_path(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def endpoint_slug(endpoint_name: str) -> str:
    cleaned = endpoint_name.strip().strip("/")
    if not cleaned:
        raise ValueError("endpoint_name must not be empty")
    return (
        cleaned.replace("://", "_")
        .replace("/", "__")
        .replace("?", "_")
        .replace("&", "_")
        .replace("=", "-")
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def metadata_for_raw_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(raw_path.suffix + ".metadata.json")


def _snapshot_extension(content_type: str) -> str:
    if content_type == "json":
        return ".json"
    if content_type == "csv":
        return ".csv"
    raise ValueError(f"Unsupported snapshot content type: {content_type}")


def snapshot_path(
    base_dir: Path,
    *,
    season: str,
    endpoint_name: str,
    retrieved_at: datetime,
    content_type: str,
) -> Path:
    extension = _snapshot_extension(content_type)
    return base_dir / season / endpoint_slug(endpoint_name) / f"{timestamp_for_path(retrieved_at)}{extension}"


def write_raw_snapshot(
    base_dir: Path,
    *,
    season: str,
    endpoint_name: str,
    content: bytes,
    source_url: str,
    http_status: int,
    response_headers: dict[str, str] | None = None,
    target_season: str | None = None,
    source: str,
    source_version: str | None = None,
    content_type: str = "json",
    retrieved_at: datetime | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> SnapshotRecord:
    retrieved_at = retrieved_at or utc_now()
    raw_path = snapshot_path(
        base_dir,
        season=season,
        endpoint_name=endpoint_name,
        retrieved_at=retrieved_at,
        content_type=content_type,
    )
    if raw_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw snapshot: {raw_path}")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    checksum = sha256_bytes(content)
    raw_path.write_bytes(content)

    headers = {
        key.lower(): value
        for key, value in (response_headers or {}).items()
        if key.lower() in RELEVANT_RESPONSE_HEADERS
    }
    metadata: dict[str, Any] = {
        "source": source,
        "source_url": source_url,
        "source_version": source_version,
        "endpoint_name": endpoint_name,
        "retrieved_at": format_retrieved_at(retrieved_at),
        "http_status": http_status,
        "sha256": checksum,
        "content_length": len(content),
        "season": season,
        "target_season": target_season or season,
        "raw_path": str(raw_path),
        "response_headers": headers,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = metadata_for_raw_path(raw_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SnapshotRecord(
        endpoint_name=endpoint_name,
        season=season,
        raw_path=raw_path,
        metadata_path=metadata_path,
        retrieved_at=metadata["retrieved_at"],
        source_url=source_url,
        source_version=source_version,
        checksum_sha256=checksum,
        content_length=len(content),
    )


def latest_snapshot_path(
    base_dir: Path,
    *,
    season: str,
    endpoint_name: str,
    content_type: str,
) -> Path | None:
    extension = _snapshot_extension(content_type)
    directory = base_dir / season / endpoint_slug(endpoint_name)
    if not directory.exists():
        return None
    candidates = sorted(
        path
        for path in directory.glob(f"*{extension}")
        if not path.name.endswith(".metadata.json")
    )
    return candidates[-1] if candidates else None


def read_metadata(raw_path: Path) -> dict[str, Any]:
    metadata_path = metadata_for_raw_path(raw_path)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_json_snapshot(raw_path: Path) -> Any:
    return json.loads(raw_path.read_text(encoding="utf-8"))
