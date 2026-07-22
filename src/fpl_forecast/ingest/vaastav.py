from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

from fpl_forecast.config import (
    HttpSettings,
    RAW_VAASTAV_DIR,
    VAASTAV_GITHUB_API_BASE_URL,
    VAASTAV_RAW_BASE_URL,
)
from fpl_forecast.ingest.snapshots import (
    SnapshotRecord,
    latest_snapshot_path,
    read_metadata,
    write_raw_snapshot,
)


MERGED_GW = "merged_gw"
PLAYERS_RAW = "players_raw"

VAASTAV_FILES = {
    MERGED_GW: "gws/merged_gw.csv",
    PLAYERS_RAW: "players_raw.csv",
}

REQUIRED_COLUMNS = {
    MERGED_GW: {"element", "fixture", "round", "minutes", "total_points"},
    PLAYERS_RAW: {"id", "code"},
}


class VaastavDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class VaastavIngestResult:
    records: list[SnapshotRecord]
    source_version: str
    warnings: list[str] = field(default_factory=list)


class VaastavIngestor:
    def __init__(
        self,
        *,
        raw_dir=RAW_VAASTAV_DIR,
        settings: HttpSettings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.settings = settings or HttpSettings()
        self.transport = transport

    def ingest_season(
        self,
        *,
        season: str,
        refresh: bool = False,
        revision: str | None = None,
    ) -> VaastavIngestResult:
        cached_paths = {
            dataset_name: latest_snapshot_path(
                self.raw_dir,
                season=season,
                endpoint_name=dataset_name,
                content_type="csv",
            )
            for dataset_name in VAASTAV_FILES
        }
        if revision is None and not refresh and all(cached_paths.values()):
            first_cached = next(path for path in cached_paths.values() if path is not None)
            source_version = read_metadata(first_cached).get("source_version") or "cached"
        else:
            source_version = revision or self.resolve_revision()

        records: list[SnapshotRecord] = []
        warnings: list[str] = []

        for dataset_name, repo_path in VAASTAV_FILES.items():
            endpoint_name = dataset_name
            cached_path = cached_paths[dataset_name]
            if cached_path is not None and not refresh:
                metadata = read_metadata(cached_path)
                records.append(
                    SnapshotRecord(
                        endpoint_name=endpoint_name,
                        season=season,
                        raw_path=cached_path,
                        metadata_path=cached_path.with_suffix(
                            cached_path.suffix + ".metadata.json"
                        ),
                        retrieved_at=metadata["retrieved_at"],
                        source_url=metadata["source_url"],
                        source_version=metadata.get("source_version"),
                        checksum_sha256=metadata["sha256"],
                        content_length=metadata["content_length"],
                    )
                )
                if dataset_name == MERGED_GW and _csv_has_column(cached_path.read_bytes(), "xP"):
                    warnings.append("Vaastav merged_gw.csv contains xP; keep it out of features.")
                continue

            source_url = self.raw_url(season=season, repo_path=repo_path, revision=source_version)
            content, status_code, headers = self._get_bytes(source_url)
            columns = _csv_columns(content)
            missing = sorted(REQUIRED_COLUMNS[dataset_name].difference(columns))
            if missing:
                raise VaastavDataError(
                    f"Vaastav {dataset_name} for {season} is missing columns: "
                    f"{', '.join(missing)}"
                )
            if dataset_name == MERGED_GW and "xP" in columns:
                warnings.append("Vaastav merged_gw.csv contains xP; keep it out of features.")

            records.append(
                write_raw_snapshot(
                    self.raw_dir,
                    season=season,
                    endpoint_name=endpoint_name,
                    content=content,
                    source_url=source_url,
                    http_status=status_code,
                    response_headers=dict(headers),
                    target_season=season,
                    source="vaastav",
                    source_version=source_version,
                    content_type="csv",
                    extra_metadata={"repo_path": f"data/{season}/{repo_path}"},
                )
            )

        return VaastavIngestResult(records=records, source_version=source_version, warnings=warnings)

    def resolve_revision(self) -> str:
        url = f"{VAASTAV_GITHUB_API_BASE_URL}/branches/master"
        payload = self._get_json(url)
        try:
            revision = payload["commit"]["sha"]
        except (KeyError, TypeError) as exc:
            raise VaastavDataError(f"Could not resolve Vaastav master revision from {url}") from exc
        if not isinstance(revision, str) or not revision:
            raise VaastavDataError(f"Vaastav revision from {url} is empty or invalid.")
        return revision

    @staticmethod
    def raw_url(*, season: str, repo_path: str, revision: str) -> str:
        return f"{VAASTAV_RAW_BASE_URL}/{revision}/data/{season}/{repo_path}"

    def _get_json(self, url: str) -> Any:
        content, _, _ = self._get_bytes(url, accept="application/vnd.github+json")
        try:
            return httpx.Response(200, content=content).json()
        except ValueError as exc:
            raise VaastavDataError(f"Invalid JSON while reading Vaastav source metadata: {url}") from exc

    def _get_bytes(
        self,
        url: str,
        *,
        accept: str = "text/csv,application/json",
    ) -> tuple[bytes, int, httpx.Headers]:
        headers = {"User-Agent": self.settings.user_agent, "Accept": accept}
        last_error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                with httpx.Client(
                    timeout=self.settings.timeout_seconds,
                    headers=headers,
                    transport=self.transport,
                    follow_redirects=True,
                ) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    return response.content, response.status_code, response.headers
            except httpx.HTTPStatusError as exc:
                raise VaastavDataError(
                    f"Vaastav source returned HTTP {exc.response.status_code} for {url}: "
                    f"{exc.response.text[:300]}"
                ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= self.settings.retries:
                    break
                time.sleep(self.settings.backoff_seconds * (2**attempt))
        raise VaastavDataError(f"Vaastav request failed for {url}: {last_error}") from last_error


def _csv_columns(content: bytes) -> set[str]:
    try:
        return set(pd.read_csv(io.BytesIO(content), nrows=0).columns)
    except Exception as exc:  # noqa: BLE001
        raise VaastavDataError("Vaastav CSV could not be parsed.") from exc


def _csv_has_column(content: bytes, column: str) -> bool:
    return column in _csv_columns(content)


def load_latest_vaastav_csv(*, raw_dir=RAW_VAASTAV_DIR, season: str, dataset_name: str) -> tuple[pd.DataFrame, dict[str, Any], str]:
    raw_path = latest_snapshot_path(
        raw_dir,
        season=season,
        endpoint_name=dataset_name,
        content_type="csv",
    )
    if raw_path is None:
        raise VaastavDataError(f"No cached Vaastav {dataset_name!r} CSV exists for season {season}.")
    metadata = read_metadata(raw_path)
    try:
        dataframe = pd.read_csv(raw_path)
    except Exception as exc:  # noqa: BLE001
        raise VaastavDataError(f"Could not parse cached Vaastav CSV: {raw_path}") from exc
    return dataframe, metadata, str(raw_path)
