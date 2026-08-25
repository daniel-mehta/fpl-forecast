from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_forecast.ingest.fpl_api import BOOTSTRAP_STATIC, ELEMENT_SUMMARY, FPLApiClient
from fpl_forecast.ingest.season import SeasonIdentityError, infer_bootstrap_season
from fpl_forecast.ingest.snapshots import (
    read_json_snapshot,
    read_metadata,
    sha256_bytes,
)


POSITION_BY_ELEMENT_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD", 5: "AM"}


class HistoricalClubResolutionError(ValueError):
    """Raised when a player-fixture club cannot be resolved without guessing."""


@dataclass(frozen=True)
class PlayerIdentityEvidence:
    player_id: int
    player_uid: str
    player_code: int | None
    player_name: str
    entity_type: str
    fpl_position: str
    current_team_id: int | None
    identity_source: str
    identity_raw_snapshot_path: str | None
    identity_sha256: str | None
    identity_retrieved_at: str | None


@dataclass(frozen=True)
class HistoricalClubAssignment:
    player_id: int
    fixture_id: int
    historical_team_id: int
    current_team_id: int | None
    opponent_team_id: int
    was_home: bool
    resolution_method: str
    evidence_endpoint: str
    evidence_raw_snapshot_path: str
    evidence_sha256: str
    evidence_retrieved_at: str


@dataclass(frozen=True)
class ResolvedEventPlayers:
    fixture_blocks: dict[int, tuple[dict[str, Any], ...]]
    identities: dict[int, PlayerIdentityEvidence]
    club_assignments: dict[tuple[int, int], HistoricalClubAssignment]


@dataclass(frozen=True)
class _ArchivedPlayerRecord:
    player_id: int
    player_code: int | None
    player_name: str | None
    element_type: int | None
    team_id: int | None
    raw_path: str
    metadata: dict[str, Any]


class HistoricalClubResolver:
    """Resolve temporal player clubs from fixture-specific official evidence.

    The latest bootstrap remains authoritative for current selection. Historical
    rows are resolved independently for each official fixture. Network requests
    are cache-first and are made only for a fixture mismatch or a missing
    fixture identity.
    """

    def __init__(
        self,
        *,
        season: str,
        bootstrap_payload: dict[str, Any],
        bootstrap_metadata: dict[str, Any],
        bootstrap_raw_path: str,
        fixtures_payload: list[dict[str, Any]],
        raw_fpl_dir: Path,
        normalized_dir: Path,
        client: FPLApiClient,
        information_cutoff: str | pd.Timestamp,
        allow_network: bool,
        source_context: dict[str, Any],
    ) -> None:
        self.season = season
        self.bootstrap_payload = bootstrap_payload
        self.bootstrap_metadata = bootstrap_metadata
        self.bootstrap_raw_path = str(bootstrap_raw_path)
        self.raw_fpl_dir = Path(raw_fpl_dir)
        self.client = client
        self.cutoff = _utc_timestamp(information_cutoff)
        self.allow_network = allow_network
        self.source_context = source_context
        self.fixtures = _fixture_lookup(fixtures_payload)
        self.current_players = _element_lookup(bootstrap_payload.get("elements", []), "bootstrap-static")
        self.excluded_archive_snapshots: list[dict[str, Any]] = []
        self.archive = self._load_bootstrap_archive()
        self.historical_identity = _historical_identity_lookup(normalized_dir)
        self._summaries: dict[int, tuple[dict[str, Any], dict[str, Any], str]] = {}
        self._refreshed_summary_players: set[int] = set()
        self.source_hashes: dict[str, str] = {}
        self.snapshot_entries: dict[str, dict[str, Any]] = {}
        self.resolution_records: list[dict[str, Any]] = []
        self.summary_player_ids: set[int] = set()
        self._resolved_keys: set[tuple[int, int]] = set()

    def resolve_event(
        self,
        *,
        gameweek: int,
        payload: dict[str, Any],
        eligible_fixture_ids: set[int],
    ) -> ResolvedEventPlayers:
        elements = payload.get("elements")
        if not isinstance(elements, list):
            raise HistoricalClubResolutionError(
                "Official event-live payload must contain an elements list before club resolution."
            )
        element_lookup = _element_lookup(elements, f"event-live GW{gameweek}")
        blocks_by_player: dict[int, tuple[dict[str, Any], ...]] = {}
        identities: dict[int, PlayerIdentityEvidence] = {}
        assignments: dict[tuple[int, int], HistoricalClubAssignment] = {}

        for player_id, element in element_lookup.items():
            raw_blocks = element.get("explain") or []
            if not isinstance(raw_blocks, list):
                raise HistoricalClubResolutionError(
                    f"Official event-live player {player_id} has malformed fixture explain evidence."
                )
            selected_blocks = self._eligible_explain_blocks(
                player_id=player_id,
                gameweek=gameweek,
                raw_blocks=raw_blocks,
                eligible_fixture_ids=eligible_fixture_ids,
            )
            summary: dict[str, Any] | None = None
            if not raw_blocks and eligible_fixture_ids:
                summary, _, _ = self._load_element_summary(
                    player_id,
                    required_fixture_ids=eligible_fixture_ids,
                )
                selected_blocks = self._summary_fixture_blocks(
                    player_id=player_id,
                    gameweek=gameweek,
                    summary=summary,
                    eligible_fixture_ids=eligible_fixture_ids,
                )
            if not selected_blocks:
                continue

            player_assignments: list[HistoricalClubAssignment] = []
            for block in selected_blocks:
                fixture_id = int(block["fixture"])
                assignment = self._resolve_fixture_club(
                    player_id=player_id,
                    fixture_id=fixture_id,
                    summary=summary,
                )
                summary = summary or self._summaries.get(player_id, (None, {}, ""))[0]
                key = (player_id, fixture_id)
                if key in assignments or key in self._resolved_keys:
                    raise HistoricalClubResolutionError(
                        f"Duplicate historical club resolution for player {player_id}, fixture {fixture_id}."
                    )
                assignments[key] = assignment
                self._resolved_keys.add(key)
                player_assignments.append(assignment)
                record = asdict(assignment)
                record["gameweek"] = gameweek
                self.resolution_records.append(record)

            identities[player_id] = self._resolve_identity(
                player_id,
                summary=summary,
                assignments=player_assignments,
            )
            blocks_by_player[player_id] = tuple(selected_blocks)

        return ResolvedEventPlayers(
            fixture_blocks=blocks_by_player,
            identities=identities,
            club_assignments=assignments,
        )

    def _eligible_explain_blocks(
        self,
        *,
        player_id: int,
        gameweek: int,
        raw_blocks: list[dict[str, Any]],
        eligible_fixture_ids: set[int],
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen: set[int] = set()
        for block in raw_blocks:
            if not isinstance(block, dict) or block.get("fixture") is None:
                raise HistoricalClubResolutionError(
                    f"Official event-live player {player_id} has a fixture block without an ID."
                )
            fixture_id = int(block["fixture"])
            fixture = self.fixtures.get(fixture_id)
            if fixture is None:
                raise HistoricalClubResolutionError(
                    f"Official event-live player {player_id} references unknown fixture {fixture_id}."
                )
            if int(fixture.get("event") or 0) != gameweek:
                raise HistoricalClubResolutionError(
                    f"Official event-live player {player_id} references fixture {fixture_id} from "
                    f"gameweek {fixture.get('event')}, not {gameweek}."
                )
            if fixture_id not in eligible_fixture_ids:
                continue
            if fixture_id in seen:
                raise HistoricalClubResolutionError(
                    f"Official event-live player {player_id} repeats fixture {fixture_id}."
                )
            seen.add(fixture_id)
            selected.append(block)
        return selected

    def _summary_fixture_blocks(
        self,
        *,
        player_id: int,
        gameweek: int,
        summary: dict[str, Any],
        eligible_fixture_ids: set[int],
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for item in summary.get("history", []):
            if not isinstance(item, dict) or item.get("fixture") is None:
                continue
            fixture_id = int(item["fixture"])
            if fixture_id not in eligible_fixture_ids:
                continue
            if item.get("round") is not None and int(item["round"]) != gameweek:
                raise self._diagnostic_error(
                    player_id,
                    fixture_id,
                    "element-summary round conflicts with the official fixture event",
                    summary_record=item,
                )
            blocks.append({"fixture": fixture_id, "stats": []})
        fixture_ids = [int(block["fixture"]) for block in blocks]
        if len(fixture_ids) != len(set(fixture_ids)):
            raise HistoricalClubResolutionError(
                f"Element-summary history repeats a fixture for player {player_id}: {fixture_ids}."
            )
        return blocks

    def _resolve_fixture_club(
        self,
        *,
        player_id: int,
        fixture_id: int,
        summary: dict[str, Any] | None,
    ) -> HistoricalClubAssignment:
        fixture = self.fixtures[fixture_id]
        sides = {int(fixture["team_h"]), int(fixture["team_a"])}
        current = self.current_players.get(player_id)
        current_team_id = _optional_int(current.get("team")) if current else None
        if current_team_id in sides:
            return self._assignment_from_team(
                player_id=player_id,
                fixture_id=fixture_id,
                historical_team_id=int(current_team_id),
                current_team_id=current_team_id,
                resolution_method="current_bootstrap_fixture_compatible",
                endpoint=BOOTSTRAP_STATIC,
                raw_path=self.bootstrap_raw_path,
                metadata=self.bootstrap_metadata,
            )

        archived_candidates = [
            record
            for record in self._eligible_archived_records(player_id)
            if record.team_id is not None and record.team_id in sides
        ]
        archived_teams = {record.team_id for record in archived_candidates}
        if len(archived_teams) == 1:
            historical_team_id = int(next(iter(archived_teams)))
            evidence = max(
                (record for record in archived_candidates if record.team_id == historical_team_id),
                key=lambda record: str(record.metadata.get("retrieved_at") or ""),
            )
            endpoint = self._register_archived_bootstrap(evidence)
            return self._assignment_from_team(
                player_id=player_id,
                fixture_id=fixture_id,
                historical_team_id=historical_team_id,
                current_team_id=current_team_id,
                resolution_method="archived_bootstrap_fixture_compatible",
                endpoint=endpoint,
                raw_path=evidence.raw_path,
                metadata=evidence.metadata,
            )

        if summary is None:
            summary, _, _ = self._load_element_summary(
                player_id,
                required_fixture_ids={fixture_id},
            )
        return self._assignment_from_summary(
            player_id=player_id,
            fixture_id=fixture_id,
            current_team_id=current_team_id,
            summary=summary,
            archived_candidate_teams=sorted(int(team) for team in archived_teams if team is not None),
        )

    def _assignment_from_summary(
        self,
        *,
        player_id: int,
        fixture_id: int,
        current_team_id: int | None,
        summary: dict[str, Any],
        archived_candidate_teams: list[int],
    ) -> HistoricalClubAssignment:
        records = [
            item
            for item in summary.get("history", [])
            if isinstance(item, dict)
            and item.get("fixture") is not None
            and int(item["fixture"]) == fixture_id
        ]
        if len(records) != 1:
            raise self._diagnostic_error(
                player_id,
                fixture_id,
                f"element-summary fixture match count is {len(records)}, expected 1",
                archived_candidate_teams=archived_candidate_teams,
            )
        record = records[0]
        if record.get("element") is not None and int(record["element"]) != player_id:
            raise self._diagnostic_error(
                player_id,
                fixture_id,
                "element-summary element ID contradicts the requested player",
                summary_record=record,
                archived_candidate_teams=archived_candidate_teams,
            )
        if not isinstance(record.get("was_home"), bool):
            raise self._diagnostic_error(
                player_id,
                fixture_id,
                "element-summary was_home evidence is missing or ambiguous",
                summary_record=record,
                archived_candidate_teams=archived_candidate_teams,
            )
        fixture = self.fixtures[fixture_id]
        was_home = bool(record["was_home"])
        historical_team_id = int(fixture["team_h"] if was_home else fixture["team_a"])
        opponent_team_id = int(fixture["team_a"] if was_home else fixture["team_h"])
        if _optional_int(record.get("opponent_team")) != opponent_team_id:
            raise self._diagnostic_error(
                player_id,
                fixture_id,
                "element-summary opponent/home-away evidence contradicts fixture sides",
                summary_record=record,
                archived_candidate_teams=archived_candidate_teams,
            )
        summary_kickoff = pd.to_datetime(record.get("kickoff_time"), utc=True, errors="coerce")
        fixture_kickoff = pd.to_datetime(fixture.get("kickoff_time"), utc=True, errors="coerce")
        if pd.notna(summary_kickoff) and pd.notna(fixture_kickoff) and summary_kickoff != fixture_kickoff:
            raise self._diagnostic_error(
                player_id,
                fixture_id,
                "element-summary kickoff contradicts the fixture snapshot",
                summary_record=record,
                archived_candidate_teams=archived_candidate_teams,
            )
        _, metadata, raw_path = self._summaries[player_id]
        return self._assignment_from_team(
            player_id=player_id,
            fixture_id=fixture_id,
            historical_team_id=historical_team_id,
            current_team_id=current_team_id,
            resolution_method="element_summary_fixture_history",
            endpoint=f"{ELEMENT_SUMMARY}_{player_id}",
            raw_path=raw_path,
            metadata=metadata,
        )

    def _assignment_from_team(
        self,
        *,
        player_id: int,
        fixture_id: int,
        historical_team_id: int,
        current_team_id: int | None,
        resolution_method: str,
        endpoint: str,
        raw_path: str,
        metadata: dict[str, Any],
    ) -> HistoricalClubAssignment:
        fixture = self.fixtures[fixture_id]
        home = int(fixture["team_h"])
        away = int(fixture["team_a"])
        if historical_team_id not in {home, away}:
            raise self._diagnostic_error(
                player_id,
                fixture_id,
                f"resolved team {historical_team_id} is not a fixture participant",
            )
        evidence_time = _utc_timestamp(metadata.get("retrieved_at"))
        if evidence_time >= self.cutoff:
            raise self._diagnostic_error(
                player_id,
                fixture_id,
                f"club evidence was retrieved at {evidence_time.isoformat()}, not before cutoff {self.cutoff.isoformat()}",
            )
        return HistoricalClubAssignment(
            player_id=player_id,
            fixture_id=fixture_id,
            historical_team_id=historical_team_id,
            current_team_id=current_team_id,
            opponent_team_id=away if historical_team_id == home else home,
            was_home=historical_team_id == home,
            resolution_method=resolution_method,
            evidence_endpoint=endpoint,
            evidence_raw_snapshot_path=str(raw_path),
            evidence_sha256=str(metadata.get("sha256") or ""),
            evidence_retrieved_at=str(metadata.get("retrieved_at") or ""),
        )

    def _resolve_identity(
        self,
        player_id: int,
        *,
        summary: dict[str, Any] | None,
        assignments: list[HistoricalClubAssignment],
    ) -> PlayerIdentityEvidence:
        current = self.current_players.get(player_id)
        archived = self._eligible_archived_records(player_id)
        codes = {
            code
            for code in [
                _optional_int(current.get("code")) if current else None,
                *(record.player_code for record in archived),
            ]
            if code is not None
        }
        if summary:
            codes.update(
                int(item["element_code"])
                for item in summary.get("history_past", [])
                if isinstance(item, dict) and item.get("element_code") is not None
            )
        if len(codes) > 1:
            raise HistoricalClubResolutionError(
                f"Conflicting stable player codes for official player {player_id}: {sorted(codes)}."
            )
        player_code = next(iter(codes)) if codes else None
        source_record = current
        identity_source = "current_bootstrap"
        raw_path = self.bootstrap_raw_path
        metadata = self.bootstrap_metadata
        if source_record is None and archived:
            latest = max(archived, key=lambda item: str(item.metadata.get("retrieved_at") or ""))
            source_record = {
                "web_name": latest.player_name,
                "element_type": latest.element_type,
                "team": latest.team_id,
            }
            identity_source = "archived_bootstrap"
            raw_path = latest.raw_path
            metadata = latest.metadata

        player_uid = f"player_code_{player_code}" if player_code is not None else f"official_player_id_{player_id}"
        historical = self.historical_identity.get(player_uid, {})
        player_name = str(
            (source_record or {}).get("web_name")
            or historical.get("player_name")
            or ""
        ).strip()
        element_type = _optional_int((source_record or {}).get("element_type"))
        fpl_position = POSITION_BY_ELEMENT_TYPE.get(element_type) or str(
            historical.get("fpl_position") or ""
        ).strip()
        if not player_name or fpl_position not in set(POSITION_BY_ELEMENT_TYPE.values()):
            fixtures = sorted(assignment.fixture_id for assignment in assignments)
            raise HistoricalClubResolutionError(
                "Historical player identity is insufficient after disappearance from bootstrap-static: "
                f"player={player_id}, fixtures={fixtures}, player_code={player_code}, "
                f"name={player_name!r}, position={fpl_position!r}, evidence="
                "current bootstrap, archived bootstrap snapshots, element-summary history_past, "
                "and persisted Phase 2 identities."
            )
        current_team_id = _optional_int(current.get("team")) if current else None
        return PlayerIdentityEvidence(
            player_id=player_id,
            player_uid=player_uid,
            player_code=player_code,
            player_name=player_name,
            entity_type="assistant_manager" if fpl_position == "AM" else "player",
            fpl_position=fpl_position,
            current_team_id=current_team_id,
            identity_source=identity_source if source_record is not None else "persisted_historical_identity",
            identity_raw_snapshot_path=raw_path if source_record is not None else None,
            identity_sha256=str(metadata.get("sha256") or "") if source_record is not None else None,
            identity_retrieved_at=str(metadata.get("retrieved_at") or "") if source_record is not None else None,
        )

    def _load_element_summary(
        self,
        player_id: int,
        *,
        required_fixture_ids: set[int] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        required_fixture_ids = required_fixture_ids or set()
        loaded = self._summaries.get(player_id)
        if loaded is None:
            loaded = self._latest_cached_summary_before_cutoff(
                player_id,
                required_fixture_ids=required_fixture_ids,
            )
        if loaded is not None and (
            not required_fixture_ids
            or _summary_fixture_ids(loaded[0]).intersection(required_fixture_ids)
        ):
            return self._register_summary(player_id, loaded)

        if self.allow_network and player_id not in self._refreshed_summary_players:
            record = self.client.fetch_element_summary(
                season=self.season,
                player_id=player_id,
                refresh=True,
                offline=False,
                extra_metadata={
                    **self.source_context,
                    "element_id": player_id,
                    "evidence_purpose": "historical_fixture_club_resolution",
                },
            )
            self._refreshed_summary_players.add(player_id)
            metadata = read_metadata(record.raw_path)
            _verify_snapshot_checksum(record.raw_path, metadata)
            loaded = (read_json_snapshot(record.raw_path), metadata, str(record.raw_path))

        if loaded is None:
            raise HistoricalClubResolutionError(
                "No cutoff-eligible cached element-summary evidence is available and network "
                f"retrieval is disabled: player={player_id}, fixtures={sorted(required_fixture_ids)}, "
                f"cutoff={self.cutoff.isoformat()}."
            )
        return self._register_summary(player_id, loaded)

    def _register_summary(
        self,
        player_id: int,
        loaded: tuple[dict[str, Any], dict[str, Any], str],
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        payload, metadata, raw_path = loaded
        retrieved_at = _utc_timestamp(metadata.get("retrieved_at"))
        if retrieved_at >= self.cutoff:
            raise HistoricalClubResolutionError(
                f"Element-summary evidence for player {player_id} was retrieved at "
                f"{retrieved_at.isoformat()}, not before cutoff {self.cutoff.isoformat()}."
            )
        endpoint = f"{ELEMENT_SUMMARY}_{player_id}"
        self.source_hashes[endpoint] = str(metadata.get("sha256") or "")
        self.snapshot_entries[endpoint] = _snapshot_entry(
            metadata,
            raw_path,
            element_id=player_id,
        )
        self.summary_player_ids.add(player_id)
        loaded = (payload, metadata, raw_path)
        self._summaries[player_id] = loaded
        return loaded

    def _latest_cached_summary_before_cutoff(
        self,
        player_id: int,
        *,
        required_fixture_ids: set[int],
    ) -> tuple[dict[str, Any], dict[str, Any], str] | None:
        directory = self.raw_fpl_dir / self.season / f"{ELEMENT_SUMMARY}_{player_id}"
        candidates: list[tuple[pd.Timestamp, dict[str, Any], dict[str, Any], str]] = []
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".metadata.json"):
                continue
            metadata = read_metadata(path)
            _verify_snapshot_checksum(path, metadata)
            retrieved_at = _utc_timestamp(metadata.get("retrieved_at"))
            if retrieved_at >= self.cutoff:
                continue
            payload = read_json_snapshot(path)
            candidates.append((retrieved_at, payload, metadata, str(path)))
        sufficient = [
            item
            for item in candidates
            if not required_fixture_ids
            or _summary_fixture_ids(item[1]).intersection(required_fixture_ids)
        ]
        if not sufficient:
            return None
        _, payload, metadata, raw_path = max(sufficient, key=lambda item: item[0])
        return payload, metadata, raw_path

    def _eligible_archived_records(self, player_id: int) -> list[_ArchivedPlayerRecord]:
        return [
            record
            for record in self.archive.get(player_id, [])
            if _utc_timestamp(record.metadata.get("retrieved_at")) < self.cutoff
        ]

    def _load_bootstrap_archive(self) -> dict[int, list[_ArchivedPlayerRecord]]:
        archive: dict[int, list[_ArchivedPlayerRecord]] = {}
        directory = self.raw_fpl_dir / self.season / BOOTSTRAP_STATIC
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".metadata.json"):
                continue
            metadata = read_metadata(path)
            _verify_snapshot_checksum(path, metadata)
            payload = read_json_snapshot(path)
            try:
                inferred_season = infer_bootstrap_season(payload).inferred_season
            except SeasonIdentityError as exc:
                raise HistoricalClubResolutionError(
                    "Cannot safely scope archived bootstrap-static identity evidence by season: "
                    f"path={path}, requested_season={self.season}, error={exc}."
                ) from exc
            if inferred_season != self.season:
                self.excluded_archive_snapshots.append(
                    {
                        "endpoint": BOOTSTRAP_STATIC,
                        "requested_season": self.season,
                        "inferred_season": inferred_season,
                        "raw_snapshot_path": str(path),
                        "retrieved_at": metadata.get("retrieved_at"),
                        "sha256": metadata.get("sha256"),
                        "reason": "payload season differs from reconstruction season",
                    }
                )
                continue
            elements = _element_lookup(payload.get("elements", []), f"archived bootstrap {path.name}")
            for player_id, element in elements.items():
                archive.setdefault(player_id, []).append(
                    _ArchivedPlayerRecord(
                        player_id=player_id,
                        player_code=_optional_int(element.get("code")),
                        player_name=element.get("web_name")
                        or element.get("second_name")
                        or element.get("first_name"),
                        element_type=_optional_int(element.get("element_type")),
                        team_id=_optional_int(element.get("team")),
                        raw_path=str(path),
                        metadata=metadata,
                    )
                )
        return archive

    def _register_archived_bootstrap(self, record: _ArchivedPlayerRecord) -> str:
        checksum = str(record.metadata.get("sha256") or "")
        if record.raw_path == self.bootstrap_raw_path:
            return BOOTSTRAP_STATIC
        endpoint = f"bootstrap_static_archive_{checksum[:12]}"
        self.source_hashes[endpoint] = checksum
        self.snapshot_entries[endpoint] = _snapshot_entry(record.metadata, record.raw_path)
        return endpoint

    def _diagnostic_error(
        self,
        player_id: int,
        fixture_id: int,
        reason: str,
        *,
        summary_record: dict[str, Any] | None = None,
        archived_candidate_teams: list[int] | None = None,
    ) -> HistoricalClubResolutionError:
        fixture = self.fixtures.get(fixture_id, {})
        current = self.current_players.get(player_id, {})
        return HistoricalClubResolutionError(
            "Ambiguous or contradictory historical club evidence: "
            f"player={player_id}, fixture={fixture_id}, "
            f"candidate_clubs={[fixture.get('team_h'), fixture.get('team_a')]}, "
            f"current_club={current.get('team')}, "
            f"archived_candidate_clubs={archived_candidate_teams or []}, "
            f"summary_record={summary_record}, reason={reason}."
        )


def _fixture_lookup(fixtures: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    ids = [int(item["id"]) for item in fixtures if isinstance(item, dict) and item.get("id") is not None]
    if len(ids) != len(fixtures) or len(ids) != len(set(ids)):
        raise HistoricalClubResolutionError("Official fixtures contain missing or duplicate IDs.")
    return {int(item["id"]): item for item in fixtures}


def _element_lookup(elements: Any, source: str) -> dict[int, dict[str, Any]]:
    if not isinstance(elements, list):
        raise HistoricalClubResolutionError(f"{source} elements must be a list.")
    ids = [int(item["id"]) for item in elements if isinstance(item, dict) and item.get("id") is not None]
    if len(ids) != len(elements) or len(ids) != len(set(ids)):
        raise HistoricalClubResolutionError(f"{source} contains missing or duplicate player IDs.")
    return {int(item["id"]): item for item in elements}


def _historical_identity_lookup(normalized_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(normalized_dir) / "phase2" / "fact_player_fixture.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    required = {"player_uid", "player_name", "fpl_position", "source_available_time"}
    if not required.issubset(frame.columns):
        return {}
    entity_type = (
        frame["entity_type"]
        if "entity_type" in frame.columns
        else pd.Series("player", index=frame.index)
    )
    latest = (
        frame.loc[entity_type.eq("player")]
        .sort_values("source_available_time")
        .groupby("player_uid", as_index=False)
        .tail(1)
    )
    return latest.set_index("player_uid")[["player_name", "fpl_position"]].to_dict("index")


def _snapshot_entry(
    metadata: dict[str, Any],
    raw_path: str,
    *,
    element_id: int | None = None,
) -> dict[str, Any]:
    entry = {
        "endpoint": metadata.get("source_url"),
        "retrieved_at": metadata.get("retrieved_at"),
        "sha256": metadata.get("sha256"),
        "bytes": metadata.get("content_length"),
        "source_mode": "official_current_season",
        "raw_snapshot_path": raw_path,
    }
    if element_id is not None:
        entry["element_id"] = element_id
    return entry


def _verify_snapshot_checksum(path: Path, metadata: dict[str, Any]) -> None:
    expected = str(metadata.get("sha256") or "")
    actual = sha256_bytes(path.read_bytes())
    if expected != actual:
        raise HistoricalClubResolutionError(
            f"Cached official snapshot checksum mismatch for {path}: expected {expected}, got {actual}."
        )


def _optional_int(value: Any) -> int | None:
    if value is None or value is pd.NA:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_fixture_ids(payload: dict[str, Any]) -> set[int]:
    return {
        int(item["fixture"])
        for item in payload.get("history", [])
        if isinstance(item, dict) and item.get("fixture") is not None
    }


def _utc_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if not isinstance(parsed, pd.Timestamp) or pd.isna(parsed):
        raise HistoricalClubResolutionError(f"Official evidence timestamp is missing or malformed: {value!r}.")
    return parsed
