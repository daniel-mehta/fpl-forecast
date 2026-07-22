from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fpl_forecast.config import NORMALIZED_DIR
from fpl_forecast.features.build import FeatureBuildResult, build_features_player_fixture
from fpl_forecast.panel.common import parse_seasons
from fpl_forecast.panel.facts import FactBuildResult, build_fact_player_fixture
from fpl_forecast.panel.fixtures import FixtureBuildResult, build_fixture_dimension
from fpl_forecast.panel.players import PlayerIdentityResult, build_player_identities
from fpl_forecast.panel.teams import TeamIdentityResult, build_team_identities


@dataclass(frozen=True)
class IdentityBuildResult:
    teams: TeamIdentityResult
    players: PlayerIdentityResult


@dataclass(frozen=True)
class PanelBuildResult:
    fixtures: FixtureBuildResult
    facts: FactBuildResult
    features: FeatureBuildResult


def build_identities(
    *,
    seasons: str | list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> IdentityBuildResult:
    season_list = parse_seasons(seasons)
    teams = build_team_identities(seasons=season_list, normalized_dir=normalized_dir)
    players = build_player_identities(seasons=season_list, normalized_dir=normalized_dir)
    return IdentityBuildResult(teams=teams, players=players)


def build_panel(
    *,
    seasons: str | list[str],
    normalized_dir: Path | str = NORMALIZED_DIR,
) -> PanelBuildResult:
    season_list = parse_seasons(seasons)
    fixtures = build_fixture_dimension(seasons=season_list, normalized_dir=normalized_dir)
    facts = build_fact_player_fixture(seasons=season_list, normalized_dir=normalized_dir)
    features = build_features_player_fixture(seasons=season_list, normalized_dir=normalized_dir)
    return PanelBuildResult(fixtures=fixtures, facts=facts, features=features)
