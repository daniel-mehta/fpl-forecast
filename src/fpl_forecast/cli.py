from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fpl_forecast.config import (
    NORMALIZED_DIR,
    RAW_FPL_API_DIR,
    RAW_VAASTAV_DIR,
    ensure_data_directories,
)
from fpl_forecast.backtest.runner import (
    compare_baselines as compare_baseline_run,
    inspect_backtest_run,
    run_baseline_backtest,
)
from fpl_forecast.team_model.runner import (
    compare_team_models,
    forecast_team_fixtures as forecast_team_fixtures_run,
    inspect_team_run,
    run_team_backtest,
)
from fpl_forecast.minutes_model.runner import (
    compare_minutes_models,
    forecast_minutes as forecast_minutes_run,
    inspect_minutes_run,
    run_minutes_backtest,
)
from fpl_forecast.xpoints.runner import (
    compare_xpoints as compare_xpoints_run,
    forecast_xpoints as forecast_xpoints_run,
    inspect_xpoints as inspect_xpoints_run,
    run_xpoints_backtest,
    validate_scoring as validate_scoring_run,
)
from fpl_forecast.ingest.fpl_api import FPLApiClient, FPLApiError
from fpl_forecast.ingest.vaastav import VaastavDataError, VaastavIngestor
from fpl_forecast.normalize.current import normalize_current as normalize_current_tables
from fpl_forecast.normalize.historical import normalize_historical as normalize_historical_tables
from fpl_forecast.features.leakage import audit_leakage as audit_leakage_tables
from fpl_forecast.panel.build import build_identities as build_identity_tables
from fpl_forecast.panel.build import build_panel as build_panel_tables
from fpl_forecast.panel.common import parse_seasons
from fpl_forecast.panel.inspect import inspect_panel as inspect_panel_tables
from fpl_forecast.validation.data_quality import ERROR, validate_all


app = typer.Typer(
    help="Real-data foundation commands for FPL Forecast.",
    no_args_is_help=True,
)
console = Console()


@app.command("snapshot-current")
def snapshot_current(
    season: Annotated[str, typer.Option(help="Target season label, for example 2026-27.")],
    refresh: Annotated[bool, typer.Option(help="Fetch fresh snapshots even when cache exists.")] = False,
    offline: Annotated[bool, typer.Option(help="Use cached snapshots only.")] = False,
    raw_dir: Annotated[
        Path,
        typer.Option(help="Raw FPL API directory."),
    ] = RAW_FPL_API_DIR,
) -> None:
    ensure_data_directories()
    client = FPLApiClient(raw_dir=raw_dir)
    try:
        records = client.snapshot_current(season=season, refresh=refresh, offline=offline)
    except FPLApiError as exc:
        console.print(f"[red]FPL snapshot failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_records("FPL API snapshots", records)


@app.command("ingest-historical")
def ingest_historical(
    season: Annotated[str, typer.Option(help="Historical season label, for example 2024-25.")] = "2024-25",
    refresh: Annotated[bool, typer.Option(help="Fetch fresh CSVs even when cache exists.")] = False,
    revision: Annotated[
        str | None,
        typer.Option(help="Specific Vaastav Git revision to fetch. Defaults to master HEAD."),
    ] = None,
    raw_dir: Annotated[
        Path,
        typer.Option(help="Raw Vaastav directory."),
    ] = RAW_VAASTAV_DIR,
) -> None:
    ensure_data_directories()
    ingestor = VaastavIngestor(raw_dir=raw_dir)
    try:
        result = ingestor.ingest_season(season=season, refresh=refresh, revision=revision)
    except VaastavDataError as exc:
        console.print(f"[red]Historical ingest failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_records("Vaastav snapshots", result.records)
    for warning in result.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"source_version={result.source_version}")


@app.command("normalize-current")
def normalize_current(
    season: Annotated[str, typer.Option(help="Season label to normalize.")],
    raw_dir: Annotated[
        Path,
        typer.Option(help="Raw FPL API directory."),
    ] = RAW_FPL_API_DIR,
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized output directory."),
    ] = NORMALIZED_DIR,
) -> None:
    try:
        outputs = normalize_current_tables(
            season=season,
            raw_dir=raw_dir,
            normalized_dir=normalized_dir,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Current normalization failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_paths("Normalized current tables", outputs)


@app.command("normalize-historical")
def normalize_historical(
    season: Annotated[str, typer.Option(help="Historical season label to normalize.")] = "2024-25",
    raw_dir: Annotated[
        Path,
        typer.Option(help="Raw Vaastav directory."),
    ] = RAW_VAASTAV_DIR,
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized output directory."),
    ] = NORMALIZED_DIR,
) -> None:
    try:
        outputs = normalize_historical_tables(
            season=season,
            raw_dir=raw_dir,
            normalized_dir=normalized_dir,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Historical normalization failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_paths("Normalized historical tables", outputs)


@app.command("validate-data")
def validate_data(
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
    raw_vaastav_dir: Annotated[
        Path,
        typer.Option(help="Raw Vaastav directory."),
    ] = RAW_VAASTAV_DIR,
) -> None:
    result = validate_all(normalized_dir=normalized_dir, raw_vaastav_dir=raw_vaastav_dir)
    if not result.issues:
        console.print("[green]No data-quality issues found.[/green]")
        return

    table = Table(title="Data-quality issues")
    table.add_column("Severity")
    table.add_column("Table")
    table.add_column("Message")
    for issue in result.issues:
        style = "red" if issue.severity == ERROR else "yellow"
        table.add_row(f"[{style}]{issue.severity}[/{style}]", issue.table, issue.message)
    console.print(table)

    if result.errors:
        raise typer.Exit(1)
    if result.warnings:
        console.print(f"[yellow]{len(result.warnings)} warning(s), 0 errors.[/yellow]")


@app.command("build-identities")
def build_identities(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated historical seasons, for example 2022-23,2023-24,2024-25."),
    ],
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
) -> None:
    try:
        result = build_identity_tables(seasons=seasons, normalized_dir=normalized_dir)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Identity build failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_paths(
        "Identity tables",
        [
            result.teams.dim_team_path,
            result.teams.team_season_map_path,
            result.players.dim_player_path,
            result.players.player_season_map_path,
            result.players.review_path,
        ],
    )
    _print_identity_counts(result.players.player_season_map)


@app.command("build-panel")
def build_panel(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated historical seasons, for example 2022-23,2023-24,2024-25."),
    ],
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
) -> None:
    try:
        result = build_panel_tables(seasons=seasons, normalized_dir=normalized_dir)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Panel build failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_paths(
        "Panel tables",
        [result.fixtures.fixture_path, result.facts.fact_path, result.features.feature_path],
    )


@app.command("audit-leakage")
def audit_leakage(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated historical seasons, for example 2022-23,2023-24,2024-25."),
    ],
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
) -> None:
    season_list = parse_seasons(seasons)
    result = audit_leakage_tables(seasons=season_list, normalized_dir=normalized_dir)
    if not result.issues:
        console.print("[green]Leakage audit passed.[/green]")
        return
    table = Table(title="Leakage audit issues")
    table.add_column("Severity")
    table.add_column("Message")
    for issue in result.issues:
        style = "red" if issue.severity == "error" else "yellow"
        table.add_row(f"[{style}]{issue.severity}[/{style}]", issue.message)
    console.print(table)
    if result.errors:
        raise typer.Exit(1)


@app.command("inspect-panel")
def inspect_panel(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated historical seasons, for example 2022-23,2023-24,2024-25."),
    ],
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
) -> None:
    try:
        result = inspect_panel_tables(
            seasons=parse_seasons(seasons),
            normalized_dir=normalized_dir,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Panel inspection failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in result.lines:
        console.print(line)


@app.command("backtest-baselines")
def backtest_baselines(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated seasons available for training and testing."),
    ],
    test_seasons: Annotated[
        str,
        typer.Option(help="Comma-separated seasons to score out of sample."),
    ],
    mode: Annotated[str, typer.Option(help="Backtest mode: rolling or gw1.")] = "rolling",
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
    run_id: Annotated[str | None, typer.Option(help="Optional deterministic run id.")] = None,
    bootstrap_samples: Annotated[
        int | None,
        typer.Option(help="Override bootstrap sample count."),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Override bootstrap random seed.")] = None,
) -> None:
    try:
        result = run_baseline_backtest(
            seasons=seasons,
            test_seasons=test_seasons,
            mode=mode,
            normalized_dir=normalized_dir,
            run_id=run_id,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Baseline backtest failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"run_id={result.run_id}")
    console.print(f"run_dir={result.run_dir}")
    console.print(f"folds={len(result.folds)}")
    console.print(f"frozen_predictions={result.frozen_predictions_path}")
    console.print(f"scored_fixture_predictions={result.scored_fixture_path}")
    console.print(f"scored_player_gameweek_predictions={result.player_gameweek_path}")
    console.print(f"manifest={result.manifest_path}")


@app.command("compare-baselines")
def compare_baselines(
    run_id: Annotated[str, typer.Option(help="Backtest run id to compare.")],
) -> None:
    try:
        lines = compare_baseline_run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Baseline comparison failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


@app.command("inspect-backtest")
def inspect_backtest(
    run_id: Annotated[str, typer.Option(help="Backtest run id to inspect.")],
) -> None:
    try:
        lines = inspect_backtest_run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Backtest inspection failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


@app.command("backtest-team-model")
def backtest_team_model(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated historical seasons available for training and testing."),
    ],
    test_seasons: Annotated[
        str,
        typer.Option(help="Comma-separated seasons to score out of sample."),
    ],
    mode: Annotated[str, typer.Option(help="Backtest mode: rolling or gw1.")] = "rolling",
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
    run_id: Annotated[str | None, typer.Option(help="Optional deterministic run id.")] = None,
) -> None:
    try:
        result = run_team_backtest(
            seasons=seasons,
            test_seasons=test_seasons,
            mode=mode,
            normalized_dir=normalized_dir,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Team-model backtest failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"run_id={result.run_id}")
    console.print(f"run_dir={result.run_dir}")
    console.print(f"folds={len(result.folds)}")
    console.print(f"frozen_predictions={result.frozen_predictions_path}")
    console.print(f"scored_predictions={result.scored_predictions_path}")
    console.print(f"manifest={result.manifest_path}")


@app.command("compare-team-models")
def compare_team_model_runs(
    run_id: Annotated[str, typer.Option(help="Team-model run id to compare.")],
) -> None:
    try:
        lines = compare_team_models(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Team-model comparison failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


@app.command("inspect-team-model")
def inspect_team_model(
    run_id: Annotated[str, typer.Option(help="Team-model run id to inspect.")],
) -> None:
    try:
        lines = inspect_team_run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Team-model inspection failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


@app.command("forecast-team-fixtures")
def forecast_team_fixtures(
    season: Annotated[str, typer.Option(help="Official current season to forecast.")],
    as_of: Annotated[str, typer.Option(help="UTC cutoff timestamp for the forecast.")],
    gameweek: Annotated[int | None, typer.Option(help="Optional FPL gameweek/event filter.")] = None,
    seasons: Annotated[
        str,
        typer.Option(help="Historical seasons to train from."),
    ] = "2022-23,2023-24,2024-25",
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
    run_id: Annotated[str | None, typer.Option(help="Optional deterministic run id.")] = None,
) -> None:
    try:
        path = forecast_team_fixtures_run(
            season=season,
            gameweek=gameweek,
            as_of=as_of,
            seasons=seasons,
            normalized_dir=normalized_dir,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Team-fixture forecast failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"future_fixture_predictions={path}")


@app.command("backtest-minutes")
def backtest_minutes(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated historical seasons available for training and testing."),
    ],
    test_seasons: Annotated[
        str,
        typer.Option(help="Comma-separated seasons to score out of sample."),
    ],
    mode: Annotated[str, typer.Option(help="Backtest mode: rolling or gw1.")] = "rolling",
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
    run_id: Annotated[str | None, typer.Option(help="Optional deterministic run id.")] = None,
    bootstrap_samples: Annotated[
        int | None,
        typer.Option(help="Override bootstrap sample count."),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Override bootstrap random seed.")] = None,
) -> None:
    try:
        result = run_minutes_backtest(
            seasons=seasons,
            test_seasons=test_seasons,
            mode=mode,
            normalized_dir=normalized_dir,
            run_id=run_id,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Minutes backtest failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"run_id={result.run_id}")
    console.print(f"run_dir={result.run_dir}")
    console.print(f"folds={len(result.folds)}")
    console.print(f"frozen_predictions={result.frozen_predictions_path}")
    console.print(f"scored_predictions={result.scored_predictions_path}")
    console.print(f"manifest={result.manifest_path}")


@app.command("compare-minutes")
def compare_minutes(
    run_id: Annotated[str, typer.Option(help="Minutes run id to compare.")],
) -> None:
    try:
        lines = compare_minutes_models(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Minutes comparison failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


@app.command("inspect-minutes")
def inspect_minutes(
    run_id: Annotated[str, typer.Option(help="Minutes run id to inspect.")],
) -> None:
    try:
        lines = inspect_minutes_run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Minutes inspection failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


@app.command("forecast-minutes")
def forecast_minutes(
    season: Annotated[str, typer.Option(help="Official current season to forecast.")],
    as_of: Annotated[str, typer.Option(help="UTC cutoff timestamp for the forecast.")],
    gameweek: Annotated[int | None, typer.Option(help="Optional FPL gameweek/event filter.")] = None,
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
) -> None:
    try:
        path = forecast_minutes_run(
            season=season,
            gameweek=gameweek,
            as_of=as_of,
            normalized_dir=normalized_dir,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Minutes forecast failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"future_minutes_predictions={path}")


@app.command("validate-scoring")
def validate_scoring(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated historical seasons to audit."),
    ] = "2022-23,2023-24,2024-25",
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
    run_id: Annotated[str, typer.Option(help="Scoring-audit run id.")] = "scoring_reconstruction",
) -> None:
    try:
        outputs = validate_scoring_run(seasons=seasons, normalized_dir=normalized_dir, run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Scoring validation failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_paths("Scoring reconstruction audit", list(outputs.values()))


@app.command("backtest-xpoints")
def backtest_xpoints(
    seasons: Annotated[
        str,
        typer.Option(help="Comma-separated seasons available for training and testing."),
    ],
    test_seasons: Annotated[
        str,
        typer.Option(help="Comma-separated seasons to score out of sample."),
    ],
    mode: Annotated[str, typer.Option(help="Backtest mode: rolling or gw1.")] = "rolling",
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
    run_id: Annotated[str | None, typer.Option(help="Optional deterministic run id.")] = None,
) -> None:
    try:
        result = run_xpoints_backtest(
            seasons=seasons,
            test_seasons=test_seasons,
            mode=mode,
            normalized_dir=normalized_dir,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]xPoints backtest failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"run_id={result.run_id}")
    console.print(f"run_dir={result.run_dir}")
    console.print(f"folds={len(result.folds)}")
    console.print(f"frozen_predictions={result.frozen_predictions_path}")
    console.print(f"scored_predictions={result.scored_predictions_path}")
    console.print(f"player_gameweek_predictions={result.player_gameweek_path}")
    console.print(f"manifest={result.manifest_path}")


@app.command("compare-xpoints")
def compare_xpoints(
    run_id: Annotated[str, typer.Option(help="xPoints run id to compare.")],
) -> None:
    try:
        lines = compare_xpoints_run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]xPoints comparison failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


@app.command("inspect-xpoints")
def inspect_xpoints(
    run_id: Annotated[str, typer.Option(help="xPoints run id to inspect.")],
) -> None:
    try:
        lines = inspect_xpoints_run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]xPoints inspection failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    for line in lines:
        console.print(line)


@app.command("forecast-xpoints")
def forecast_xpoints(
    season: Annotated[str, typer.Option(help="Official current season to forecast.")],
    as_of: Annotated[str, typer.Option(help="UTC cutoff timestamp for the forecast.")],
    gameweek: Annotated[int | None, typer.Option(help="Optional FPL gameweek/event filter.")] = None,
    normalized_dir: Annotated[
        Path,
        typer.Option(help="Normalized data directory."),
    ] = NORMALIZED_DIR,
) -> None:
    try:
        path = forecast_xpoints_run(
            season=season,
            gameweek=gameweek,
            as_of=as_of,
            normalized_dir=normalized_dir,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]xPoints forecast failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"future_xpoints_predictions={path}")


def _print_records(title: str, records) -> None:
    table = Table(title=title)
    table.add_column("Endpoint")
    table.add_column("Retrieved At")
    table.add_column("Bytes", justify="right")
    table.add_column("SHA-256")
    table.add_column("Path")
    for record in records:
        table.add_row(
            record.endpoint_name,
            record.retrieved_at,
            str(record.content_length),
            record.checksum_sha256[:16],
            str(record.raw_path),
        )
    console.print(table)


def _print_paths(title: str, paths: list[Path]) -> None:
    table = Table(title=title)
    table.add_column("Path")
    for path in paths:
        table.add_row(str(path))
    console.print(table)


def _print_identity_counts(player_season_map) -> None:
    table = Table(title="Player identity counts")
    table.add_column("Season")
    table.add_column("Method")
    table.add_column("Count", justify="right")
    counts = player_season_map.groupby(["season", "match_method"]).size().reset_index(name="count")
    for row in counts.itertuples(index=False):
        table.add_row(str(row.season), str(row.match_method), str(row.count))
    console.print(table)
