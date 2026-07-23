from __future__ import annotations

import pandas as pd

from fpl_forecast.dashboard.data import DashboardData
from fpl_forecast.dashboard.formatting import esc


def render_html(data: DashboardData) -> str:
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'><title>FPL Forecast Operational Dashboard</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.4}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px;font-size:13px}th{background:#f4f4f4}.warn{background:#fff3cd;padding:10px;border:1px solid #e6c86e}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}</style>",
            "</head><body>",
            "<h1>FPL Forecast</h1>",
            _overview(data),
            "<div class='grid'><section><h2>Recommended Squad</h2>",
            _table(data.squad.head(15), ["player_name", "fpl_position", "player_team_uid", "price_tenths", "expected_points", "selected_role"]),
            "</section><section><h2>Lineup And Captaincy</h2>",
            _table(data.lineup, ["formation", "captain", "vice_captain", "cost_tenths", "bank_tenths", "objective", "solver_status"]),
            "</section></div>",
            "<h2>Player Projections</h2>",
            _table(data.projections, ["player", "team", "position", "price_tenths", "expected_points", "expected_minutes", "p_appearance", "p_start", "prob_points_ge_5", "prob_points_ge_10", "status", "news", "cold_start_no_history", "model_variant"]),
            "<h2>Model Comparison</h2>",
            _table(data.comparison, list(data.comparison.columns)),
            "<h2>Methodology And Limitations</h2>",
            "<p>xPoints are expected FPL points from the local model stack. Expected minutes and appearance probability matter because non-appearances change both captaincy and autosub outcomes. The optimizer is exact for the displayed weekly-reset forecast and constraints, not a guarantee of real-world points. Transfer planning is disabled for full live use until manager-specific bank, purchase prices and free transfers are supplied.</p>",
            "</body></html>",
        ]
    )


def _overview(data: DashboardData) -> str:
    status = data.status
    warning = status.get("warning") or data.error or ""
    return "\n".join(
        [
            "<section><h2>Operational Overview</h2>",
            f"<p><strong>State:</strong> {esc(status.get('state'))}</p>",
            f"<p><strong>Target season:</strong> {esc(status.get('target_season'))}</p>",
            f"<p><strong>Run directory:</strong> {esc(data.run_dir)}</p>",
            f"<p><strong>Freshness:</strong> {esc(data.freshness.get('generated_at'))}</p>",
            f"<p><strong>Source:</strong> {esc(data.freshness.get('source'))}</p>",
            f"<p><strong>Team model run:</strong> {esc(data.freshness.get('team_model_run_id'))}</p>",
            f"<p><strong>Reason:</strong> {esc(status.get('reason'))}</p>",
            f"<div class='warn'>{esc(warning)}</div>" if warning else "",
            "</section>",
        ]
    )


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "<p>No rows available.</p>"
    present = [column for column in columns if column in frame.columns]
    if not present:
        return "<p>No displayable columns available.</p>"
    rows = ["<table><thead><tr>", *[f"<th>{esc(column)}</th>" for column in present], "</tr></thead><tbody>"]
    for record in frame[present].head(100).fillna("").to_dict(orient="records"):
        rows.append("<tr>")
        rows.extend(f"<td>{esc(record[column])}</td>" for column in present)
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)
