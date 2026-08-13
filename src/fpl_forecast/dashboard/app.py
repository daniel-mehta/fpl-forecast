from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fpl_forecast.config import OUTPUTS_DIR
from fpl_forecast.dashboard.data import load_dashboard_data
from fpl_forecast.dashboard.views import render_html
from fpl_forecast.operations.config import LATEST_SUCCESSFUL_PATH


DASHBOARD_DIR = OUTPUTS_DIR / "operational" / "dashboard"
DASHBOARD_HTML = DASHBOARD_DIR / "index.html"


def build_dashboard_html(
    *,
    pointer_path: Path = LATEST_SUCCESSFUL_PATH,
    output_path: Path = DASHBOARD_HTML,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_dashboard_data(pointer_path)
    output_path.write_text(render_html(data), encoding="utf-8")
    return output_path


def run_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    smoke: bool = False,
    pointer_path: Path = LATEST_SUCCESSFUL_PATH,
    output_path: Path = DASHBOARD_HTML,
) -> Path:
    path = build_dashboard_html(pointer_path=pointer_path, output_path=output_path)
    if smoke:
        return path

    def handler(*args, **kwargs):
        return SimpleHTTPRequestHandler(*args, directory=str(path.parent), **kwargs)

    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dashboard available at http://{host}:{port}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return path
