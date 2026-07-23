from __future__ import annotations

from fpl_forecast.operations.publication import latest_successful
from fpl_forecast.operations.state import OperationalStatus, read_status


def operational_status() -> tuple[OperationalStatus | None, dict | None]:
    return read_status(), latest_successful()
