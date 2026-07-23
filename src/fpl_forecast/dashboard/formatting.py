from __future__ import annotations

import html
from typing import Any


def esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return esc(value)
