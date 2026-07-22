from __future__ import annotations

from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def fixture_bytes(relative_path: str) -> bytes:
    return (FIXTURES_DIR / relative_path).read_bytes()


def fixture_text(relative_path: str) -> str:
    return (FIXTURES_DIR / relative_path).read_text(encoding="utf-8")
