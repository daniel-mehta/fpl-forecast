from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FPL_API_DIR = RAW_DIR / "fpl_api"
RAW_VAASTAV_DIR = RAW_DIR / "vaastav"
NORMALIZED_DIR = DATA_DIR / "normalized"
MANUAL_DIR = DATA_DIR / "manual"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SYNTHETIC_DEMO_DIR = OUTPUTS_DIR / "synthetic_demo"

FPL_API_BASE_URL = "https://fantasy.premierleague.com/api/"
VAASTAV_GITHUB_API_BASE_URL = "https://api.github.com/repos/vaastav/Fantasy-Premier-League"
VAASTAV_RAW_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League"

USER_AGENT = "fpl-forecast/0.1 (+https://github.com/local/fpl-forecast)"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 0.5

POSITION_BY_ID = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
    5: "AM",
}
VALID_POSITIONS = frozenset(POSITION_BY_ID.values())


class RuntimePaths(BaseModel):
    project_root: Path = PROJECT_ROOT
    raw_fpl_api_dir: Path = RAW_FPL_API_DIR
    raw_vaastav_dir: Path = RAW_VAASTAV_DIR
    normalized_dir: Path = NORMALIZED_DIR
    manual_dir: Path = MANUAL_DIR

    model_config = {"arbitrary_types_allowed": True}


class HttpSettings(BaseModel):
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    retries: int = Field(default=DEFAULT_RETRIES, ge=0, le=10)
    backoff_seconds: float = Field(default=DEFAULT_BACKOFF_SECONDS, ge=0)
    user_agent: str = USER_AGENT


def ensure_data_directories(paths: RuntimePaths | None = None) -> None:
    paths = paths or RuntimePaths()
    for directory in (
        paths.raw_fpl_api_dir,
        paths.raw_vaastav_dir,
        paths.normalized_dir,
        paths.manual_dir,
        SYNTHETIC_DEMO_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
