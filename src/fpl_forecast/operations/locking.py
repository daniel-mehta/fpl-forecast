from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fpl_forecast.operations.config import LOCK_PATH


class RefreshLockError(RuntimeError):
    pass


@dataclass
class RefreshLock:
    path: Path = LOCK_PATH
    stale_seconds: int = 3600

    def __enter__(self) -> "RefreshLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(self.path, flags)
        except FileExistsError as exc:
            raise RefreshLockError(f"Operational refresh is already locked by {self.path}.") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
