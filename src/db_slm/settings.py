from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Minimal .env parser (no external dependency required)."""
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


@dataclass(frozen=True)
class DBSLMSettings:
    backend: str
    sqlite_path: str
    dataset_path: str
    mariadb_host: str
    mariadb_port: int
    mariadb_user: str
    mariadb_password: str
    mariadb_database: str
    env_file: Path | None

    def sqlite_dsn(self) -> str:
        """Return the SQLite DSN currently used by the CLI utilities."""
        return self.sqlite_path

    def mariadb_dsn(self) -> str:
        """Return a DSN-like string for the MariaDB target (advisory for now)."""
        password = self.mariadb_password.replace("@", "%40")
        return (
            f"mariadb://{self.mariadb_user}:{password}@"
            f"{self.mariadb_host}:{self.mariadb_port}/{self.mariadb_database}"
        )


def load_settings(env_path: str | Path = ".env") -> DBSLMSettings:
    """Load DBSLM settings from .env (if present) + real environment."""
    env_file = Path(env_path)
    file_values = _parse_env_file(env_file)

    def read(key: str, default: str) -> str:
        return os.environ.get(key, file_values.get(key, default))

    backend = read("DBSLM_BACKEND", "sqlite").lower()
    sqlite_path = read("DBSLM_SQLITE_PATH", "var/db_slm.sqlite3")
    dataset_path = read("DBSLM_DATASET_PATH", "datasets/emotion_data.json")
    mariadb_host = read("DBSLM_MARIADB_HOST", "127.0.0.1")
    mariadb_port = int(read("DBSLM_MARIADB_PORT", "3306"))
    mariadb_user = read("DBSLM_MARIADB_USER", "dbslm")
    mariadb_password = read("DBSLM_MARIADB_PASSWORD", "change-me")
    mariadb_database = read("DBSLM_MARIADB_DATABASE", "db_slm")

    env_file_used = env_file if env_file.exists() else None
    return DBSLMSettings(
        backend=backend,
        sqlite_path=sqlite_path,
        dataset_path=dataset_path,
        mariadb_host=mariadb_host,
        mariadb_port=mariadb_port,
        mariadb_user=mariadb_user,
        mariadb_password=mariadb_password,
        mariadb_database=mariadb_database,
        env_file=env_file_used,
    )
