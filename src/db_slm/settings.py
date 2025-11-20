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
    quality_queue_path: str
    env_file: Path | None
    embedder_model: str
    cheetah_host: str
    cheetah_port: int
    cheetah_database: str
    cheetah_timeout_seconds: float
    cheetah_idle_grace_seconds: float
    cheetah_mirror: bool
    tokenizer_backend: str
    tokenizer_json_path: str | None
    tokenizer_lowercase: bool

    def sqlite_dsn(self) -> str:
        """Return the SQLite DSN currently used by the CLI utilities."""
        return self.sqlite_path


def load_settings(env_path: str | Path = ".env") -> DBSLMSettings:
    """Load DBSLM settings from .env (if present) + real environment."""
    env_file = Path(env_path)
    file_values = _parse_env_file(env_file)

    def read(key: str, default: str) -> str:
        return os.environ.get(key, file_values.get(key, default))

    backend = read("DBSLM_BACKEND", "cheetah-db").lower()
    sqlite_path = read("DBSLM_SQLITE_PATH", "var/db_slm.sqlite3")
    dataset_path = read("DBSLM_DATASET_PATH", "datasets/emotion_data.json")
    quality_queue_path = read(
        "DBSLM_QUALITY_QUEUE_PATH", "var/eval_logs/quality_retrain_queue.jsonl"
    )
    embedder_model = read("DBSLM_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
    cheetah_host = read("DBSLM_CHEETAH_HOST", "127.0.0.1")
    cheetah_port = int(read("DBSLM_CHEETAH_PORT", "4455"))
    cheetah_database = read("DBSLM_CHEETAH_DATABASE", "default")
    cheetah_timeout_seconds = float(read("DBSLM_CHEETAH_TIMEOUT_SECONDS", "1.0"))
    idle_grace_raw = read("DBSLM_CHEETAH_IDLE_GRACE_SECONDS", "").strip()
    if idle_grace_raw:
        cheetah_idle_grace_seconds = max(0.0, float(idle_grace_raw))
    else:
        cheetah_idle_grace_seconds = max(cheetah_timeout_seconds * 180.0, 60.0)
    mirror_flag = read("DBSLM_CHEETAH_MIRROR", "0").lower()
    cheetah_mirror = mirror_flag in {"1", "true", "yes", "on"}
    tokenizer_backend = read("DBSLM_TOKENIZER_BACKEND", "regex").strip().lower()
    tokenizer_json_raw = read("DBSLM_TOKENIZER_JSON", "").strip()
    tokenizer_json_path = tokenizer_json_raw or None
    tokenizer_lower_flag = read("DBSLM_TOKENIZER_LOWERCASE", "1").strip().lower()
    tokenizer_lowercase = tokenizer_lower_flag not in {"0", "false", "no", "off"}

    env_file_used = env_file if env_file.exists() else None
    return DBSLMSettings(
        backend=backend,
        sqlite_path=sqlite_path,
        dataset_path=dataset_path,
        quality_queue_path=quality_queue_path,
        env_file=env_file_used,
        embedder_model=embedder_model,
        cheetah_host=cheetah_host,
        cheetah_port=cheetah_port,
        cheetah_database=cheetah_database,
        cheetah_timeout_seconds=cheetah_timeout_seconds,
        cheetah_idle_grace_seconds=cheetah_idle_grace_seconds,
        cheetah_mirror=cheetah_mirror,
        tokenizer_backend=tokenizer_backend,
        tokenizer_json_path=tokenizer_json_path,
        tokenizer_lowercase=tokenizer_lowercase,
    )
