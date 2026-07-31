"""Import bridge to the generic Cheetah client that lives in the submodule.

The protocol half of talking to cheetah-db — command encoding, response
parsing, socket lifecycle, the KV/graph/job/prediction/admin call shapes — is
generic Cheetah work and therefore owned by the Cheetah repository, in
``cheetah-db/binders/python``. DB-SLM keeps only what is DB-SLM: the fixed-size
payload serializer, the namespace conventions, the projections in
``cheetah_types``, and the hot-path adapter.

The binder is not installed as a package (it is not on PyPI, and vendoring it
would recreate the duplication this bridge exists to remove), so this module
puts the submodule directory on ``sys.path`` once and re-exports what the
adapter uses. ``DBSLM_CHEETAH_BINDER_PATH`` overrides the location for a
checkout that keeps the Cheetah repository elsewhere.

If the submodule has never been initialized the import fails with the command
that fixes it, rather than with a bare ``ModuleNotFoundError`` naming a package
that was never meant to be installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BINDER_PATH = _REPOSITORY_ROOT / "cheetah-db" / "binders" / "python"


def binder_path() -> Path:
    override = os.environ.get("DBSLM_CHEETAH_BINDER_PATH", "").strip()
    return Path(override).expanduser() if override else _DEFAULT_BINDER_PATH


def _ensure_importable() -> None:
    path = binder_path()
    if not (path / "cheetah_db" / "__init__.py").exists():
        raise ImportError(
            f"the Cheetah Python binder is not present at {path}. It ships with the "
            "cheetah-db submodule: run `git submodule update --init --recursive`, or set "
            "DBSLM_CHEETAH_BINDER_PATH to a checkout that holds binders/python."
        )
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


_ensure_importable()

from cheetah_db import admin, graph, jobs, kv, predict, protocol  # noqa: E402
from cheetah_db.client import (  # noqa: E402
    CheetahClient as BinderCheetahClient,
    CheetahConnectionError,
    CheetahError,
    ThreadLocalClientPool,
)
from cheetah_db.protocol import (  # noqa: E402
    RawArgument,
    Response,
    ScanItem,
    build_command,
    build_key_value_command,
    encode_argument,
    parse_response,
)

__all__ = [
    "BinderCheetahClient",
    "CheetahConnectionError",
    "CheetahError",
    "RawArgument",
    "Response",
    "ScanItem",
    "ThreadLocalClientPool",
    "admin",
    "binder_path",
    "build_command",
    "build_key_value_command",
    "encode_argument",
    "graph",
    "jobs",
    "kv",
    "parse_response",
    "predict",
    "protocol",
]
