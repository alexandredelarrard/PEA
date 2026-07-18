"""
db.py  (src/utils/db.py)
------------------------
SQLAlchemy engine factory. The connection URL comes from the DATABASE_URL env
var (set by docker-compose), falling back to the local compose default so the
migrator can be run from the host against the exposed container port.

Kept dependency-light and engine-agnostic: the store layer (src/data_store)
adapts its upsert to whichever dialect the engine reports, so the same code
runs against Postgres (production) and SQLite (fast offline sanity checks).
"""
from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine

# matches docker-compose.yml (service name `db`, or localhost from the host)
def database_url() -> str:
    string_url = f"postgresql+psycopg2://{os.environ.get("POSTGRES_USER", 'pea')}:{os.environ.get("POSTGRES_PASSWORD", 'pea')}@localhost:{os.environ.get("POSTGRES_PORT", '5432')}/{os.environ.get("POSTGRES_DB", 'pea')}"
    return os.getenv("DATABASE_URL", string_url)

@lru_cache(maxsize=8)
def get_engine(url: str | None = None) -> Engine:
    """Return a cached SQLAlchemy Engine. `pool_pre_ping` avoids stale
    connections when the container restarts. For Postgres we batch executemany
    into multi-row VALUES (`values_plus_batch`) so wide upserts (e.g. the
    159-column cube) don't degrade to one round-trip per row."""
    resolved = url or database_url()
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if resolved.startswith("postgresql"):
        kwargs["executemany_mode"] = "values_plus_batch"
        kwargs["insertmanyvalues_page_size"] = 1000
    return create_engine(resolved, **kwargs)
