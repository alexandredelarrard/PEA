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

from sqlalchemy import Engine, URL, create_engine


def database_url() -> str | URL:
    """Connection target, in priority order:
      1. `DATABASE_URL` env (set explicitly, e.g. in CI),
      2. a URL built from the POSTGRES_* env vars, with `POSTGRES_HOST` defaulting to `localhost`
         (the host reaching the exposed container port) — set it to the compose SERVICE name
         (`db`) inside a container. Built via `URL.create`, which URL-escapes the password, so
         special characters (`!`, `@`, …) no longer break the connection string.
    """
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    return URL.create(
        "postgresql+psycopg2",
        username=os.environ.get("POSTGRES_USER", "pea"),
        password=os.environ.get("POSTGRES_PASSWORD", "pea"),
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "pea"),
    )


@lru_cache(maxsize=8)
def get_engine(url: str | URL | None = None) -> Engine:
    """Return a cached SQLAlchemy Engine. `pool_pre_ping` avoids stale
    connections when the container restarts. For Postgres we batch executemany
    into multi-row VALUES (`values_plus_batch`) so wide upserts (e.g. the
    159-column cube) don't degrade to one round-trip per row."""
    resolved = url or database_url()
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if str(resolved).startswith("postgresql"):
        kwargs["executemany_mode"] = "values_plus_batch"
        kwargs["insertmanyvalues_page_size"] = 1000
    return create_engine(resolved, **kwargs)
