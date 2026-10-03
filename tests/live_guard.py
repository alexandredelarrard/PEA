"""Tripwire that refuses live side effects inside a pytest process.

`find_dotenv` walks up from any worktree to the parent repo's `.env`, so a real `Context`
built by a test gets the live `DATABASE_URL` and every API key. A test that writes or calls a
vendor therefore does it for real unless something stops it. `tests/conftest.py` installs
this tripwire for every run that is not opted in with `PEA_LIVE_TESTS=1`; it can also be
loaded on its own (`PYTHONPATH=tests pytest -p live_guard ...`) to probe a tree that has no
gate.

What it blocks, each BEFORE the effect leaves the process:

* a Python socket `connect` to a non-loopback host, or to a loopback port named by a
  `*_PROXY` variable (requests, httpx -> edgartools / OpenAI, urllib, asyncio);
* `curl_cffi` transfers, which go through libcurl and never touch a Python socket
  (the Yahoo, Sharadar and polite-HTTP paths);
* every `DataStore` write method on a non-SQLite engine (`save`, `bulk_seed`, `append_tail`,
  `delete`, `replace`, `ensure_columns`, `drop`) -- `bulk_seed` / `replace` COPY through a
  raw psycopg2 cursor that SQLAlchemy events never see;
* any SQL statement other than a read on a non-SQLite engine (a raw `DELETE` in a test).

Every trip is recorded in `TRIPS` and reported at the end of the session, which then fails:
a trip swallowed by a broad `except Exception: pytest.skip(...)` still shows up.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import pytest

LIVE_FLAG = "PEA_LIVE_TESTS"
LIVE_MARKER = "live"

#: Every blocked attempt, in order: "<kind>: <detail>".
TRIPS: list[str] = []

_STORE_WRITES = ("save", "bulk_seed", "append_tail", "delete", "replace", "ensure_columns", "drop")
_READ_SQL = re.compile(r"^\s*(select|show|set|explain|begin|commit|rollback|savepoint|release)\b", re.IGNORECASE)
_WITH_WRITE = re.compile(r"\b(insert|update|delete|merge|truncate|create|alter|drop|copy)\b", re.IGNORECASE)
_installed = False


class LiveEffectBlockedError(RuntimeError):
    """A test tried to write the live database or reach an external host."""


def live_enabled() -> bool:
    """True only when the run is explicitly opted in to live tests."""
    return os.getenv(LIVE_FLAG, "").strip() == "1"


def _trip(kind: str, detail: str) -> None:
    TRIPS.append(f"{kind}: {detail}")
    raise LiveEffectBlockedError(f"live {kind} blocked by tests/live_guard.py ({detail}); mark the test @pytest.mark.live and run with {LIVE_FLAG}=1")


def _proxy_ports() -> set[int]:
    ports: set[int] = set()
    for key, value in os.environ.items():
        if key.lower().endswith("_proxy") and value:
            parsed = urlsplit(value if "://" in value else f"http://{value}")
            if parsed.port:
                ports.add(parsed.port)
    return ports


def _is_local(address: Any) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX path or an unknown family: never a remote host
    host, port = str(address[0]), address[1] if len(address) > 1 else None
    if port in _proxy_ports():
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _guard_socket() -> None:
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def connect(self: socket.socket, address: Any) -> None:
        if not _is_local(address):
            _trip("network", f"socket connect to {address!r}")
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_local(address):
            _trip("network", f"socket connect_ex to {address!r}")
        return real_connect_ex(self, address)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]


def _guard_curl() -> None:
    try:
        from curl_cffi import curl as curl_mod
    except ImportError:
        return

    def perform(self: Any, *args: Any, **kwargs: Any) -> Any:
        _trip("network", "curl_cffi Curl.perform")

    curl_mod.Curl.perform = perform  # type: ignore[method-assign]
    try:
        from curl_cffi import aio as aio_mod
    except ImportError:
        return

    def add_handle(self: Any, *args: Any, **kwargs: Any) -> Any:
        _trip("network", "curl_cffi AsyncCurl.add_handle")

    aio_mod.AsyncCurl.add_handle = add_handle  # type: ignore[method-assign]


def _store_write_guard(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(self: Any, table: Any, *args: Any, **kwargs: Any) -> Any:
        if self.engine.dialect.name != "sqlite":
            _trip("db-write", f"DataStore.{name}({table!r}) on {self.engine.dialect.name}")
        return real(self, table, *args, **kwargs)

    guarded.__name__ = name
    return guarded


def _guard_store() -> None:
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    from src.data_store.store import DataStore

    for name in _STORE_WRITES:
        setattr(DataStore, name, _store_write_guard(name, getattr(DataStore, name)))

    def before_cursor_execute(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool) -> None:
        if conn.dialect.name == "sqlite":
            return
        text = statement.lstrip()
        if _READ_SQL.match(text) or (re.match(r"with\b", text, re.IGNORECASE) and not _WITH_WRITE.search(text)):
            return
        _trip("db-write", f"SQL on {conn.dialect.name}: {' '.join(text.split())[:80]}")

    event.listen(Engine, "before_cursor_execute", before_cursor_execute)


def install() -> None:
    """Install every guard once per process (idempotent)."""
    global _installed
    if _installed:
        return
    _guard_socket()
    _guard_curl()
    _guard_store()
    _installed = True


def pytest_configure(config: pytest.Config) -> None:
    if not live_enabled():
        install()


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    if not _installed:
        return
    terminalreporter.section("live_guard")
    terminalreporter.write_line(f"tripwire installed ({LIVE_FLAG} unset); blocked attempts: {len(TRIPS)}")
    for trip in TRIPS:
        terminalreporter.write_line(f"  BLOCKED {trip}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if TRIPS and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
