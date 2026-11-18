"""Run the identity step's real sync stage (`StepExtractIdentity.run_sync`) over a hand-built resolver, or stub it.

`run_sync`: only the rebuild of the resolver from the published tables is replaced (by `identity`); the move, the purge
and the insider re-stamp run as in production. `stub_sync`: every operation is recorded instead of run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.transformers import step_extract_identity as step_module
from src.data_extract.transformers.step_extract_identity import StepExtractIdentity
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.identity.identity_sync import SyncResult
from src.utils.window import WindowRun
from tests.data_extract.fake_context import extract_config
from tests.support.universe import universe_of

#: an empty published lineage, as the stubbed resolver inputs carry
_NO_LINEAGE = pd.DataFrame(columns=["canonical_ticker", "cik", "role", "valid_from"])


def stub_sync(monkeypatch: pytest.MonkeyPatch, module: Any = step_module) -> dict[str, list[Any]]:
    """Every operation `run_sync` calls recorded instead of run, over a resolver whose tickers carry no scope stamp:
    `reload` (the rebuild from the published tables), `recheck` (tickers, as_of, every)."""
    calls: dict[str, list[Any]] = {"reload": [], "recheck": []}
    inputs = SimpleNamespace(lineage=_NO_LINEAGE, tenure=None, master=None, co_registrants=frozenset(), mergers=())
    resolver = SimpleNamespace(filing_scope=lambda ticker: SimpleNamespace(scope_changed_at=None), bounds={})
    monkeypatch.setattr(module, "identity_inputs", lambda context: calls["reload"].append(context) or inputs)
    monkeypatch.setattr(module, "build_identity", lambda *args: resolver)
    monkeypatch.setattr(module, "bind_identity", lambda context, built, universe: built)
    monkeypatch.setattr(
        module,
        "recheck_tickers",
        lambda context, scopes, every: calls["recheck"].append({"tickers": sorted(scopes), "as_of": context.run.as_of, "every": every}) or [],
    )
    monkeypatch.setattr(module, "owned_scopes", lambda identity: ({}, {}))
    monkeypatch.setattr(module, "sync_filer_rows", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(module, "restamp_insider_lineage", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "log_sync", lambda *args, **kwargs: None)
    return calls


def run_sync(
    context: Any, identity: Identity, monkeypatch: pytest.MonkeyPatch, *, full: bool = False, as_of: pd.Timestamp | None = None
) -> SyncResult:
    """`identity --only sync [--full] [--as-of]` with `identity` as the rebuilt resolver."""
    inputs = SimpleNamespace(lineage=_NO_LINEAGE, tenure=None, master=None, co_registrants=frozenset(), mergers=())
    monkeypatch.setattr(step_module, "identity_inputs", lambda context: inputs)
    monkeypatch.setattr(step_module, "build_identity", lambda *args: identity)
    monkeypatch.setattr(step_module, "bind_identity", lambda context, built, universe: built)
    monkeypatch.setattr(step_module, "cache_dir", lambda context, name: Path(name))
    if getattr(context, "universe", None) is None:
        context.universe = universe_of(sorted(identity.roster_cik))
    if getattr(context, "config_dir", None) is None:
        context.config_dir = "configs"
    if getattr(getattr(context, "config", None), "data", None) is None:
        context.config = extract_config()
    current = getattr(context, "run", None)
    anchor = as_of if as_of is not None else current.instant if current is not None else pd.Timestamp("2026-11-16")
    context.run = WindowRun("full" if full else "incremental", "identity sync fixture", anchor)
    return StepExtractIdentity(context, context.config).run_sync()  # type: ignore[arg-type]
