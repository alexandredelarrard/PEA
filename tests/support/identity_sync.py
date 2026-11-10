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
from src.data_extract.transformers.step_extract_identity import IdentityOptions, StepExtractIdentity
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.identity.identity_sync import SyncResult
from tests.support.universe import universe_of

#: an empty published lineage, as the stubbed resolver inputs carry
_NO_LINEAGE = pd.DataFrame(columns=["canonical_ticker", "cik", "role", "valid_from"])


def _config() -> SimpleNamespace:
    paths = SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="sec_fails_to_deliver")
    return SimpleNamespace(local=SimpleNamespace(paths=paths), data_extract=SimpleNamespace(years_history=15))


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
        lambda scopes, as_of, every: calls["recheck"].append({"tickers": sorted(scopes), "as_of": as_of, "every": every}) or [],
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
    return StepExtractIdentity(context, _config(), IdentityOptions(full=full, as_of=as_of)).run_sync()  # type: ignore[arg-type]
