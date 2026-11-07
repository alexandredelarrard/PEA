"""Run the identity step's real propagation stage (`StepExtractIdentity.run_propagate`) over a hand-built resolver.

Only the rebuild of the resolver from the published tables is replaced (by `identity`); every stage of the propagation
runs as in production. Tests that stub a stage patch its name on the step module (`step_module.<stage>`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.transformers import step_extract_identity as step_module
from src.data_extract.transformers.step_extract_identity import IdentityOptions, StepExtractIdentity
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.identity.identity_propagate import RESTAMP_COLUMNS, PropagationResult
from tests.support.universe import universe_of

#: an empty published lineage, as the stubbed resolver inputs carry
_NO_LINEAGE = pd.DataFrame(columns=["canonical_ticker", "cik", "role", "valid_from"])


def _config() -> SimpleNamespace:
    paths = SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="sec_fails_to_deliver")
    return SimpleNamespace(local=SimpleNamespace(paths=paths), data_extract=SimpleNamespace(years_history=15))


def stub_propagation(monkeypatch: pytest.MonkeyPatch, module: Any = step_module) -> dict[str, list[Any]]:
    """Every stage `run_propagate` calls recorded instead of run, over a resolver whose tickers carry no scope stamp:
    `reload` (the rebuild from the published tables), `recheck` (tickers, as_of, every), `retag` (dry run)."""
    calls: dict[str, list[Any]] = {"reload": [], "recheck": [], "retag": []}
    inputs = SimpleNamespace(lineage=_NO_LINEAGE, tenure=None, master=None, co_registrants=frozenset(), mergers=())
    resolver = SimpleNamespace(filing_scope=lambda ticker: SimpleNamespace(scope_changed_at=None), bounds={})
    monkeypatch.setattr(module, "identity_inputs", lambda context: calls["reload"].append(context) or inputs)
    monkeypatch.setattr(module, "build_identity", lambda *args: resolver)
    monkeypatch.setattr(module, "bind_identity", lambda context, built: built)
    monkeypatch.setattr(
        module,
        "recheck_tickers",
        lambda scopes, as_of, every: calls["recheck"].append({"tickers": sorted(scopes), "as_of": as_of, "every": every}) or [],
    )
    monkeypatch.setattr(module, "owned_scopes", lambda identity: ({}, {}))
    for name in ("purge_rows", "refresh_insider", "refresh_fails", "refresh_short_volume"):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: [])
    for name in ("reparse_notes", "reparse_insider"):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "thirteenf_moves", lambda *args, **kwargs: (pd.DataFrame(), None))
    monkeypatch.setattr(
        module, "retag_13f", lambda context, moves, tags, dry_run: calls["retag"].append(dry_run) or pd.DataFrame(columns=list(RESTAMP_COLUMNS))
    )
    monkeypatch.setattr(module, "rebuild_history", lambda *args, **kwargs: ())
    monkeypatch.setattr(module, "log_propagation", lambda *args, **kwargs: None)
    return calls


def run_propagation(
    context: Any,
    identity: Identity,
    monkeypatch: pytest.MonkeyPatch,
    tickers: Sequence[str],
    *,
    dry_run: bool = False,
    full: bool = False,
    as_of: pd.Timestamp | None = None,
) -> PropagationResult:
    """`identity --only propagate [-t tickers] [--full] [--dry-run] [--as-of]` with `identity` as the rebuilt resolver."""
    inputs = SimpleNamespace(lineage=_NO_LINEAGE, tenure=None, master=None, co_registrants=frozenset(), mergers=())
    monkeypatch.setattr(step_module, "identity_inputs", lambda context: inputs)
    monkeypatch.setattr(step_module, "build_identity", lambda *args: identity)
    monkeypatch.setattr(step_module, "bind_identity", lambda context, built: built)
    monkeypatch.setattr(step_module, "cache_dir", lambda context, name: Path(name))
    if getattr(context, "universe", None) is None:
        context.universe = universe_of(sorted(identity.roster_cik))
    if getattr(context, "config_dir", None) is None:
        context.config_dir = "configs"
    options = IdentityOptions(tickers=list(tickers), full=full, dry_run=dry_run, as_of=as_of)
    return StepExtractIdentity(context, _config(), options).run_propagate()  # type: ignore[arg-type]
