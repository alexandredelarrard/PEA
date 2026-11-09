"""
`StepExtractFundamentals`: `run_history` replays the stored SEC facts into `sec_fundamentals_history` offline, and
`run()` runs it. The facts walk is `StepExtractEdgar.run_fundamentals_facts`; earnings surprises are
`StepExtractPrices.run_earnings_surprises`, the notes zips `StepExtractZips`.
"""

from __future__ import annotations

import ast
import logging
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

import src.data_extract.transformers.step_extract_fundamentals as mod
from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals
from src.data_extract.utils.fundamentals import build_history
from src.data_store.schema import Tables
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

#: Modules that reach the network; none may be imported by the history step or its own fundamentals modules.
_NETWORK = ("edgar", "httpx", "requests", "src.data_extract.utils.common.sec_io", "src.data_extract.utils.fundamentals.fetch_fundamentals_sec")
_ROOT = Path(__file__).resolve().parents[2]
_HISTORY_MODULES = (
    "src/data_extract/transformers/step_extract_fundamentals.py",
    "src/data_extract/utils/fundamentals/build_history.py",
    "src/data_extract/utils/fundamentals/periods.py",
    "src/data_extract/utils/fundamentals/kpi_catalogue.py",
)


def _step(context: Any) -> StepExtractFundamentals:
    limits = {"gap_pct": 0.03, "field_share": 0.05, "top_tickers": 3}
    return StepExtractFundamentals(context, OmegaConf.create({"data_extract": {"years_history": 15, "fundamentals_check": limits}}))


def _imports(path: str) -> set[str]:
    tree = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    return names | {str(node.module) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}


def test_run_history_run_check_and_run_reach_their_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str], dict]] = []
    monkeypatch.setattr(mod, "build_fundamentals_history", lambda context, tickers, **kwargs: calls.append(("history", tickers, kwargs)))
    monkeypatch.setattr(mod, "check_fundamentals_sources", lambda context, tickers, **kwargs: calls.append(("check", tickers, kwargs)))
    step = _step(SimpleNamespace(config_dir="./configs"))

    step.run_history(["AAPL"], full=True)
    step.run_history(["AAPL"], verify=True)
    step.run(tickers=["MSFT"])

    limits = {"gap_pct": 0.03, "field_share": 0.05, "top_tickers": 3}
    assert calls == [
        ("history", ["AAPL"], {"rebuild_history": True, "verify_history": False}),
        ("history", ["AAPL"], {"rebuild_history": False, "verify_history": True}),
        ("history", ["MSFT"], {"rebuild_history": False, "verify_history": False}),
        ("check", ["MSFT"], limits),
    ], calls
    assert not hasattr(mod, "fetch_fundamentals_sec") and not hasattr(mod, "fetch_earnings_surprises")
    print("\n=== SANITY CHECK: StepExtractFundamentals ===")
    print(f"  run_history(full) / run_history(verify) / run() -> {[(name, kw) for name, _, kw in calls]};")
    print("  run() = history then check with the configured thresholds; no fetcher in the step. Validated.")


def test_the_history_step_imports_no_network_module() -> None:
    """The step and its own fundamentals modules import nothing that reaches the network."""
    found = {path: sorted(name for name in _imports(path) if name.startswith(_NETWORK)) for path in _HISTORY_MODULES}

    assert not any(found.values()), found
    print("\n=== SANITY CHECK: no network import in the history step's own modules ===")
    for path, names in found.items():
        print(f"  {path}: {names or 'none'}")
    print("  Validated.")


def test_importing_the_history_step_loads_no_edgar_or_sec_io() -> None:
    """A fresh interpreter importing the step loads neither `edgar` nor `sec_io` through any transitive import."""
    probe = (
        "import sys, src.data_extract.transformers.step_extract_fundamentals\n"
        "print(sorted(m for m in sys.modules if m == 'edgar' or m.startswith('edgar.') or m == 'src.data_extract.utils.common.sec_io'))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=_ROOT, capture_output=True, text=True, check=True)
    loaded = result.stdout.strip().splitlines()[-1]

    assert loaded == "[]", loaded
    print("\n=== SANITY CHECK: transitive imports of step_extract_fundamentals ===")
    print(f"  edgar / sec_io modules loaded by a fresh import: {loaded}")
    print("  Validated: the offline history step pulls in no EDGAR network code.")


def test_run_history_builds_with_every_socket_refused(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end on the in-memory store with every socket connection refused: the replay saves its rows."""
    rows = [
        {
            "ticker": "TST",
            "cik": "0000000001",
            "accession_number": f"acc-{i}",
            "field": "totalAssets",
            "fiscal_year": 2023,
            "fiscal_period": f"Q{i + 1}",
            "duration_type": "instant",
            "form": "10-Q",
            "filing_date": pd.Timestamp(filed),
            "period_of_report": pd.Timestamp(end),
            "regime": "industrial",
            "period_start": None,
            "period_end": pd.Timestamp(end),
            "period_days": None,
            "value": 1000.0 + i,
            "unit": "USD",
            "source_concept": "us-gaap:Assets",
        }
        for i, (end, filed) in enumerate((("2023-03-31", "2023-05-01"), ("2023-06-30", "2023-08-01")))
    ]
    sqlite_store.save(Tables.sec_fundamentals_facts, pd.DataFrame(rows)[list(build_history.FACT_COLUMNS)])
    identity = dated_identity([("ZZZ", "0000009999", "cik_window", SENTINEL, None)], {"ZZZ": "0000009999"})
    monkeypatch.setattr(build_history, "load_identity", lambda context: identity)
    monkeypatch.setattr(build_history, "_scope_changed_recently", lambda context, tickers, as_of=None: frozenset())

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the history replay opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    _step(SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.step_fundamentals"), config_dir="./configs")).run_history(["TST"])

    stored = sqlite_store.load(Tables.sec_fundamentals_history, columns=["ticker", "as_of", "totalAssets"])
    assert list(stored["totalAssets"]) == [1000.0, 1001.0], stored
    print("\n=== SANITY CHECK: offline history replay ===")
    print(f"  every socket connect refused; run_history saved {len(stored)} rows, totalAssets {list(stored['totalAssets'])}. Validated.")
