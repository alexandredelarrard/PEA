"""`StepExtractUniverse`: the evidence sub-runs (`run_sharadar_actions`, `run_sec_submissions`) land before the sync."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.transformers import step_extract_universe as step_module

_AS_OF = pd.Timestamp("2026-11-04")


def _step(tmp_path: Path) -> step_module.StepExtractUniverse:
    config = SimpleNamespace(
        data_extract=SimpleNamespace(sharadar_years_history=31), local=SimpleNamespace(paths=SimpleNamespace(sec_submissions="sec_submissions"))
    )
    context = SimpleNamespace(config_dir="configs", paths={"DATA_STORE": tmp_path}, store=None)
    return step_module.StepExtractUniverse(context, config, _AS_OF)  # type: ignore[arg-type]


def test_run_lands_both_ticker_lists_and_the_evidence_before_the_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    received: dict[str, Any] = {}
    monkeypatch.setattr(step_module, "download_sec_company_tickers", lambda context: ran.append("sec-tickers") or 1)
    monkeypatch.setattr(step_module, "fetch_sharadar_tickers", lambda context: ran.append("sharadar-tickers"))

    def actions(context: Any, *, years_history: int, as_of: pd.Timestamp) -> int:
        received["actions"] = (years_history, as_of)
        ran.append("sharadar-actions")
        return 7

    def profile(context: Any, cache: Path, start_date: pd.Timestamp) -> int:
        received["profile"] = (cache, start_date)
        ran.append("sec-submissions")
        return 3

    monkeypatch.setattr(step_module, "fetch_sharadar_actions", actions)
    monkeypatch.setattr(step_module, "refresh_sec_cik_profile", profile)
    monkeypatch.setattr(step_module, "sync", lambda store, config, as_of: ran.append("sync") or 0)

    step = _step(tmp_path)
    step.run()

    assert ran == ["sec-tickers", "sharadar-tickers", "sharadar-actions", "sec-submissions", "sync"]
    assert received["actions"] == (31, _AS_OF)
    assert received["profile"] == (tmp_path / "sec_submissions", _AS_OF) and (tmp_path / "sec_submissions").is_dir()
    assert step.run_sharadar_actions() == 7 and step.run_sec_submissions() == 3

    print("\n=== SANITY CHECK: universe step order ===")
    print(f"  run() -> {ran}; actions over sharadar_years_history=31 as of {_AS_OF.date()}; submissions cached in <DATA_STORE>/sec_submissions.")
