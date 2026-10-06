"""P38 / P40: the tickers whose history waits for the traded-security realignment (plan §13.10).

Declared once in `configs/sec/security_master_manual.json` (`deferred_to_traded_security`); the shipped entries
are known truth, and a malformed entry is refused at load rather than silently ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.traded_security_deferrals import (
    EVENT_FILING_WINDOWS,
    PREDECESSOR_SERIES,
    DeferralConfigError,
    deferred_tickers,
    load_deferrals,
)

REPO_CONFIGS = Path("configs")


def test_the_shipped_config_defers_pld_dd_series_and_pld_jci_dd_event_windows() -> None:
    deferrals = load_deferrals(REPO_CONFIGS)
    print("\n=== SANITY CHECK: shipped traded-security deferrals ===")
    for d in deferrals.values():
        print(f"  {d.ticker}: {sorted(d.defers)} ({d.source})")
    assert deferred_tickers(REPO_CONFIGS, PREDECESSOR_SERIES) == {"PLD", "DD"}
    assert deferred_tickers(REPO_CONFIGS, EVENT_FILING_WINDOWS) == {"PLD", "JCI", "DD"}
    assert all("13.10" in d.source for d in deferrals.values())
    assert not {"STE", "BKR", "LIN", "EVRG", "MRK"} & set(deferrals)
    print("  OK: PLD and DD keep Sharadar's rows; PLD, JCI and DD keep undated event filings; STE, BKR, LIN, EVRG, MRK untouched")


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"ticker": "AAA", "defers": [PREDECESSOR_SERIES], "source": ""}, "no source"),
        ({"ticker": "AAA", "defers": ["prices"], "source": "plan §13.10"}, "unknown"),
        ({"ticker": "AAA", "defers": [], "source": "plan §13.10"}, "defers nothing"),
    ],
)
def test_a_malformed_entry_is_refused(tmp_path: Path, entry: dict, match: str) -> None:
    (tmp_path / "sec").mkdir()
    (tmp_path / "sec" / "security_master_manual.json").write_text(json.dumps({"deferred_to_traded_security": [entry]}), encoding="utf-8")
    with pytest.raises(DeferralConfigError, match=match):
        load_deferrals(tmp_path)
    print(f"\nsanity: {entry} refused ({match})")


def test_a_ticker_declared_twice_is_refused_and_an_absent_file_defers_nothing(tmp_path: Path) -> None:
    assert load_deferrals(tmp_path) == {}
    (tmp_path / "sec").mkdir()
    twice = [{"ticker": "AAA", "defers": [PREDECESSOR_SERIES], "source": "plan §13.10"}] * 2
    (tmp_path / "sec" / "security_master_manual.json").write_text(json.dumps({"deferred_to_traded_security": twice}), encoding="utf-8")
    with pytest.raises(DeferralConfigError, match="twice"):
        load_deferrals(tmp_path)
    print("\nsanity: no file -> nothing deferred; one ticker declared twice -> refused")
