"""Every GICS name the code keys on exists in the one vocabulary (`tickers_universe.vocabulary`)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_aggregate.utils.common.sector_gates import SECTOR_KPI_SCOPE
from src.utils.tickers_universe.vocabulary import GICS_INDUSTRY_GROUPS, GICS_SECTORS
from src.utils.tickers_universe.wikipedia_bootstrap import _SUB_TO_GROUP

ROOT = Path(__file__).resolve().parents[3]
VOCABULARY = {"sector": GICS_SECTORS, "industry_group": GICS_INDUSTRY_GROUPS}


def _unknown(level: str, names: list[str] | tuple[str, ...]) -> list[str]:
    return sorted(set(names) - VOCABULARY[level])


def test_stored_scope_and_regime_names_are_in_the_vocabulary() -> None:
    roster = pd.read_csv(ROOT / "tests" / "fixtures" / "universe" / "roster.csv", dtype=str)
    regimes = json.loads((ROOT / "configs" / "fundamentals" / "fundamentals_regimes.json").read_text(encoding="utf-8"))
    gics_blocks = [regime["gics"] for regime in regimes["regimes"].values() if isinstance(regime, dict) and "gics" in regime]

    assert len(GICS_SECTORS) == 11 and len(GICS_INDUSTRY_GROUPS) == 25
    assert set(_SUB_TO_GROUP.values()) == GICS_INDUSTRY_GROUPS
    assert not _unknown("sector", roster["sector"].tolist()) and not _unknown("industry_group", roster["industry_group"].tolist())
    assert all(not _unknown(level, names) for level, names in SECTOR_KPI_SCOPE.values())
    checked = [(level, names) for block in gics_blocks for level, names in block.items() if level in VOCABULARY]
    assert checked and all(not _unknown(level, names) for level, names in checked)
    assert _unknown("sector", ["Fintech"]) == ["Fintech"]
    print("\n=== SANITY CHECK: GICS vocabulary ===")
    print(
        f"  11 sectors / 25 groups (= the bootstrap's sub-industry targets); roster (500 rows), {len(SECTOR_KPI_SCOPE)} KPI scopes and {len(checked)} regime GICS lists all known."
    )
