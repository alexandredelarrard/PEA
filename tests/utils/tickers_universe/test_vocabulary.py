"""Every GICS name the code or the configs key on exists in the one vocabulary (`tickers_universe.vocabulary`)."""

from __future__ import annotations

import json
from pathlib import Path

from src.data_aggregate.utils.common.sector_gates import SECTOR_KPI_SCOPE
from src.utils.config import read_config
from src.utils.tickers_universe.universe import universe_classification
from src.utils.tickers_universe.vocabulary import GICS_GROUP_SECTOR, GICS_INDUSTRY_GROUPS, GICS_SECTORS, GICS_SUB_INDUSTRY_GROUP

ROOT = Path(__file__).resolve().parents[3]
VOCABULARY = {"sector": GICS_SECTORS, "industry_group": GICS_INDUSTRY_GROUPS, "sub_industry": frozenset(GICS_SUB_INDUSTRY_GROUP)}


def _unknown(level: str, names: list[str] | tuple[str, ...]) -> list[str]:
    return sorted(set(names) - VOCABULARY[level])


def test_the_hierarchy_and_every_keyed_name_are_in_the_vocabulary() -> None:
    gics = universe_classification(read_config(path=str(ROOT / "configs")))
    regimes = json.loads((ROOT / "configs" / "fundamentals" / "fundamentals_regimes.json").read_text(encoding="utf-8"))
    gics_blocks = [regime["gics"] for regime in regimes["regimes"].values() if isinstance(regime, dict) and "gics" in regime]

    assert len(GICS_SECTORS) == 11 and set(GICS_GROUP_SECTOR) == GICS_INDUSTRY_GROUPS and len(GICS_INDUSTRY_GROUPS) == 25
    assert len(gics) == 491 and gics.notna().all().all()
    assert all(not _unknown(level, names) for level, names in SECTOR_KPI_SCOPE.values())
    checked = [(level, names) for block in gics_blocks for level, names in block.items() if level in VOCABULARY]
    assert checked and all(not _unknown(level, names) for level, names in checked)
    assert _unknown("sector", ["Fintech"]) == ["Fintech"]
    print("\n=== SANITY CHECK: GICS vocabulary ===")
    print(
        f"  11 sectors / 25 groups / {len(GICS_SUB_INDUSTRY_GROUP)} sub-industries; the yml's 491 entries resolve; "
        f"{len(SECTOR_KPI_SCOPE)} KPI scopes and {len(checked)} regime GICS lists all known."
    )
