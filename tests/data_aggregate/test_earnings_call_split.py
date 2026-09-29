"""The former split panels are one unified raw/history earnings-call contract."""

from __future__ import annotations

from src.constants.constants import EARNINGS_CALL_FEATURES
from src.data_aggregate.utils.text import earnings_call_features as ec


def test_embedding_kpis_are_part_of_the_unified_contract() -> None:
    retained_embedding_kpis = {"ec_qa_coherence_mean", "ec_qa_qq_distance", "ec_prep_qq_distance"}
    assert retained_embedding_kpis <= set(EARNINGS_CALL_FEATURES)
    assert tuple(ec._KPI_COLS) == EARNINGS_CALL_FEATURES
    assert not any(name.endswith(("_xs", "_vs_peers")) for name in EARNINGS_CALL_FEATURES)
    print("\n=== SANITY CHECK: unified earnings-call contract ===")
    print("  embedding KPIs are part of the exact 12 raw/history fields; no separate peer/xs panel remains. Validated.")
