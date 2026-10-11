"""
The insider-ownership estimate of the cleaning stage (clean.py, REQ-007).

Known-truth fixture: Σ director/officer shares / point-in-time shares outstanding fills a NULL
`insider_ownership_pct` only on a single-class ticker, with at least 5 holders and a result in
(0, 1]. A filed, computed or carried value is never overwritten, and the written cell takes the
proxy date and kind `estimate`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.governance.clean import GovernanceInputs, clean_governance_inputs
from src.data_aggregate.utils.governance.def14a_impute import FILL_ESTIMATE, fill_kind_column
from src.data_aggregate.utils.governance.staleness import source_date_column

_COL = "insider_ownership_pct"


def _holders(ticker: str, accession: str, n: int, each: float, kind: str = "director_officer") -> list[dict[str, object]]:
    return [{"ticker": ticker, "accession_number": accession, "holder_type": kind, "shares": each} for _ in range(n)]


def _bundle() -> tuple[GovernanceInputs, pd.DataFrame]:
    """Five tickers, one rule each; every share count is over 1e8 outstanding."""
    rows = [
        # SSS: S-0 estimated, S-1 filed, S-2 carried from S-1 (an estimate exists, not used)
        ("SSS", "S-0", "2021-03-01", np.nan, np.nan, 0.0),
        ("SSS", "S-1", "2022-03-01", 0.05, np.nan, 0.0),
        ("SSS", "S-2", "2023-03-01", np.nan, np.nan, 0.0),
        # BBB: Σ holders = 2x shares outstanding -> outside (0, 1]
        ("BBB", "B-0", "2021-03-01", np.nan, np.nan, 0.0),
        # DDD: single-class on D-0 but flagged dual on D-1 -> never eligible
        ("DDD", "D-0", "2021-03-01", np.nan, np.nan, 0.0),
        ("DDD", "D-1", "2022-03-01", np.nan, np.nan, 1.0),
        # FFF: only 4 holders
        ("FFF", "F-0", "2021-03-01", np.nan, np.nan, 0.0),
        # CCC: the disclosed share count computes the value first
        ("CCC", "C-0", "2021-03-01", np.nan, 3.0e6, 0.0),
    ]
    def14a = pd.DataFrame(rows, columns=["ticker", "accession_number", "as_of", _COL, "insider_shares", "dual_class_shares"])
    def14a["ceo_name_proxy"] = "Jane Doe"
    def14a["insider_voting_pct"] = np.nan
    # a stale stamp on the NULL cell the estimate fills must not survive the write
    def14a[source_date_column(_COL)] = pd.to_datetime(["2010-01-01", "2022-03-01"] + [None] * 6)

    ownership = pd.DataFrame(
        _holders("SSS", "S-0", 6, 1.0e6 / 3)
        + _holders("SSS", "S-0", 1, 5.0e7, kind="5pct_holder")
        + _holders("SSS", "S-1", 6, 1.0e6)
        + _holders("SSS", "S-2", 6, 1.0e6)
        + _holders("BBB", "B-0", 6, 1.0e8 / 3)
        + _holders("DDD", "D-0", 6, 1.0e6)
        + _holders("FFF", "F-0", 4, 1.0e6)
        + _holders("CCC", "C-0", 6, 1.0e6)
    )
    fundamentals = pd.DataFrame(
        {
            "ticker": ["SSS", "BBB", "DDD", "FFF", "CCC"],
            "as_of": pd.to_datetime(["2020-12-31"] * 5),
            "sharesOutstandingPit": [1.0e8] * 5,
        }
    )
    return GovernanceInputs(def14a=def14a, ownership=ownership), fundamentals


def test_estimate_fills_only_eligible_null_cells() -> None:
    raw, fundamentals = _bundle()
    cleaned, stats = clean_governance_inputs(raw, fundamentals)
    d = cast(pd.DataFrame, cleaned.def14a).set_index("accession_number")
    kind, src = fill_kind_column(_COL), source_date_column(_COL)

    # S-0: 6 x 333,333 director/officer shares (the 5% holder ignored) / 1e8
    assert np.isclose(d.loc["S-0", _COL], 0.02) and d.loc["S-0", kind] == FILL_ESTIMATE
    assert pd.Timestamp(d.loc["S-0", src]) == pd.Timestamp("2021-03-01")
    # filed, carried and computed values win over an available estimate
    assert d.loc["S-1", _COL] == 0.05 and d.loc["S-1", kind] == "filed"
    assert d.loc["S-2", _COL] == 0.05 and d.loc["S-2", kind] == "carry"
    assert np.isclose(d.loc["C-0", _COL], 0.03) and d.loc["C-0", kind] == "cross_source"
    # bound, dual eligibility and holder floor refuse
    for acc in ("B-0", "D-0", "F-0"):
        assert pd.isna(d.loc[acc, _COL]) and pd.isna(d.loc[acc, kind]), acc

    est = {k.split("|", 1)[1]: v for k, v in stats.items() if k.startswith("ownership_estimate|")}
    assert est["insider_ownership_pct: ESTIMATED from director/officer holder rows"] == 1
    assert est["insider_ownership_pct: estimate REJECTED (outside (0, 1])"] == 1
    assert est["insider_ownership_pct: estimate REFUSED (ticker ever flagged dual-class)"] == 1
    assert est["insider_ownership_pct: estimate REFUSED (< 5 director/officer holders)"] == 1
    assert est["insider_ownership_pct: carried value KEPT over a same-filing estimate"] == 1

    print("\n=== SANITY CHECK: the insider-ownership estimate fills only eligible NULL cells ===")
    print("  S-0 -> 0.02 estimate @ 2021-03-01 (stale 2010 stamp reset); filed S-1, carried S-2, computed C-0 kept;")
    print("  B-0 (ratio 2.0), D-0 (dual on a later proxy), F-0 (4 holders) left NULL; every rule counted once. Validated.")


def test_estimate_without_ownership_table_is_a_no_op() -> None:
    raw, fundamentals = _bundle()
    cleaned, stats = clean_governance_inputs(GovernanceInputs(def14a=raw.def14a), fundamentals)
    d = cast(pd.DataFrame, cleaned.def14a).set_index("accession_number")
    assert pd.isna(d.loc["S-0", _COL])
    assert not any(k.startswith("ownership_estimate|") for k in stats)

    print("\n=== SANITY CHECK: no holder table, no estimate ===")
    print("  S-0 stays NULL and no ownership_estimate stats are emitted. Validated.")


def test_entity_holders_and_implausible_estimates_are_excluded() -> None:
    """Institutions mis-tagged `director_officer` are not summed; an estimate far above the filed history is refused."""
    def14a = pd.DataFrame(
        [
            ("TXN", "T-0", "2021-03-01", 0.005, np.nan, 0.0),
            ("TXN", "T-1", "2026-03-01", np.nan, np.nan, 0.0),
            ("GGG", "G-0", "2021-03-01", 0.001, np.nan, 0.0),
            ("GGG", "G-1", "2026-03-01", np.nan, np.nan, 0.0),
        ],
        columns=["ticker", "accession_number", "as_of", _COL, "insider_shares", "dual_class_shares"],
    )
    def14a["ceo_name_proxy"] = "Jane Doe"
    def14a["insider_voting_pct"] = np.nan
    people = [f"Director {c}. Smith" for c in "ABCDEF"]
    rows = [{"ticker": "TXN", "accession_number": "T-1", "holder_type": "director_officer", "holder_name": n, "shares": 1.0e5} for n in people]
    rows += [
        {"ticker": "TXN", "accession_number": "T-1", "holder_type": "director_officer", "holder_name": n, "shares": 1.0e7}
        for n in ("The Vanguard Group", "BlackRock, Inc.", "All directors and executive officers as a group")
    ]
    rows += [{"ticker": "GGG", "accession_number": "G-1", "holder_type": "director_officer", "holder_name": n, "shares": 1.0e5} for n in people]
    fundamentals = pd.DataFrame({"ticker": ["TXN", "GGG"], "as_of": pd.to_datetime(["2020-12-31"] * 2), "sharesOutstandingPit": [1.0e8] * 2})
    cleaned, stats = clean_governance_inputs(GovernanceInputs(def14a=def14a, ownership=pd.DataFrame(rows)), fundamentals)
    d = cast(pd.DataFrame, cleaned.def14a).set_index("accession_number")
    # TXN: 6 x 1e5 / 1e8 = 0.006, the three entity rows (3e7 shares) are not summed
    assert np.isclose(d.loc["T-1", _COL], 0.006) and d.loc["T-1", fill_kind_column(_COL)] == FILL_ESTIMATE
    # GGG: 0.006 is 6x the largest filed 0.001 -> refused
    assert pd.isna(d.loc["G-1", _COL])
    est = {k.split("|", 1)[1]: v for k, v in stats.items() if k.startswith("ownership_estimate|")}
    assert est["insider_ownership_pct: director/officer holder rows EXCLUDED (entity name)"] == 3
    assert est["insider_ownership_pct: estimate REJECTED (> 3x the ticker's largest filed value)"] == 1
    print("\n=== SANITY CHECK: entity holder rows and implausible estimates ===")
    print(
        f"  TXN T-1 {d.loc['T-1', _COL]:.4f} (Vanguard, BlackRock and the group row excluded) | GGG G-1 {d.loc['G-1', _COL]} (6x filed max refused)"
    )
    print("  CONCLUSION: the estimate sums people only and cannot jump far above what the ticker ever filed.")


def test_board_trust_row_excluded_and_titles_not_tested() -> None:
    """MET 2014: the Policyholder Trust row "Board of Directors of MetLife, but not in each
    Director's individual capacity" is not a person; an INTC executive whose TITLE names a
    "Group" is one, and so is a director whose title names "the Board of Directors"."""
    def14a = pd.DataFrame(
        [("MET", "M-0", "2014-03-25", np.nan, np.nan, 0.0), ("INT", "I-0", "2014-03-25", np.nan, np.nan, 0.0)],
        columns=["ticker", "accession_number", "as_of", _COL, "insider_shares", "dual_class_shares"],
    )
    def14a["ceo_name_proxy"] = "Jane Doe"
    def14a["insider_voting_pct"] = np.nan
    people = [f"Director {c}. Smith" for c in "ABCDE"]

    def holder(t: str, a: str, name: str, shares: float) -> dict[str, object]:
        return {"ticker": t, "accession_number": a, "holder_type": "director_officer", "holder_name": name, "shares": shares}

    rows = [holder("MET", "M-0", n, 2.0e5) for n in people]
    rows.append(holder("MET", "M-0", "Board of Directors of MetLife, but not in each Director's individual capacity", 1.9e8))
    rows += [holder("INT", "I-0", n, 2.0e5) for n in people]
    rows.append(holder("INT", "I-0", "Navin Shenoy, Executive Vice President and General Manager, Data Center Group", 1.0e6))
    rows.append(holder("INT", "I-0", "Aart J. de Geus Chief Executive Officer and Chairperson of the Board of Directors", 1.0e6))
    fundamentals = pd.DataFrame({"ticker": ["MET", "INT"], "as_of": pd.to_datetime(["2013-12-31"] * 2), "sharesOutstandingPit": [1.0e9] * 2})
    cleaned, stats = clean_governance_inputs(GovernanceInputs(def14a=def14a, ownership=pd.DataFrame(rows)), fundamentals)
    d = cast(pd.DataFrame, cleaned.def14a).set_index("accession_number")
    assert np.isclose(d.loc["M-0", _COL], 1.0e-3), f"MET must sum the five people only: {d.loc['M-0', _COL]}"
    assert np.isclose(d.loc["I-0", _COL], 3.0e-3), f"INT must keep both executives: {d.loc['I-0', _COL]}"
    assert stats["ownership_estimate|insider_ownership_pct: director/officer holder rows EXCLUDED (entity name)"] == 1
    print("\n=== SANITY CHECK: board trust row and executive titles ===")
    print(f"  MET {d.loc['M-0', _COL]:.4f} (trust row excluded) | INT {d.loc['I-0', _COL]:.4f} (Group / Board titles kept)")
    print("  CONCLUSION: only the holder's name is tested for an entity, and the board-held trust is not a person's holding.")
