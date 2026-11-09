"""
test_hard_guards.py  (tests/data_extract/)
--------------------------------------------------------------------------------------------
Plan-5b decision 46: FOUR impossible-only rules, applied in `build_history` before the row is
written, and NOTHING ELSE.

## Why the guards live in the builder and not in the validator

Decision 40. v2's design had the validator null impossible values with a post-hoc UPDATE on
`sec_fundamentals_history`. That contradicts the table's append-only contract in the most damaging
possible way -- a historical row would change value after publication, so yesterday's cube and
today's would disagree about the same event, and `diff_against_stored` would start reporting
drift it had itself caused. So the guards run before the write, and the validator only reports.

## Why only four, and why they are so conservative

The 745-row lesson. Over-strict Q4 guards once nulled **745 correct rows**, and v2 proposed
carrying that forward as a `[-1, 1]` bound on `debtToEquity` -- which nulls HCA's correct
negative ratio (its equity IS negative) and every filer whose debt exceeds its equity.

The split this file pins: a rule that can be WRONG may only ever raise a question
(`impossible_value`, Tier 1, flag-only). A rule may delete a value only when no filer could
report it and be right.

The second half is the one people forget, so it is tested first: **a planted HCA-shaped
negative ratio must reach the table untouched.**
"""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals.build_history import HARD_GUARDS, _hard_guard, build_ticker

_FLOW, _LEVEL = "totalRevenue", "totalAssets"


def _fact(**kwargs) -> dict:
    """One `sec_fundamentals_facts` row with every column the build reads defaulted."""
    row = {
        "ticker": "TST",
        "accession_number": "a-1",
        "field": _FLOW,
        "fiscal_year": 2023,
        "fiscal_period": "Q1",
        "duration_type": "quarterly",
        "form": "10-Q",
        "filing_date": "2023-05-01",
        "period_of_report": "2023-03-31",
        "regime": "industrial",
        "period_start": "2023-01-01",
        "period_end": "2023-03-31",
        "period_days": 89,
        "value": 100.0,
        "unit": "USD",
        "source_concept": "us-gaap:Revenues",
    }
    row.update(kwargs)
    return row


def _facts_with(level_value: float) -> pd.DataFrame:
    """Four filed quarters whose balance-sheet level carries `level_value`."""
    windows = [
        ("2023-01-01", "2023-03-31", "Q1", "2023-05-01"),
        ("2023-04-01", "2023-06-30", "Q2", "2023-08-01"),
        ("2023-07-01", "2023-09-30", "Q3", "2023-11-01"),
        ("2023-10-01", "2023-12-31", "Q4", "2024-02-15"),
    ]
    rows = []
    for i, (start, end, label, filed) in enumerate(windows):
        rows.append(
            _fact(
                accession_number=f"acc-{i}",
                fiscal_period=label,
                period_start=start,
                period_end=end,
                filing_date=filed,
                period_of_report=end,
                value=100.0 + i,
                form="10-K" if label == "Q4" else "10-Q",
            )
        )
        rows.append(
            _fact(
                accession_number=f"acc-{i}",
                field=_LEVEL,
                fiscal_period=label,
                duration_type="instant",
                period_start=None,
                period_end=end,
                period_days=None,
                filing_date=filed,
                period_of_report=end,
                value=level_value,
                source_concept="us-gaap:Assets",
                form="10-K" if label == "Q4" else "10-Q",
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# what MUST survive -- tested first, because it is the half people forget      #
# --------------------------------------------------------------------------- #


def test_an_hca_shaped_negative_ratio_reaches_the_table_untouched() -> None:
    """A negative `debtToEquity` is CORRECT for a filer with negative equity. Never nulled.

    v2 proposed a `[-1, 1]` bound on ratios as a hard guard. It would null HCA's real number
    and every filer whose debt exceeds its equity -- the 745-correct-rows failure mode with a
    different threshold on it.
    """
    row = {
        "ticker": "HCA",
        "debtToEquity": -3.4,
        "totalAssets": 50_000_000_000.0,
        "stockholdersEquity": -2_000_000_000.0,
        "sharesOutstanding": 270_000_000.0,
    }
    before = dict(row)
    _hard_guard(row)

    print("\nplanted the HCA shape: debtToEquity=-3.4, stockholdersEquity=-$2bn")
    print(
        f"  after the guards: debtToEquity={row['debtToEquity']}, stockholdersEquity={row['stockholdersEquity']:,.0f}, row unchanged={row == before}"
    )
    print(
        "  SANITY: NOTHING was nulled. A negative equity is a real accounting position, "
        "and a rule that can be wrong may only ever raise a question -- that is Tier 1's "
        "flag-only `impossible_value`, not a guard."
    )
    assert row["debtToEquity"] == -3.4
    assert row["stockholdersEquity"] == -2_000_000_000.0
    assert row == before


def test_a_shell_footing_to_exactly_zero_assets_survives() -> None:
    """`totalAssets` guards on `< 0`, not `<= 0`. VRT's pre-merger SPAC is the reason.

    A shell in its first period legitimately foots to zero. Nulling that would destroy a
    correct value to catch nothing -- the guard's whole justification is that no filer could
    report the value and be right, and zero fails that test.
    """
    row = {"ticker": "VRT", "totalAssets": 0.0}
    _hard_guard(row)

    print(f"\nplanted totalAssets = 0.0 (a pre-merger shell) -> totalAssets={row['totalAssets']}")
    print("  SANITY: survives. `< 0` and not `<= 0` -- exactly zero is a real balance sheet.")
    assert row["totalAssets"] == 0.0


# --------------------------------------------------------------------------- #
# what MUST be nulled                                                          #
# --------------------------------------------------------------------------- #


def test_a_negative_totalassets_is_nulled() -> None:
    """A negative balance-sheet total no filer could report and be right: nulled before the write."""
    row = {"ticker": "TST", "totalAssets": -1.0}
    _hard_guard(row)

    print(f"\nplanted totalAssets = -1 -> value is now {row['totalAssets']}")
    print("  SANITY: nulled before the write, so the append-only row never carries it.")
    assert row["totalAssets"] is None


def test_a_zero_share_count_is_nulled() -> None:
    """The three share counts guard on `<= 0`: a company with no shares has no equity to report.

    Different from `totalAssets` on purpose. A zero balance sheet is a real (if unusual) filing;
    a zero share count is a filing error, and every ratio built on it is a division by zero.
    """
    row = {"ticker": "TST", "sharesOutstanding": 0.0, "basicShares": -5.0, "dilutedShares": 1_000.0}
    _hard_guard(row)

    nulled = sorted(field for field in ("sharesOutstanding", "basicShares", "dilutedShares") if row[field] is None)
    print(f"\nplanted sharesOutstanding=0, basicShares=-5, dilutedShares=1000 -> nulled {nulled}; dilutedShares survived at {row['dilutedShares']}")
    print(
        "  SANITY: `<= 0` for the counts, and only the offending fields move. A guard that "
        "nulls a neighbouring correct cell is the 745-row failure mode."
    )
    assert nulled == ["basicShares", "sharesOutstanding"]
    assert row["dilutedShares"] == 1_000.0


# --------------------------------------------------------------------------- #
# the guards on the real build path                                            #
# --------------------------------------------------------------------------- #


def test_the_guards_run_inside_build_ticker_and_leave_the_grain_intact() -> None:
    """End to end: a filer whose `totalAssets` is negative in every filing, through the real builder, so the
    guard also leaves the column contract and the grain assertions intact."""
    history = build_ticker("TST", _facts_with(-1000.0))

    print(f"\n{len(history)} publication event(s) built; totalAssets non-null: {int(history['totalAssets'].notna().sum())}")
    print("  SANITY: every negative level was nulled at build time, and the guards did not disturb the grain or the column contract.")
    assert len(history) == 4 and history["totalAssets"].isna().all()
    assert history["totalAssets"].dtype == float


def test_a_clean_filer_keeps_its_values() -> None:
    """The other side: a normal filer keeps every level."""
    history = build_ticker("TST", _facts_with(1000.0))

    print(f"\nclean filer: {len(history)} event(s), totalAssets non-null {int(history['totalAssets'].notna().sum())}")
    print("  SANITY: no guard fires on correct data.")
    assert history["totalAssets"].notna().all() and set(history["totalAssets"]) == {1000.0}


def test_the_guard_set_is_exactly_the_four_decision_46_names() -> None:
    """Four rules, named. A fifth is a decision, not an edit."""
    print(f"\nHARD_GUARDS: {sorted(HARD_GUARDS)}")
    print(
        "  SANITY: exactly four, all impossible-only. Everything else v2 listed became "
        "Tier 1's flag-only `impossible_value`, which reports and never deletes."
    )
    assert sorted(HARD_GUARDS) == ["basicShares", "dilutedShares", "sharesOutstanding", "totalAssets"]
