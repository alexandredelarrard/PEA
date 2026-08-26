"""
test_hard_guards.py  (tests/data_extract/)
--------------------------------------------------------------------------------------------
Plan-5b decision 46: FOUR impossible-only rules, applied in `build_history` before the row is
written, and NOTHING ELSE.

## Why the guards live in the builder and not in the validator

Decision 40. v2's design had the validator null impossible values with a post-hoc UPDATE on
`fundamentals_history_sec`. That contradicts the table's append-only contract in the most damaging
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

from src.data_extract.utils.fundamentals import reason_codes as rc
from src.data_extract.utils.fundamentals.build_history import (
    HARD_GUARDS, _hard_guard, build_ticker)

_FLOW, _LEVEL = "totalRevenue", "totalAssets"


def _fact(**kwargs) -> dict:
    """One `fundamentals_facts` row with every column the build reads defaulted."""
    row = {"ticker": "TST", "accession_number": "a-1", "field": _FLOW,
           "fiscal_year": 2023, "fiscal_period": "Q1", "duration_type": "quarterly",
           "form": "10-Q", "filing_date": "2023-05-01", "is_amendment": False,
           "period_of_report": "2023-03-31", "regime": "industrial",
           "period_start": "2023-01-01", "period_end": "2023-03-31", "period_days": 89,
           "value": 100.0, "unit": "USD", "source_concept": "us-gaap:Revenues",
           "dc_code": None, "adjustment": None}
    row.update(kwargs)
    return row


def _facts_with(level_value: float) -> pd.DataFrame:
    """Four filed quarters whose balance-sheet level carries `level_value`."""
    windows = [("2023-01-01", "2023-03-31", "Q1", "2023-05-01"),
               ("2023-04-01", "2023-06-30", "Q2", "2023-08-01"),
               ("2023-07-01", "2023-09-30", "Q3", "2023-11-01"),
               ("2023-10-01", "2023-12-31", "Q4", "2024-02-15")]
    rows = []
    for i, (start, end, label, filed) in enumerate(windows):
        rows.append(_fact(accession_number=f"acc-{i}", fiscal_period=label,
                          period_start=start, period_end=end, filing_date=filed,
                          period_of_report=end, value=100.0 + i,
                          form="10-K" if label == "Q4" else "10-Q"))
        rows.append(_fact(accession_number=f"acc-{i}", field=_LEVEL, fiscal_period=label,
                          duration_type="instant", period_start=None, period_end=end,
                          period_days=None, filing_date=filed, period_of_report=end,
                          value=level_value, source_concept="us-gaap:Assets",
                          form="10-K" if label == "Q4" else "10-Q"))
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
    row = {"ticker": "HCA", "debtToEquity": -3.4, "totalAssets": 50_000_000_000.0,
           "stockholdersEquity": -2_000_000_000.0, "sharesOutstanding": 270_000_000.0}
    codes: list[dict] = []
    _hard_guard("HCA", pd.Timestamp("2024-02-15"), row, codes)

    print(f"\nplanted the HCA shape: debtToEquity=-3.4, stockholdersEquity=-$2bn")
    print(f"  after the guards: debtToEquity={row['debtToEquity']}, "
          f"stockholdersEquity={row['stockholdersEquity']:,.0f}, "
          f"reason codes written={len(codes)}")
    print("  SANITY: NOTHING was nulled. A negative equity is a real accounting position, "
          "and a rule that can be wrong may only ever raise a question -- that is Tier 1's "
          "flag-only `impossible_value`, not a guard.")
    assert row["debtToEquity"] == -3.4
    assert row["stockholdersEquity"] == -2_000_000_000.0
    assert not codes


def test_a_shell_footing_to_exactly_zero_assets_survives() -> None:
    """`totalAssets` guards on `< 0`, not `<= 0`. VRT's pre-merger SPAC is the reason.

    A shell in its first period legitimately foots to zero. Nulling that would destroy a
    correct value to catch nothing -- the guard's whole justification is that no filer could
    report the value and be right, and zero fails that test.
    """
    row = {"ticker": "VRT", "totalAssets": 0.0}
    codes: list[dict] = []
    _hard_guard("VRT", pd.Timestamp("2019-05-01"), row, codes)

    print(f"\nplanted totalAssets = 0.0 (a pre-merger shell) -> "
          f"totalAssets={row['totalAssets']}, codes={len(codes)}")
    print("  SANITY: survives. `< 0` and not `<= 0` -- exactly zero is a real balance sheet.")
    assert row["totalAssets"] == 0.0 and not codes


# --------------------------------------------------------------------------- #
# what MUST be nulled, and what the row records about it                       #
# --------------------------------------------------------------------------- #

def test_a_negative_totalAssets_is_nulled_and_the_value_is_recorded() -> None:
    """The rejected number goes onto the reason-code row as `rejected_value`.

    Not into a log line. A DERIVED cell -- a TTM, a `derived_identity` total -- has no fact row
    anywhere, so without this the refused number is simply gone and "did this guard null
    something correct?" becomes archaeology rather than a query.
    """
    row = {"ticker": "TST", "totalAssets": -1.0}
    codes: list[dict] = []
    _hard_guard("TST", pd.Timestamp("2024-02-15"), row, codes)

    print(f"\nplanted totalAssets = -1 -> value is now {row['totalAssets']}, "
          f"{len(codes)} code row: dc_code={codes[0]['dc_code']!r}, "
          f"rejected_value={codes[0]['rejected_value']}")
    print("  SANITY: nulled, AND the refused number is on the record. A derived cell has no "
          "fact row to go back to, so the reason-code row is the only place it can live.")
    assert row["totalAssets"] is None
    assert len(codes) == 1
    assert codes[0]["dc_code"] == rc.FAILED_HARD_GUARD
    assert codes[0]["rejected_value"] == -1.0


def test_a_zero_share_count_is_nulled() -> None:
    """The three share counts guard on `<= 0`: a company with no shares has no equity to report.

    Different from `totalAssets` on purpose. A zero balance sheet is a real (if unusual) filing;
    a zero share count is a filing error, and every ratio built on it is a division by zero.
    """
    row = {"ticker": "TST", "sharesOutstanding": 0.0, "basicShares": -5.0,
           "dilutedShares": 1_000.0}
    codes: list[dict] = []
    _hard_guard("TST", pd.Timestamp("2024-02-15"), row, codes)

    nulled = sorted(c["field"] for c in codes)
    print(f"\nplanted sharesOutstanding=0, basicShares=-5, dilutedShares=1000 -> "
          f"nulled {nulled}; dilutedShares survived at {row['dilutedShares']}")
    print("  SANITY: `<= 0` for the counts, and only the offending fields move. A guard that "
          "nulls a neighbouring correct cell is the 745-row failure mode.")
    assert nulled == ["basicShares", "sharesOutstanding"]
    assert row["dilutedShares"] == 1_000.0


def test_the_guard_replaces_an_absence_code_rather_than_accumulating_one() -> None:
    """A value that was PRESENT and refused is not `not_disclosed`. One code, not two.

    A row asserting both "the filer did not disclose it" and "we threw the number away" is
    incoherent, and a consumer reading the first would draw the wrong conclusion about our
    coverage.
    """
    row = {"ticker": "TST", "totalAssets": -1.0}
    codes = [{"ticker": "TST", "as_of": pd.Timestamp("2024-02-15"), "field": "totalAssets",
              "dc_code": rc.NOT_DISCLOSED, "combined_into": None, "rejected_value": None}]
    _hard_guard("TST", pd.Timestamp("2024-02-15"), row, codes)

    print(f"\na stale not_disclosed code was present -> after the guard the field carries "
          f"{[c['dc_code'] for c in codes if c['field'] == 'totalAssets']}")
    print("  SANITY: replaced, not accumulated. The cell is not absent; it was refused.")
    assert [c["dc_code"] for c in codes] == [rc.FAILED_HARD_GUARD]


# --------------------------------------------------------------------------- #
# the guards on the real build path                                            #
# --------------------------------------------------------------------------- #

def test_the_guards_run_inside_build_ticker_and_leave_the_grain_intact() -> None:
    """End to end: a filer whose `totalAssets` is negative in every filing.

    Through the real builder, so this also proves the guard does not break the column contract,
    the grain assertions, or the `rejected_value` dtype -- an all-None float column that infers
    as `object` becomes TEXT in Postgres, which is how a real number once landed in the
    database as the string '1997000000.0'.
    """
    built = build_ticker("TST", _facts_with(-1000.0))
    guarded = built.reason_codes[built.reason_codes["dc_code"] == rc.FAILED_HARD_GUARD]

    print(f"\n{len(built.history)} publication event(s) built; "
          f"totalAssets non-null: {int(built.history['totalAssets'].notna().sum())}; "
          f"failed_hard_guard rows: {len(guarded)} carrying rejected_value "
          f"{sorted(set(guarded['rejected_value']))}")
    print(f"  rejected_value dtype: {built.reason_codes['rejected_value'].dtype}")
    print("  SANITY: every negative level was nulled at build time with its number recorded, "
          "and the guards did not disturb the grain or the column contract.")
    assert built.history["totalAssets"].isna().all()
    assert len(guarded) == len(built.history)
    assert set(guarded["rejected_value"]) == {-1000.0}
    assert built.reason_codes["rejected_value"].dtype == float


def test_a_clean_filer_earns_no_hard_guard_rows_and_keeps_the_float_dtype() -> None:
    """The other side: a normal filer, zero guard rows, and `rejected_value` still float64.

    The dtype half is not cosmetic. `store.ensure_table` infers a missing table's column types
    from the FIRST frame it is handed, and almost every ticker produces an entirely-null
    `rejected_value` -- so if that column ever infers as `object`, the column is TEXT forever
    and the first real rejection is stored as a string.
    """
    built = build_ticker("TST", _facts_with(1000.0))
    guarded = built.reason_codes[built.reason_codes["dc_code"] == rc.FAILED_HARD_GUARD]

    print(f"\nclean filer: {len(built.history)} event(s), "
          f"totalAssets non-null {int(built.history['totalAssets'].notna().sum())}, "
          f"failed_hard_guard rows {len(guarded)}, "
          f"rejected_value dtype {built.reason_codes['rejected_value'].dtype} "
          f"(all null: {built.reason_codes['rejected_value'].isna().all()})")
    print("  SANITY: no guard fires on correct data, and the all-null payload column is "
          "still float64 -- an object column here becomes TEXT in Postgres, permanently.")
    assert guarded.empty
    assert built.history["totalAssets"].notna().all()
    assert built.reason_codes["rejected_value"].dtype == float


def test_the_guard_set_is_exactly_the_four_decision_46_names() -> None:
    """Four rules, named. A fifth is a decision, not an edit."""
    print(f"\nHARD_GUARDS: {sorted(HARD_GUARDS)}")
    print("  SANITY: exactly four, all impossible-only. Everything else v2 listed became "
          "Tier 1's flag-only `impossible_value`, which reports and never deletes.")
    assert sorted(HARD_GUARDS) == ["basicShares", "dilutedShares", "sharesOutstanding",
                                   "totalAssets"]
