"""Insider transaction-date repair
(src/data_extract/utils/prices/fetch_insider_transactions.py::_repair_transaction_dates).

A Form 3/4/5 discloses a COMPLETED transaction, so `transaction_date <= filing_date` always.
That makes the field self-validating. The live table breached it 14 times in 1.39M rows, two ways:
  * lost century -- `0015-11-23` filed 2015-11-25, `0024-02-01` filed 2024-02-05 (a 2-digit
    source year read as year 15 / 24 AD) -> repairable from the filing's century;
  * filer typo   -- `2028-05-24` filed 2024-05-28, `2031-01-29` filed 2021-02-02 -> no safe
    reading, so the date is NULLED rather than guessed.
Small in count, but a transaction stamped 2031 dominates any recency-weighted insider feature.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.prices.fetch_insider_transactions import _repair_transaction_dates


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_date": pd.to_datetime([r[0] for r in rows], errors="coerce"),
        "filing_date": pd.to_datetime([r[1] for r in rows], errors="coerce"),
    })


def test_lost_century_is_lifted_into_the_filing_century():
    """Real cases from the live table."""
    out = _repair_transaction_dates(_frame([
        ("0015-11-23", "2015-11-25"),      # NDSN
        ("0024-02-01", "2024-02-05"),      # SPGI
        ("0013-06-02", "2014-02-12"),      # TAP — crosses into the prior year, still <= filing
    ]))
    got = list(out["transaction_date"].dt.strftime("%Y-%m-%d"))
    assert got == ["2015-11-23", "2024-02-01", "2013-06-02"], got
    assert (out["transaction_date"] <= out["filing_date"]).all()
    print("\n=== SANITY CHECK: lost century repaired ===")
    for a, b in zip(["0015-11-23", "0024-02-01", "0013-06-02"], got):
        print(f"  {a} -> {b}")
    print("  century taken from the filing date; every result <= its filing. Validated.")


def test_post_filing_dates_are_nulled_not_guessed():
    """A transaction cannot post-date its own filing; there is no safe correction, so blank it."""
    out = _repair_transaction_dates(_frame([
        ("2028-05-24", "2024-05-28"),      # TMUS — day/year digits transposed
        ("2031-01-29", "2021-02-02"),      # AMP
        ("2029-08-12", "2019-08-13"),      # MCHP
    ]))
    assert out["transaction_date"].isna().all(), out["transaction_date"].tolist()
    print("\n=== SANITY CHECK: impossible dates nulled ===")
    print("  2028-05-24/2024-05-28, 2031-01-29/2021-02-02, 2029-08-12/2019-08-13 -> all NULL")
    print("  no guessing: the amounts still count, only the timing is unknown. Validated.")


def test_valid_dates_are_untouched():
    """The guard must be inert on good data — including a genuinely old transaction."""
    rows = [("2024-05-24", "2024-05-28"), ("2011-03-01", "2011-03-03"),
            ("1995-06-15", "1995-06-20"), ("2024-05-28", "2024-05-28")]  # same-day is legal
    out = _repair_transaction_dates(_frame(rows))
    assert list(out["transaction_date"].dt.strftime("%Y-%m-%d")) == [r[0] for r in rows]
    print("\n=== SANITY CHECK: good dates untouched ===")
    print(f"  {len(rows)} valid rows (incl. a 1995 transaction and a same-day filing) unchanged. "
          "Validated.")


def test_missing_dates_and_empty_frame_are_safe():
    out = _repair_transaction_dates(_frame([(None, "2024-05-28"), ("2024-05-24", None)]))
    assert out["transaction_date"].isna().iloc[0]
    assert out["transaction_date"].notna().iloc[1]          # no filing date -> nothing to check
    assert _repair_transaction_dates(pd.DataFrame()).empty
    print("\n=== SANITY CHECK: NULL-tolerant ===")
    print("  a missing transaction_date stays NULL; a missing filing_date leaves the "
          "transaction alone; an empty frame is a no-op. Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
