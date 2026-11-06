"""U-67: `insider_transactions` share quantities are stored in canonical-class shares at extraction.

Known truth on Berkshire's dated `universe_cusip` lines (one Class A = 30 Class B before 2010-01-21, 1,500 after): a
non-derivative Class A row of 1 share at $120,000 with 10 shares after is stored as 1,500 B shares at $80 with 15,000
after in 2012, and 30 B shares at $4,000 with 300 after in 2008; the dollar value is unchanged. Class B rows,
derivative rows and every class without a contractual ratio (GOOGL Class C) are stored as filed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_extract.utils.common.identity import SecurityHit
from src.data_extract.utils.edgar_extract.insider_common import canonical_class_units

BRK_A = SecurityHit("C084670108", "BRK-B", "secondary_class", "class_A", 30.0)
BRK_A_SPLIT = SecurityHit("C084670108", "BRK-B", "secondary_class", "class_A", 1500.0)
BRK_B = SecurityHit("C084670702", "BRK-B", "canonical_current", "class_B", 1.0)
GOOG = SecurityHit("C02079K107", "GOOGL", "secondary_class", "class_C", 1.0)
IDENTITY: Any = SimpleNamespace(
    securities_by_cusip={
        "084670108": ((pd.Timestamp("1998-12-31"), pd.Timestamp("2010-01-21"), BRK_A), (pd.Timestamp("2010-01-21"), None, BRK_A_SPLIT)),
        "084670702": ((pd.Timestamp("1998-12-31"), None, BRK_B),),
        "02079K107": ((pd.Timestamp("2015-10-01"), None, GOOG),),
    }
)


def _row(ticker: str, kind: str, title: str, day: str, shares: float, price: float, after: float) -> dict[str, Any]:
    return {
        "ticker": ticker, "security_type": kind, "security_title": title, "economic_date": pd.Timestamp(day),
        "shares": shares, "price_per_share": price, "value_usd": shares * price, "shares_owned_after": after,
    }  # fmt: skip


def test_berkshire_class_a_rows_are_stored_in_class_b_shares_at_the_dated_ratio() -> None:
    rows = pd.DataFrame(
        [
            _row("BRK-B", "nonderiv", "Class A Common Stock", "2012-05-01", 1.0, 120_000.0, 10.0),
            _row("BRK-B", "nonderiv", "Class A Common Stock", "2008-05-01", 1.0, 120_000.0, 10.0),
            _row("BRK-B", "nonderiv", "Class B Common Stock", "2012-05-01", 100.0, 80.0, 1_000.0),
            _row("BRK-B", "deriv", "Class A Common Stock", "2012-05-01", 1.0, 0.0, 10.0),
            _row("GOOGL", "nonderiv", "Class C Capital Stock", "2020-05-01", 5.0, 1_400.0, 50.0),
        ]
    )

    out = canonical_class_units(rows, IDENTITY)

    assert out[["shares", "price_per_share", "shares_owned_after"]].values.tolist() == [
        [1_500.0, 80.0, 15_000.0],
        [30.0, 4_000.0, 300.0],
        [100.0, 80.0, 1_000.0],
        [1.0, 0.0, 10.0],
        [5.0, 1_400.0, 50.0],
    ]
    assert out["value_usd"].tolist() == rows["value_usd"].tolist()
    print("\n=== SANITY CHECK: insider Class A rows of Berkshire in Class B shares ===")
    print("  2012: 1 A @ $120,000 (10 after) -> 1,500 B @ $80 (15,000 after); 2008: -> 30 B @ $4,000 (300 after)")
    print("  Class B, derivative and GOOGL Class C rows unchanged; every dollar value unchanged. Validated.")
