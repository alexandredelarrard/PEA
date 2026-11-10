"""The cube reads insider shares as filed and applies the class ratio (P3, REQ-006, AC-005).

Known truth on Berkshire's dated `universe_cusip` lines (one Class A = 30 Class B before 2010-01-21, 1,500 after): a
Class A purchase of 2 shares at $700,000 in 2024 reaches the cleaner and the builders as 3,000 Class B shares at
$466.67, and one Class A share bought in 2008 as 30; the dollar value is unchanged. Class B rows, derivative rows and
GOOGL Class C (ratio 1) read as filed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals import inputs
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from src.data_store.schema import Tables
from tests.conftest import _store
from tests.data_aggregate.insider_pit_reference import row
from tests.data_aggregate.test_ftd_series import LINE_READ

BRK_A, BRK_B, GOOG = "084670108", "084670702", "02079K107"
_KEY = ["accession_number", "security_type", "row_sequence"]


def _line(cusip: str, company: str, klass: str, role: str, ratio: float, start: str, end: str | None) -> dict:
    return {
        "canonical_company": company, "market_symbol": company, "conversion_ratio": ratio, "valid_from": pd.Timestamp(start),
        "valid_to": pd.Timestamp(end) if end else pd.NaT, "security_id": f"C{cusip}", "source": "ftd", "source_symbol": company,
        "security_class": klass, "lineage_role": role, "cusip": cusip,
    }  # fmt: skip


def _lines() -> pd.DataFrame:
    """The `universe_cusip` lines as the institutionals step reads them once."""
    return pd.DataFrame(
        [
            _line(BRK_A, "BRK-B", "class_A", "secondary_class", 30.0, "1998-12-31", "2010-01-21"),
            _line(BRK_A, "BRK-B", "class_A", "secondary_class", 1_500.0, "2010-01-21", None),
            _line(BRK_B, "BRK-B", "class_B", "canonical_current", 1.0, "2010-01-19", None),
            _line(GOOG, "GOOGL", "class_C", "secondary_class", 1.0, "2015-10-01", None),
        ]
    )


def _trade(accession: str, ticker: str, title: str, day: str, shares: float, price: float, kind: str = "nonderiv") -> dict:
    filed = (pd.Timestamp(day) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    return row(
        accession_number=accession, ticker=ticker, security_title=title, security_type=kind, transaction_date=day,
        economic_date=day, filing_date=filed, shares=shares, price_per_share=price, shares_owned_after=10 * shares,
        lineage_role="canonical_current",
    )  # fmt: skip


def test_a_berkshire_class_a_purchase_reaches_the_builders_in_class_b_shares(sqlite_store) -> None:
    rows = pd.DataFrame(
        [
            _trade("a-2024", "BRK-B", "Class A Common Stock", "2024-03-04", 2.0, 700_000.0),
            _trade("a-2008", "BRK-B", "Class A Common Stock", "2008-05-01", 1.0, 120_000.0),
            _trade("b-2024", "BRK-B", "Class B Common Stock", "2024-03-04", 100.0, 400.0),
            _trade("d-2024", "BRK-B", "Class A Common Stock", "2024-03-04", 1.0, 0.0, kind="deriv"),
            _trade("c-2020", "GOOGL", "Class C Capital Stock", "2020-05-01", 5.0, 1_400.0),
        ]
    )
    sqlite_store.save(Tables.insider_transactions, rows)

    source = inputs.load_insider_transactions(sqlite_store, ["BRK-B", "GOOGL"], _lines())
    assert source is not None
    loaded = source.transactions.set_index("accession_number")
    df_trades, _ = clean_transactions(source.transactions)
    built = df_trades.set_index("accession_number")

    # the frame the builders receive: canonical-class shares, the price per canonical share, the dollars as filed
    assert built.loc["a-2024", "shares_n"] == 3_000.0
    assert np.isclose(built.loc["a-2024", "price_per_share"], 700_000.0 / 1_500.0)
    assert built.loc["a-2024", "value_usd"] == 1_400_000.0
    assert built.loc["a-2008", "shares_n"] == 30.0
    assert built.loc["b-2024", "shares_n"] == 100.0 and built.loc["c-2020", "shares_n"] == 5.0
    assert loaded.loc["a-2024", "shares_owned_after"] == 30_000.0
    assert loaded.loc["d-2024", "shares"] == 1.0, "a derivative row reads as filed"
    print("\n=== SANITY CHECK: insider Class A rows of Berkshire reach the builders in Class B shares ===")
    print("  2024: 2 A @ $700,000 -> 3,000 B @ $466.67 ($1.4M unchanged); 2008: 1 A -> 30 B (ratio 30 before 2010-01-21)")
    print("  Class B, derivative and GOOGL Class C rows read as filed. Validated.")


def test_real_stored_class_rows_reach_the_builders_in_canonical_shares() -> None:
    """Real rows (read-only): every non-derivative class-titled row of a company with a ratio line reads as stored
    shares x the line's ratio on its economic date, through the loader and `clean_transactions`."""
    store = _store()
    lines = store.load(Tables.universe_cusip, columns=LINE_READ, where={"source": "ftd"}, optional=True)
    if lines is None or not store.exists(Tables.insider_transactions):
        pytest.skip("no universe_cusip or no insider_transactions: run identity and the insider ingest first")
    ratio_lines = lines.loc[pd.to_numeric(lines["conversion_ratio"]).ne(1.0)]
    companies = sorted(set(ratio_lines["canonical_company"].dropna()))
    stored = store.load(
        Tables.insider_transactions, columns=[*_KEY, "ticker", "security_title", "economic_date", "shares"], where={"ticker": companies}
    )
    if not companies or stored is None or stored.empty:
        pytest.skip("no insider rows of a company with a class at a contractual ratio")

    source = inputs.load_insider_transactions(store, companies, lines)
    assert source is not None
    df_trades, _ = clean_transactions(source.transactions)

    # expected: the stored row's ratio from the line of its ticker and title class holding its economic date
    klass = "class_" + stored["security_title"].astype("string").str.extract(r"(?i)\b(?:CLASS|SERIES)\s+([A-Z])\b", expand=False).str.upper()
    hits = (
        stored.assign(security_class=klass)
        .reset_index()
        .merge(ratio_lines, left_on=["ticker", "security_class"], right_on=["canonical_company", "security_class"])
    )
    day = pd.to_datetime(hits["economic_date"])
    hits = hits.loc[
        hits["security_type"].eq("nonderiv")
        & day.ge(pd.to_datetime(hits["valid_from"]))
        & (day.lt(pd.to_datetime(hits["valid_to"])) | hits["valid_to"].isna())
    ]
    if hits.empty:
        pytest.skip("no stored non-derivative row of a ratio class")
    expected = hits.set_index(_KEY)["shares"].astype(float) * hits.set_index(_KEY)["conversion_ratio"].astype(float)
    got = source.transactions.set_index(_KEY)["shares"].astype(float).reindex(expected.index)
    assert np.allclose(got, expected, rtol=1e-12)
    built = df_trades.set_index(_KEY)["shares_n"].astype(float)
    reached = expected.index.intersection(built.index)
    assert np.allclose(built.reindex(reached), expected.reindex(reached), rtol=1e-12)
    print(f"\nSANITY (real rows): {len(expected)} class row(s) of {companies} read as stored x ratio; {len(reached)} reach the")
    print(
        f"  builders: {[(k[0], s, b) for k, s, b in zip(reached, hits.set_index(_KEY)['shares'].reindex(reached), built.reindex(reached), strict=True)]}"
    )
