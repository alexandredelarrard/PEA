"""Known-truth fixtures: 13F rows attach to the canonical ticker by dated `security_master` lines and convert by F."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.common.class_lines import cusip_windows
from src.data_aggregate.utils.sec_thirteen.holdings_clean import attach_tickers
from src.data_aggregate.utils.sec_thirteen.institutional_books import compact_institutional_books
from src.data_aggregate.utils.sec_thirteen.institutional_features import _daily_owned_shares

# The frozen master's own lines (security_master, source ftd) for the four names under test.
_LINES = pd.DataFrame(
    [
        ("38259P508", "GOOGL", "GOOG", "class_A", "canonical_predecessor", 1.0, "2009-06-26", "2014-04-01"),
        ("38259P508", "GOOGL", "GOOGL", "class_A", "canonical_predecessor", 1.0, "2014-04-02", "2015-10-01"),
        ("02079K305", "GOOGL", "GOOGL", "class_A", "canonical_current", 1.0, "2015-10-01", None),
        ("02079K107", "GOOGL", "GOOG", "class_C", "secondary_class", 1.0, "2015-10-01", None),
        ("G0408V102", "AON", "AON", "class_A", "canonical_current", 1.0, "2012-03-29", "2020-03-30"),
        ("G0408V102", "AON", "AON", "class_A", "excluded", 1.0, "2020-03-30", "2020-03-31"),
        ("G0403H108", "AON", "AON", "class_A", "canonical_current", 1.0, "2020-03-30", None),
        ("92556H206", "PSKY", "VIAC", "class_B", "secondary_class", 1.0, "2019-12-03", "2022-02-17"),
        ("92556H206", "PSKY", "PARA", "class_B", "secondary_class", 1.0, "2022-02-16", "2025-08-07"),
        ("084670207", "BRK-B", "BRK-B", "class_B", "canonical_current", 1.0, "2009-06-29", None),
        ("084670108", "BRK-B", "BRK-A", "class_A", "secondary_class", 1500.0, "2010-01-21", None),
    ],
    columns=["cusip", "canonical_company", "market_symbol", "security_class", "lineage_role", "conversion_ratio", "valid_from", "valid_to"],
)
# The static map as frozen: it misses the old Alphabet and AON CUSIPs and names the secondary classes by their own symbol.
_MAP = pd.DataFrame({"cusip": ["38259P508", "G0408V102", "G0403H108", "02079K305", "02079K107", "084670108", "084670207"]}).assign(
    ticker=[None, None, "AON", "GOOGL", "GOOG", "BRK-A", "BRK-B"]
)


def _holdings(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["cusip", "period", "shares"])
    return df.assign(period=pd.to_datetime(df["period"]), cik="0000000001", value_usd=df["shares"] * 10.0, call_value=0.0, put_value=0.0)


def test_old_alphabet_cusip_is_attributed_to_googl_in_2013() -> None:
    held = attach_tickers(_holdings([("38259P508", "2013-12-31", 10.0), ("38259P508", "2015-06-30", 10.0)]), cusip_windows(_LINES, _MAP), ("GOOGL",))
    assert held["ticker"].tolist() == ["GOOGL", "GOOGL"]
    print("SANITY: the old Alphabet CUSIP 38259P508, unmapped by cusip_ticker_map, is GOOGL on 2013-12-31 and 2015-06-30 by its dated master lines.")


def test_aon_overlap_day_keeps_one_line() -> None:
    rows = [("G0408V102", "2020-03-30", 5.0), ("G0403H108", "2020-03-30", 7.0), ("G0408V102", "2019-12-31", 3.0), ("92556H206", "2022-02-16", 4.0)]
    held = attach_tickers(_holdings(rows), cusip_windows(_LINES, _MAP), ("AON", "PSKY"))
    assert len(held) == len(rows)
    assert held["ticker"].tolist()[1:] == ["AON", "AON", "PSKY"] and pd.isna(held["ticker"].iloc[0])
    print(
        "SANITY: on AON's 2020-03-30 boundary only the new CUSIP's line is AON (old CUSIP's line has ended); PSKY's overlapping class lines give one row."
    )


def test_goog_ratio_one_adds_to_googl() -> None:
    raw = _holdings([("02079K305", "2024-12-31", 100.0), ("02079K107", "2024-12-31", 50.0)])
    compact = compact_institutional_books(attach_tickers(raw, cusip_windows(_LINES, _MAP), ("GOOGL",)))
    assert compact["shares"].tolist() == [150.0]
    assert compact.index.get_level_values("ticker").tolist() == ["GOOGL"]
    print("SANITY: one manager's 100 GOOGL + 50 GOOG (ratio 1) aggregate to 150 GOOGL shares; no GOOG output line.")


def test_brk_a_is_not_added_to_brk_b() -> None:
    rows = [("084670207", "2024-12-31", 100.0), ("084670108", "2024-12-31", 2.0), ("02079K305", "2024-12-31", 10.0), ("02079K107", "2024-12-31", 5.0)]
    windows = cusip_windows(_LINES, _MAP)
    compact = compact_institutional_books(attach_tickers(_holdings(rows), windows, ("BRK-B", "GOOGL")))
    assert compact["shares"].droplevel(["period", "cik"]).to_dict() == {"BRK-B": 100.0, "GOOGL": 15.0}
    assert attach_tickers(_holdings(rows), windows)["ticker"].tolist() == ["BRK-B", "BRK-A", "GOOGL", "GOOGL"]
    print("SANITY: in one book GOOG (ratio 1) joins GOOGL (15) while BRK-A (ratio 1,500) stays its own off-universe line; BRK-B keeps 100.")


def test_ownership_across_a_split_equals_the_unsplit_economics() -> None:
    index = pd.bdate_range("2022-06-29", "2022-07-20")
    split = index >= pd.Timestamp("2022-07-18")
    basis = pd.DataFrame({"GOOGL": [20.0 if after else 1.0 for after in split]}, index=index)
    as_of = pd.Timestamp("2022-07-14")
    books = pd.DataFrame({"ticker": ["GOOGL"], "period": pd.to_datetime(["2022-06-30"]), "as_of": [as_of], "inst_shares": [5.0]})
    owned = _daily_owned_shares(books, index, basis)["GOOGL"].loc[as_of:]
    shares_out = pd.Series([1000.0 if after else 50.0 for after in split], index=index).loc[as_of:]
    assert owned.tolist() == [5.0, 5.0, 100.0, 100.0, 100.0]
    assert (owned / shares_out).tolist() == pytest.approx([0.1] * 5)
    print("SANITY: 5 shares filed for 2022-06-30 read 100 after GOOGL's 20:1 split on 2022-07-18; ownership stays 10% across the split.")
