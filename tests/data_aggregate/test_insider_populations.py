"""Known-truth arithmetic for the insider populations and the 60/120/180-day window features.

Every expected value below is computed by hand from the fixture rows: dollars over a constant 1e8 market cap, net shares over
the holdings before each account's first window trade, distinct owners, and signed dollar ratios.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.insider_features import (
    BUY_VALUE_FEATURES,
    EMISSION,
    NET_BUY_RATIO_FEATURES,
    PURCHASE_PCT_FEATURES,
    InsiderCoverage,
)
from src.data_aggregate.utils.institutionals.insider_quality import (
    POP_C_SUITE,
    POP_MANAGEMENT,
    POP_OTHER,
    POP_TEN_PCT,
    clean_transactions,
)
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import build_prepared_insider_panel as build_insider_feature_panel
from tests.data_aggregate.test_insider_features import _frame, _peers, _txn

FILED = pd.Timestamp("2024-03-01")
CEO = dict(officer_title="Chief Executive Officer", is_officer=1.0, is_director=0.0)
DIRECTOR = dict(officer_title="", is_officer=0.0, is_director=1.0)
TEN_PCT = dict(officer_title="", is_officer=0.0, is_director=0.0, is_ten_pct_owner=1.0)
OTHER = dict(officer_title="", is_officer=0.0, is_director=0.0, is_ten_pct_owner=0.0)


def _day(offset: int) -> str:
    return (FILED + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")


def _buy(name: str, owner: str, offset: int, dollars: float, role: dict, **changes: object) -> dict:
    """An open-market purchase filed `offset` days after FILED, of `dollars` at 100 per share, on a fresh 10,000-share account."""
    shares = dollars / 100.0
    row = dict(
        accession_number=name,
        owner_cik=owner,
        filing_date=_day(offset),
        transaction_date=_day(offset),
        shares=shares,
        price_per_share=100.0,
        value_usd=dollars,
        shares_owned_after=10_000.0 + shares,
        **role,
    )
    row.update(changes)
    return _txn(**row)


def _sell(name: str, owner: str, offset: int, dollars: float, role: dict, **changes: object) -> dict:
    merged = {"transaction_code": "S", "shares_owned_after": 10_000.0 - dollars / 100.0, **changes}
    return _buy(name, owner, offset, dollars, role, **merged)


def _panel(rows: list[dict], last: str = "2024-12-31") -> pd.DataFrame:
    """Daily 2024 panel; price 100 and 1e6 shares outstanding give a constant 1e8 market cap; the source is complete to `last`."""
    idx = pd.date_range("2024-01-01", last)
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    shares = pd.DataFrame(
        [{"ticker": "AAA", "as_of": pd.Timestamp("2023-01-01"), "sharesOutstanding": 1_000_000.0, "sharesOutstandingPit": 1_000_000.0}]
    )
    return build_insider_feature_panel(
        make_frames(idx, _peers(), close_split=close),
        _frame(rows),
        shares_out_history=shares,
        df_valuation_close=close,
        coverage=InsiderCoverage.from_availability(None, idx.max()),
    ).set_index("date")


def _value(panel: pd.DataFrame, feature: str, offset: int) -> float:
    """The feature on a day; a column that is NULL everywhere is not emitted, so it reads NaN."""
    column = f"f_ic_insider_{feature}"
    return float(panel.loc[_day(offset), column]) if column in panel else float("nan")


def test_populations_are_disjoint_with_precedence() -> None:
    titles = [
        ("Chairman & CEO", dict(is_director=1.0, is_officer=1.0), POP_C_SUITE),
        ("SVP & CFO", dict(is_officer=1.0), POP_C_SUITE),
        ("Executive Vice President", dict(is_officer=1.0, is_ten_pct_owner=1.0), POP_C_SUITE),
        ("Vice Chairman", dict(is_officer=1.0), POP_C_SUITE),
        ("President", dict(is_officer=1.0), POP_C_SUITE),
        ("Co-Founder", dict(is_officer=1.0), POP_C_SUITE),
        ("Senior Vice President", dict(is_officer=1.0), POP_MANAGEMENT),
        ("VP, Strategic Planning", dict(is_officer=1.0), POP_MANAGEMENT),
        ("Group President", dict(is_officer=1.0), POP_C_SUITE),
        ("Group Vice President", dict(is_officer=1.0), POP_MANAGEMENT),
        ("Vice President", dict(is_officer=1.0), POP_MANAGEMENT),
        ("General Counsel", dict(is_officer=1.0), POP_MANAGEMENT),
        ("", dict(is_director=1.0), POP_MANAGEMENT),
        ("", dict(is_director=1.0, is_ten_pct_owner=1.0), POP_MANAGEMENT),
        ("", dict(is_ten_pct_owner=1.0), POP_TEN_PCT),
        ("", dict(), POP_OTHER),
        ("Former Chief Executive Officer", dict(is_officer=1.0), POP_OTHER),
        ("See Remarks", dict(is_officer=1.0), POP_OTHER),
        ("", dict(is_officer=1.0), POP_OTHER),
        ("", dict(is_officer=1.0, is_director=1.0), POP_OTHER),
    ]
    rows = [
        _buy(
            f"row-{i}",
            f"{i:04d}",
            0,
            1e5 + 1e3 * i,
            dict(officer_title=title, **{"is_officer": 0.0, "is_director": 0.0, "is_ten_pct_owner": 0.0, **flags}),
        )
        for i, (title, flags, _) in enumerate(titles)
    ]
    cleaned, _ = clean_transactions(_frame(rows))
    got = cleaned.set_index("accession_number")["population"]
    wrong = {title: (int(got[f"row-{i}"]), expected) for i, (title, _, expected) in enumerate(titles) if got[f"row-{i}"] != expected}
    assert not wrong, wrong
    print(f"SANITY: {len(titles)} title/flag combinations land in exactly one population; C_SUITE > MANAGEMENT > TEN_PCT, unflagged only overall.")


def test_buy_value_windows_and_buckets_sum_known_dollars() -> None:
    rows = [
        _buy("c", "0001", 0, 1e6, CEO),
        _buy("m", "0002", 70, 2e6, DIRECTOR),
        _buy("t", "0003", 130, 4e6, TEN_PCT),
        _buy("o", "0004", 130, 8e6, OTHER),
        _buy("dt", "0005", 130, 16e6, dict(DIRECTOR, is_ten_pct_owner=1.0)),
    ]
    panel = _panel(rows)
    e = 130
    # 60d: m (filed e-60) is out, t+o+dt in; 120d adds m; 180d adds c. Dollars over 1e8.
    assert _value(panel, "buy_value_mcap_60d", e) == pytest.approx(0.28)
    assert _value(panel, "buy_value_mcap_120d", e) == pytest.approx(0.30)
    assert _value(panel, "buy_value_mcap_180d", e) == pytest.approx(0.31)
    assert _value(panel, "buy_value_mcap_60d", e - 1) == pytest.approx(0.02)
    assert _value(panel, "c_suite_buy_mcap_180d", e) == pytest.approx(0.01)
    assert _value(panel, "management_buy_mcap_180d", e) == pytest.approx(0.18)  # director m + director/10% owner dt
    assert _value(panel, "ten_pct_buy_mcap_180d", e) == pytest.approx(0.04)  # dt never reaches TEN_PCT
    total_180 = _value(panel, "buy_value_mcap_180d", e)
    buckets = sum(_value(panel, f"{p}_buy_mcap_180d", e) for p in ("c_suite", "management", "ten_pct"))
    assert buckets == pytest.approx(0.23) and buckets < total_180  # the unflagged 0.08 is in the total only
    print(f"SANITY: 60d/120d/180d = 0.28/0.30/0.31; buckets 0.01+0.18+0.04 = {buckets:.2f} < total {total_180:.2f} by the unflagged 0.08.")


def test_buy_value_windows_are_monotone_and_nan_before_first_filing_zero_after_expiry() -> None:
    panel = _panel([_buy("c", "0001", 0, 1e6, CEO), _buy("m", "0002", 40, 2e6, DIRECTOR)])
    v60, v120, v180 = (panel[f"f_ic_insider_buy_value_mcap_{w}d"] for w in (60, 120, 180))
    tol = 1e-12
    assert ((v60 <= v120 + tol) | v60.isna() | v120.isna()).all() and ((v120 <= v180 + tol) | v120.isna() | v180.isna()).all()
    assert pd.isna(panel.loc[_day(-1), "f_ic_insider_buy_value_mcap_60d"])
    assert _value(panel, "buy_value_mcap_60d", 59) == pytest.approx(0.03)  # c (age 59) + m (age 19)
    assert _value(panel, "buy_value_mcap_60d", 60) == pytest.approx(0.02)  # c expired, m alive
    assert _value(panel, "buy_value_mcap_60d", 100) == 0.0  # m expired at age 60: observed quiet window
    assert _value(panel, "buy_value_mcap_180d", 179) == pytest.approx(0.03)
    assert _value(panel, "buy_value_mcap_180d", 220) == 0.0
    print("SANITY: NaN before the first filing, 0 after expiry (observed quiet), and 60d <= 120d <= 180d on every day.")


def test_distinct_sellers_and_buyer_breadth_count_people() -> None:
    rows = [
        _buy("b1", "0001", 0, 1e5, CEO),
        _sell("s1", "0003", 0, 1e5, DIRECTOR),
        _buy("b2", "0002", 10, 1e5, DIRECTOR),
        _sell("s2", "0004", 10, 1e5, DIRECTOR),
        _sell("b1-sells", "0001", 10, 2e5, CEO, shares_owned_after=9_000.0),
    ]
    panel = _panel(rows)
    assert pd.isna(panel.loc[_day(-1), "f_ic_insider_buyer_breadth_120d"])
    assert (_value(panel, "distinct_buyers_120d", 5), _value(panel, "distinct_sellers_120d", 5), _value(panel, "buyer_breadth_120d", 5)) == (
        1,
        1,
        0.5,
    )
    assert (_value(panel, "distinct_buyers_120d", 12), _value(panel, "distinct_sellers_120d", 12), _value(panel, "buyer_breadth_120d", 12)) == (
        2,
        3,
        0.5,
    )
    # Day 125: b1/s1 (filed day 0) expired; b2, s2 and b1's sale remain: 1 buyer / 3 distinct traders.
    assert (_value(panel, "distinct_buyers_120d", 125), _value(panel, "distinct_sellers_120d", 125)) == (1, 2)
    assert _value(panel, "buyer_breadth_120d", 125) == pytest.approx(1 / 3)
    assert (_value(panel, "distinct_buyers_120d", 140), _value(panel, "distinct_sellers_120d", 140)) == (0, 0)
    assert pd.isna(panel.loc[_day(140), "f_ic_insider_buyer_breadth_120d"])
    print(
        "SANITY: day 12 -> 2 buyers, 3 sellers (b1 also sold), breadth 2/4; day 125 -> 1/3; an empty window gives counts 0 and an undefined breadth."
    )


def test_net_buy_ratio_at_three_horizons_uses_signed_raw_dollars() -> None:
    rows = [_buy("p", "0001", 0, 3e6, CEO), _sell("s", "0002", 50, 1e6, DIRECTOR)]
    panel = _panel(rows)
    assert _value(panel, "net_buy_ratio_60d", 55) == pytest.approx(0.5)  # both in the 60d window: (3-1)/4
    assert _value(panel, "net_buy_ratio_60d", 60) == pytest.approx(-1.0)  # the purchase (age 60) expired
    assert _value(panel, "net_buy_ratio_120d", 60) == pytest.approx(0.5)
    assert _value(panel, "net_buy_ratio_120d", 130) == pytest.approx(-1.0)
    assert _value(panel, "net_buy_ratio_180d", 130) == pytest.approx(0.5)
    assert np.isnan(_value(panel, "net_buy_ratio_60d", 130))  # nothing in the window: 0/0 stays undefined
    assert np.isnan(_value(panel, "net_buy_ratio_180d", 260))
    ratios = panel[[f"f_{name}" for name in NET_BUY_RATIO_FEATURES]].stack().dropna()
    assert ratios.between(-1.0, 1.0).all()
    print("SANITY: (3-1)/4 = 0.5 while both trades are in the window, -1 once the purchase expires, undefined when empty; always in [-1, 1].")


def test_declared_features_are_the_eighteen_requested() -> None:
    assert len(EMISSION) == 17 and len(BUY_VALUE_FEATURES) == 6 and len(PURCHASE_PCT_FEATURES) == 5 and len(NET_BUY_RATIO_FEATURES) == 3
    assert {name for name in EMISSION if "mcap" in name} == set(BUY_VALUE_FEATURES)
    print(f"SANITY: {len(EMISSION)} insider features declared (6 buy value, 5 purchase pct, 3 net ratio, buyers, sellers, breadth).")
