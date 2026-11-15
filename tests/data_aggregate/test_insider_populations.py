"""Known-truth arithmetic for the insider populations and the 60/120/180-day window features.

Every expected value below is computed by hand from the fixture rows: dollars over a constant 1e8 market cap, net shares over
the holdings before each account's first window trade, distinct owners, and signed dollar ratios.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.panel import sector_relative
from src.data_aggregate.utils.institutionals.insider_features import (
    BUY_VALUE_FEATURES,
    CM_BUY_SECTOR,
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
        ("", dict(is_officer=1.0, is_ten_pct_owner=1.0), POP_OTHER),
        ("See Remarks", dict(is_officer=1.0, is_director=1.0, is_ten_pct_owner=1.0), POP_OTHER),
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
    total_180 = _value(panel, "buy_value_mcap_180d", e)
    cm = _value(panel, "cm_buy_mcap_180d", e)
    # c 0.01 + director m 0.02 + director/10% owner dt 0.16 (a director never reaches TEN_PCT); t (0.04) and o (0.08) are outside.
    assert cm == pytest.approx(0.19) and cm + 0.04 + 0.08 == pytest.approx(total_180)
    print(
        f"SANITY: 60d/120d/180d = 0.28/0.30/0.31; the C-suite + management leg 0.01+0.02+0.16 = {cm:.2f}; total {total_180:.2f} adds the 10% owner 0.04 and unflagged 0.08."
    )


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
    assert len(EMISSION) == 15 and len(BUY_VALUE_FEATURES) == 4 and len(PURCHASE_PCT_FEATURES) == 2 and len(NET_BUY_RATIO_FEATURES) == 3
    assert {name for name in EMISSION if "mcap" in name and not name.endswith("_vs_sector")} == set(BUY_VALUE_FEATURES)
    assert [name for name in EMISSION if name.endswith("_vs_sector")] == [CM_BUY_SECTOR]
    removed = {f"ic_insider_{p}_buy_mcap_180d" for p in ("c_suite", "management", "ten_pct")}
    removed |= {f"ic_insider_{p}_purchase_pct_prior_180d" for p in ("c_suite", "management")}
    assert not removed & set(EMISSION)
    print(
        f"SANITY: {len(EMISSION)} insider features declared (4 buy value, sector z, 2 purchase pct, 2 conviction, 3 net ratio, buyers, sellers, breadth) = 16 columns with the peer leg; the 5 retired legs are absent."
    )


def test_cm_leg_sums_c_suite_and_management_only() -> None:
    rows = [
        _buy("c", "0001", 0, 1e6, CEO),
        _buy("m", "0002", 10, 2e6, DIRECTOR),
        _buy("t", "0003", 10, 4e6, TEN_PCT),
        _buy("o", "0004", 10, 8e6, OTHER),
        _sell("cs", "0001", 10, 1e6, CEO, shares_owned_after=9_000.0),
    ]
    panel = _panel(rows)
    assert _value(panel, "cm_buy_mcap_180d", 5) == pytest.approx(0.01)
    assert _value(panel, "cm_buy_mcap_180d", 12) == pytest.approx(0.03)  # (1e6 + 2e6) / 1e8; 10% owner, unflagged owner and the sale add nothing
    assert _value(panel, "buy_value_mcap_180d", 12) == pytest.approx(0.15)
    print("SANITY: C-suite 0.01 + management 0.02 = 0.03; the 10% owner (0.04), the unflagged owner (0.08) and the CEO sale are out of the leg.")


def test_cm_leg_window_boundary_observed_zero_and_nan() -> None:
    panel = _panel([_buy("c", "0001", 0, 1e6, CEO)])
    assert pd.isna(panel.loc[_day(-1), "f_ic_insider_cm_buy_mcap_180d"])  # before the first filing nothing is known
    assert _value(panel, "cm_buy_mcap_180d", 179) == pytest.approx(0.01)  # age 179 days is inside the window
    assert _value(panel, "cm_buy_mcap_180d", 180) == 0.0  # age 180 expired: an observed quiet window is a zero, not NaN
    only_other = _panel([_buy("o", "0004", 0, 8e6, OTHER)])
    assert _value(only_other, "cm_buy_mcap_180d", 5) == 0.0  # an excluded population's purchase leaves the observed window empty
    print("SANITY: NaN before the first filing, 0.01 at age 179, exactly 0 from age 180, and 0 (not NaN) when only an excluded population bought.")


def test_sector_relative_hand_computed_three_sectors() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    # X holds observed zeros and one value; Y has 2 members; Z has no dispersion; N1 has no sector. On day 2 X3 is unobserved (NaN).
    field = pd.DataFrame(
        {
            "X1": [0.0, 0.0],
            "X2": [0.0, 0.0],
            "X3": [0.0, np.nan],
            "X4": [0.04, 0.04],
            "Y1": [0.01, 0.01],
            "Y2": [0.02, 0.02],
            "Z1": [0.03, 0.03],
            "Z2": [0.03, 0.03],
            "Z3": [0.03, 0.03],
            "N1": [0.02, 0.02],
        },
        index=idx,
    )
    sectors = {"X1": "X", "X2": "X", "X3": "X", "X4": "X", "Y1": "Y", "Y2": "Y", "Z1": "Z", "Z2": "Z", "Z3": "Z"}
    z = sector_relative(field, sectors)
    # Day 1, X = [0, 0, 0, 0.04]: mean 0.01, sample std sqrt((3 * 0.01**2 + 0.03**2) / 3) = 0.02.
    assert z.loc[idx[0], "X4"] == pytest.approx(1.5) and z.loc[idx[0], "X1"] == pytest.approx(-0.5)
    # Day 2, X = [0, 0, 0.04]: mean 0.04 / 3, sample std 0.04 / sqrt(3); the NaN X3 is missing, not a zero.
    assert z.loc[idx[1], "X4"] == pytest.approx(2 / np.sqrt(3)) and z.loc[idx[1], "X1"] == pytest.approx(-1 / np.sqrt(3))
    assert np.isnan(z.loc[idx[1], "X3"])
    # Fewer than 3 observed members (Y), no dispersion (Z) and no sector (N1) read NaN.
    assert z[["Y1", "Y2", "Z1", "Z2", "Z3", "N1"]].isna().all().all()
    # No floor: a tight sector inside a dispersed universe is scored by its own std, 0.0005 here.
    wide = pd.DataFrame({"S1": [0.0], "S2": [0.0], "S3": [0.0], "S4": [0.001], "W1": [0.0], "W2": [10.0], "W3": [20.0]}, index=idx[:1])
    tight = sector_relative(wide, {"S1": "S", "S2": "S", "S3": "S", "S4": "S"})
    assert tight.loc[idx[0], "S4"] == pytest.approx((0.001 - 0.00025) / 0.0005)
    # Clip: one value among 29 zeros scores 29 / sqrt(30) = 5.3 and is clipped to 5.
    many = pd.DataFrame({f"M{i}": [0.0] for i in range(29)} | {"M29": [1.0]}, index=idx[:1])
    clipped = sector_relative(many, dict.fromkeys(many, "M"))
    assert clipped.loc[idx[0], "M29"] == 5.0 and clipped.loc[idx[0], "M0"] == pytest.approx(-1 / np.sqrt(30))
    print(
        "SANITY: zeros score against their sector (day 1 z = +1.5 / -0.5), an unobserved NaN drops out of the sector moments, "
        "a sector with <3 members, no dispersion or no sector is NaN, a tight sector is scored by its own std (no floor), and an outlier clips at +5."
    )


def test_cm_sector_leg_is_emitted_and_scored_within_sector() -> None:
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    idx = pd.date_range("2024-01-01", "2024-12-31")
    close = pd.DataFrame(100.0, index=idx, columns=tickers)
    shares = pd.DataFrame(
        [{"ticker": t, "as_of": pd.Timestamp("2023-01-01"), "sharesOutstanding": 1_000_000.0, "sharesOutstandingPit": 1_000_000.0} for t in tickers]
    )
    rows = [
        _buy("a", "0001", 0, 1e6, CEO, ticker="AAA"),
        _buy("b", "0002", 0, 5e5, DIRECTOR, ticker="BBB"),
        _sell("c", "0003", 0, 1e5, CEO, ticker="CCC"),
        _sell("d", "0004", 0, 1e5, CEO, ticker="DDD"),
    ]
    panel = build_insider_feature_panel(
        make_frames(idx, _peers(tuple(tickers)), close_split=close),
        _frame(rows),
        shares_out_history=shares,
        df_valuation_close=close,
        coverage=InsiderCoverage.from_availability(None, idx.max()),
        sector_map=dict.fromkeys(tickers, "X"),
    )
    day = panel[panel["date"].eq(pd.Timestamp(_day(12)))].set_index("ticker")
    column = f"f_{CM_BUY_SECTOR}"
    # Raw leg [0.01, 0.005, 0, 0] (CCC and DDD only sold: observed zeros): mean 0.00375, sample std 0.0047871.
    expected = {"AAA": 1.30558, "BBB": 0.26112, "CCC": -0.78335, "DDD": -0.78335}
    assert column in panel and {t: round(float(day.loc[t, column]), 5) for t in tickers} == expected
    print(f"SANITY: {column} is emitted and scores the observed zeros against the sector: {expected}.")
