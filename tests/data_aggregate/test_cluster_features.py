"""Cluster 13F features: company books and the per-cluster feature function (REQ-011..013, AC-007).

Exact math uses hand-computed known-truth books; one real read-only check compares the company-level CUSIP mapping with the symbol match on
2024Q4 for a small CIK list per cluster.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv

from src.data_aggregate.utils.manager_profile.lines import LINE_COLUMNS, clean_book, read_lines
from src.data_aggregate.utils.manager_profile.reference import classify_securities, prefix_tickers, sharadar_reference
from src.data_aggregate.utils.price.frames import load_trading_calendar
from src.data_aggregate.utils.sec_thirteen.cluster_config import ClusterSpec, cluster_specs
from src.data_aggregate.utils.sec_thirteen.cluster_features import MARKET_COLUMNS, QuarterBooks, cluster_feature_frame
from src.data_aggregate.utils.sec_thirteen.cluster_lines import BOOK_COLUMNS, company_books
from src.data_aggregate.utils.sec_thirteen.cluster_panel import filing_deadlines
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.config import read_config
from src.utils.db import get_engine
from src.utils.identity_tables import LINE_COLUMNS as TAG_COLUMNS
from src.utils.identity_tables import CusipTags

ROOT = Path(__file__).resolve().parents[2]
PERIOD = pd.Timestamp("2024-12-31")
FILED = PERIOD + pd.Timedelta(days=40)
EIGHT = ("net_breadth", "new_position_rate", "net_share_flow", "top10_chg", "rank_gap", "rank_gap_chg", "stickiness", "cap_proximity")
FOUR = EIGHT[:4]


def _line(cik: str, ticker: str, shares: float, value: float, rank: float = 20.0, total: float = 1000.0, kind: str = "common") -> dict:
    """One company-book row; `rank` and `total` are the manager's rank of the line and its whole common book."""
    return {"cik": cik, "kind": kind, "ticker": ticker, "shares": shares, "value": value, "rank": rank, "book_total": total, "filing_date": FILED}


def _book(lines: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(lines, columns=list(BOOK_COLUMNS))


def _market(rows: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    """`ticker -> (market_cap, market_cap_prev, shares_outstanding, volatility)`."""
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(MARKET_COLUMNS))


def _spec(features: tuple[str, ...] = EIGHT, min_holders: int = 1) -> ClusterSpec:
    return ClusterSpec("fund", 0, features, min_holders)


def _presence(ciks: list[str]) -> list[dict]:
    """A line `Y` held by every manager so each is present in the book (and a ticker with no flow)."""
    return [_line(cik, "Y", 10.0, 10.0) for cik in ciks]


def _market_all(**extra: tuple[float, float, float, float]) -> pd.DataFrame:
    base = {"X": (1e6, 1e6, 1000.0, 0.3), "Y": (5e5, 5e5, 2000.0, 0.2)}
    return _market({**base, **extra})


def _scenario() -> tuple[QuarterBooks, list[str]]:
    """Six managers; ticker X: prev holders m1 m2 m3, cur holders m2 m3 m4 m5, prev4 holders m2 m4; every manager holds Y throughout."""
    ciks = [f"m{i}" for i in range(1, 7)]
    prev = [_line("m1", "X", 100, 50, rank=2), _line("m2", "X", 100, 50, rank=4), _line("m3", "X", 100, 50, rank=9)]
    cur = [
        _line("m2", "X", 150, 100, rank=3),
        _line("m3", "X", 100, 30, rank=12),
        _line("m4", "X", 40, 50, rank=1),
        _line("m5", "X", 60, 20, rank=11),
    ]
    prev4 = [_line("m2", "X", 100, 50), _line("m4", "X", 100, 50)]
    books = QuarterBooks(_book(cur + _presence(ciks)), _book(prev + _presence(ciks)), _book(prev4))
    return books, ciks


def test_breadth_new_rate_flow_top10_stickiness_cap_proximity_equal_hand_computed_truth() -> None:
    books, ciks = _scenario()
    out = cluster_feature_frame(books, ciks, _market_all(), _spec())

    x = out.loc["X"].to_dict()
    # holders 4, new {m4, m5} = 2, exited {m1} = 1, six matched managers; no split: holder ratios 1.5 and 1.0 give a 1.25 median < 1.3.
    assert x["net_breadth"] == pytest.approx((2 - 1) / (4 + 1))
    assert x["new_position_rate"] == pytest.approx(2 / 6)
    assert x["net_share_flow"] == pytest.approx((50 + 0 + 40 + 60 - 100) / 1000)
    assert x["top10_chg"] == pytest.approx((2 - 3) / 6)  # top 10 now m2, m4; before m1, m2, m3
    assert x["stickiness"] == pytest.approx(2 / 4)  # m2 and m4 held it four quarters ago
    assert x["cap_proximity"] == pytest.approx(2 / 4)  # weights 10%, 3%, 5%, 2%
    y = out.loc["Y"].to_dict()
    assert [y[name] for name in ("net_breadth", "new_position_rate", "net_share_flow", "top10_chg", "stickiness", "cap_proximity")] == [0.0] * 6
    assert (out.dtypes == "float32").all()
    print("\n=== SANITY CHECK: hand-computed cluster features ===")
    print(out.round(4).to_string())
    print("Breadth 0.2, new rate 1/3, flow 5%, top-10 change -1/6, stickiness and cap proximity 0.5 for X; Y is flat. Matches the hand truth.")


def test_a_two_for_one_split_is_not_a_flow_and_disagreeing_holders_are_not_a_split() -> None:
    ciks = [f"m{i}" for i in range(1, 9)]
    prev = [_line(c, t, 100, 10) for c in ciks for t in "STU"]
    cur = []
    for i, cik in enumerate(ciks):
        cur += [
            _line(cik, "S", 250 if i == 0 else 200, 10),  # 2-for-1, one holder adds 50 post-split shares
            _line(cik, "T", [150, 200, 250, 300][i % 4], 10),  # no common ratio: every holder is 5%+ from the median
            _line(cik, "U", 120, 10),  # 1.2x: under the 1.3 split threshold
        ]
    market = _market({t: (1e6, 1e6, 1000.0, 0.3) for t in "STU"})
    out = cluster_feature_frame(QuarterBooks(_book(cur), _book(prev), _book([])), ciks, market, _spec(("net_share_flow",)))

    naive_s = (250 + 7 * 200 - 800) / 1000
    assert out.loc["S", "net_share_flow"] == pytest.approx(50 / 1000)
    assert out.loc["T", "net_share_flow"] == pytest.approx(((150 + 200 + 250 + 300) * 2 - 800) / 1000)
    assert out.loc["U", "net_share_flow"] == pytest.approx((8 * 120 - 800) / 1000)
    print("\n=== SANITY CHECK: split factor ===")
    print(out.round(4).to_string())
    print(f"The 2-for-1 ticker reads a flow of 5.0% instead of the naive {naive_s:.0%}; the other two tickers keep their unscaled flows.")


def test_a_manager_absent_a_quarter_enters_neither_the_holders_nor_the_denominators() -> None:
    members = ["m1", "m2", "m3", "m7", "m8"]
    cur = [_line(c, "Z", 100, 10) for c in ("m1", "m2", "m3", "m7")] + _presence(["m1", "m2", "m3", "m7"])
    prev = [_line(c, "Z", 100, 10) for c in ("m1", "m2", "m3", "m8")] + _presence(["m1", "m2", "m3", "m8"])
    out = cluster_feature_frame(QuarterBooks(_book(cur), _book(prev), _book([])), members, _market_all(Z=(1e6, 1e6, 1000.0, 0.3)), _spec(FOUR))

    z = out.loc["Z"].to_dict()
    # m7 has no previous book and m8 no current book: three matched managers, no new holder, no exit.
    assert z["net_breadth"] == 0.0 and z["new_position_rate"] == 0.0 and z["net_share_flow"] == 0.0
    print("\n=== SANITY CHECK: manager absent at t or t-1 ===")
    print(out.loc[["Z"]].round(4).to_string())
    print("m7 (new filer) is not counted as a new holder and m8 (absent now) is not an exit: both are outside the matched set.")


def test_min_holders_nulls_a_ticker_held_or_exited_by_too_few_managers() -> None:
    ciks = ["m1", "m2", "m3", "m4"]
    cur = [_line(c, "P", 100, 10) for c in ciks[:3]] + [_line(c, "Q", 100, 10) for c in ciks[:2]] + _presence(ciks)
    prev = [_line(c, "P", 100, 10) for c in ciks] + [_line(c, "Q", 100, 10) for c in ciks[:3]] + _presence(ciks)
    books = QuarterBooks(_book(cur), _book(prev), _book([]))
    market = _market_all(P=(1e6, 1e6, 1000.0, 0.3), Q=(1e6, 1e6, 1000.0, 0.3))

    strict = cluster_feature_frame(books, ciks, market, _spec(FOUR, min_holders=4))
    loose = cluster_feature_frame(books, ciks, market, _spec(FOUR, min_holders=3))

    # P: 3 holders + 1 exit = 4 managers; Q: 2 holders + 1 exit = 3 managers; Y: 4 holders.
    assert sorted(strict.index) == ["P", "Y"] and sorted(loose.index) == ["P", "Q", "Y"]
    assert loose.loc["Q", "net_breadth"] == pytest.approx(-1 / 3)
    print("\n=== SANITY CHECK: min_holders ===")
    print(
        f"min_holders 4 keeps {sorted(strict.index)}, min_holders 3 keeps {sorted(loose.index)}; a ticker below the minimum has no value (null), never 0."
    )


def test_rank_gap_and_its_change_are_percentile_ranks_of_weight_minus_market_cap() -> None:
    ciks = [f"m{i}" for i in range(1, 5)]
    cur = [_line(c, t, 10, v, total=10.0) for c in ciks for t, v in (("A", 6), ("B", 3), ("C", 1))]
    prev = [_line(c, t, 10, v, total=10.0) for c in ciks for t, v in (("A", 5), ("B", 4), ("C", 1))]
    market = _market({"A": (100, 300, 1000, 0.1), "B": (300, 200, 1000, 0.1), "C": (200, 100, 1000, 0.1)})
    out = cluster_feature_frame(QuarterBooks(_book(cur), _book(prev), _book([])), ciks, market, _spec(("rank_gap", "rank_gap_chg")))

    # Weights .6 .3 .1 rank 1, 2/3, 1/3; caps 100 300 200 rank 1/3, 1, 2/3; before: weights .5 .4 .1 and caps 300 200 100 give gaps 0.
    assert out["rank_gap"].to_numpy() == pytest.approx([2 / 3, -1 / 3, -1 / 3], abs=1e-6)
    assert out["rank_gap_chg"].to_numpy() == pytest.approx([2 / 3, -1 / 3, -1 / 3], abs=1e-6)
    print("\n=== SANITY CHECK: rank gap ===")
    print(out.round(4).to_string())
    print("Gaps 2/3, -1/3, -1/3 against prior-quarter gaps of 0: the change equals the gap, as computed by hand.")


def test_option_intensity_and_flow_times_volatility_follow_their_formulas() -> None:
    ciks = ["m1", "m2", "m3", "m4"]
    cur = [_line("m1", "A", 1200, 40), _line("m2", "A", 1000, 30), _line("m2", "C", 1200, 10), _line("m3", "B", 1200, 10), _line("m4", "Y", 10, 10)]
    cur += [_line("m1", "A", 0, 20, kind="call"), _line("m2", "A", 0, 10, kind="put"), _line("o9", "A", 0, 999, kind="call")]
    prev = [_line("m1", "A", 1000, 40), _line("m2", "A", 1000, 30), _line("m2", "C", 1500, 10), _line("m3", "B", 1000, 10), _line("m4", "Y", 10, 10)]
    market = _market({"A": (1, 1, 2000, 0.1), "B": (1, 1, 1000, 0.2), "C": (1, 1, 1000, 0.3), "Y": (1, 1, 1000, 0.4)})
    books = QuarterBooks(_book(cur), _book(prev), _book([]))
    out = cluster_feature_frame(books, ciks, market, _spec(("net_share_flow", "option_intensity", "flow_x_vol")))

    # Options of managers outside the cluster (o9) never count: A has calls 20 + puts 10 over common value 70.
    assert out.loc["A", "option_intensity"] == pytest.approx(30 / (70 + 30)) and out.loc["B", "option_intensity"] == 0.0
    flows = {"A": 0.1, "B": 0.2, "C": -0.3, "Y": 0.0}
    ranks = {"A": 1 / 4, "B": 2 / 4, "C": 3 / 4, "Y": 1.0}  # percentile rank of volatility among the four tickers
    for ticker, flow in flows.items():
        assert out.loc[ticker, "net_share_flow"] == pytest.approx(flow)
        assert out.loc[ticker, "flow_x_vol"] == pytest.approx(flow * (ranks[ticker] - 0.5), abs=1e-6)
    print("\n=== SANITY CHECK: slots ===")
    print(out.round(4).to_string())
    print("Option intensity 0.30 for A (outside option lines ignored); flow x (volatility rank - 0.5) matches the hand values.")


def test_every_cluster_type_returns_exactly_its_configured_features() -> None:
    books, ciks = _scenario()
    specs = {spec.kind: spec for spec in cluster_specs(read_config(str(ROOT / "configs")).cluster)}
    seen = {}
    for kind, spec in specs.items():
        out = cluster_feature_frame(books, ciks, _market_all(), replace(spec, min_holders=1))
        assert list(out.columns) == list(spec.features) and (out.dtypes == "float32").all()
        seen[kind] = len(spec.features)
    assert seen == {"fund": 8, "bank": 8, "insurance": 8, "broker_dealer": 4, "pension": 4}
    assert "cap_proximity" in specs["fund"].features and "option_intensity" in specs["bank"].features and "flow_x_vol" in specs["insurance"].features
    print("\n=== SANITY CHECK: feature lists by cluster type ===")
    print(f"feature counts {seen}: fund slot cap_proximity, bank option_intensity, insurance flow_x_vol, broker-dealer and pension the first four.")


def test_empty_and_unmatched_clusters_have_no_feature_rows() -> None:
    books, _ = _scenario()
    none = cluster_feature_frame(books, ["nobody"], _market_all(), _spec())
    assert none.empty and list(none.columns) == list(EIGHT) and none.index.name == "ticker"
    no_prev4 = cluster_feature_frame(
        QuarterBooks(books.cur, books.prev, _book([])), [f"m{i}" for i in range(1, 7)], _market_all(), _spec(("stickiness",))
    )
    assert no_prev4["stickiness"].isna().all()
    print("\n=== SANITY CHECK: missing books ===")
    print("A cluster with no matched manager gives an empty frame, and no book four quarters back gives null stickiness (never 0).")


def _tags() -> CusipTags:
    lines = pd.DataFrame(
        [
            ["CG1", "GOOGL", "canonical_current", "GOOGL", 1.0, "2000-01-01", None],
            ["CG2", "GOOGL", "secondary_class", "GOOG", 1.0, "2000-01-01", None],
            ["CAA", "AAA", "canonical_current", "AAA", 1.0, "2000-01-01", None],
        ],
        columns=TAG_COLUMNS,
    )
    cmap = pd.DataFrame({"cusip": ["CBG", "CET", "CAD"], "ticker": ["BIGCO", "ETFX", "ADRX"]})
    return CusipTags(lines, cmap)


def _classified(cik: str, filed: pd.Timestamp, items: list[tuple]) -> pd.DataFrame:
    """Cleaned and classified lines; items are `(cusip, kind, asset, shares, value)`."""
    rows = [
        {"cik": cik, "period": PERIOD, "kind": kind, "key": cusip, "ticker": cusip, "cusip": cusip, "issuer_name": "X", "title_of_class": "COM"}
        | {"filing_date": filed, "shares": shares, "value": value, "asset": asset}
        for cusip, kind, asset, shares, value in items
    ]
    return pd.DataFrame(rows)


def test_company_books_combine_share_classes_rank_before_filtering_and_respect_the_deadline() -> None:
    items = [
        ("CG1", "common", "equity_us", 100.0, 400.0),
        ("CG2", "common", "equity_us", 50.0, 200.0),
        ("CBG", "common", "equity_us", 1000.0, 5000.0),
        ("CET", "common", "etf", 10.0, 300.0),
        ("CAA", "common", "equity_us", 10.0, 100.0),
        ("CAD", "common", "equity_foreign", 10.0, 50.0),
        ("CAA", "call", "options", 0.0, 40.0),
    ]
    on_time = _classified("m1", PERIOD + pd.Timedelta(days=45), items)
    late = _classified("m2", PERIOD + pd.Timedelta(days=46), items)
    out = company_books(pd.concat([on_time, late], ignore_index=True), _tags(), {"GOOGL", "AAA", "ADRX"}, PERIOD + pd.Timedelta(days=45))

    assert list(out.columns) == list(BOOK_COLUMNS) and set(out["cik"]) == {"m1"}
    common = out[out["kind"].eq("common")].set_index("ticker")
    # GOOG and GOOGL are one company line; the 5,000 BIGCO line is outside the universe but takes rank 1 and counts in the book total.
    googl, aaa = common.loc["GOOGL"].to_dict(), common.loc["AAA"].to_dict()
    assert [googl[name] for name in ("shares", "value", "rank", "book_total")] == [150.0, 600.0, 2.0, 6050.0]
    assert [aaa[name] for name in ("shares", "value", "rank")] == [10.0, 100.0, 4.0]
    assert sorted(common.index) == ["AAA", "GOOGL"], "foreign-class lines and non-universe tickers leave the book, ETFs too"
    call = out[out["kind"].eq("call")].iloc[0]
    assert call["ticker"] == "AAA" and call["value"] == 40.0 and math.isnan(call["rank"])
    print("\n=== SANITY CHECK: company books ===")
    print(out.drop(columns="filing_date").to_string(index=False))
    print("Share classes combine (150 sh, 600), BIGCO takes rank 1 off-universe, the filing one day late is excluded, options keep their rows.")


def test_company_books_keep_the_filings_made_by_the_rolled_deadline_of_a_weekend_day_45() -> None:
    period = pd.Timestamp("2021-09-30")  # day 45 is Sunday 2021-11-14: the SEC deadline (and the publication clock) rolls to Monday 11-15
    days = pd.bdate_range("2021-01-04", "2022-03-31")
    deadline = filing_deadlines(pd.DatetimeIndex([period]), days).iloc[0]
    assert deadline == pd.Timestamp("2021-11-15")
    items = [("CAA", "common", "equity_us", 10.0, 100.0)]
    frames = []
    for cik, filed in (("on_day_45", "2021-11-14"), ("on_deadline", "2021-11-15"), ("day_after", "2021-11-16")):
        frames.append(_classified(cik, pd.Timestamp(filed), items).assign(period=period))
    out = company_books(pd.concat(frames, ignore_index=True), _tags(), {"AAA"}, deadline)

    assert sorted(out["cik"]) == ["on_day_45", "on_deadline"]
    print("\n=== SANITY CHECK: rolled deadline ===")
    print(out[["cik", "ticker", "filing_date"]].to_string(index=False))
    print("Filings made on day 45 (Sunday) and on the rolled deadline (Monday) count; the one a day later does not.")


# --------------------------------------------------------------------------- real data


def _frame(df: pd.DataFrame | None) -> pd.DataFrame:
    assert df is not None
    return df


def _live_store() -> DataStore:
    load_dotenv(find_dotenv(usecwd=True))  # a worktree has no .env of its own: use the repository one
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable: {type(exc).__name__}")
    return DataStore(engine)


def _asof(frame: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    """The last row of `frame` on or before `day`."""
    return frame.loc[:day].iloc[-1]


def _market_at(store: DataStore, period: pd.Timestamp) -> pd.DataFrame:
    """Quarter-end market cap (now and one quarter back), shares outstanding and 250-day volatility of the priced universe."""
    days = pd.date_range(period - pd.Timedelta(days=400), period, freq="D").strftime("%Y-%m-%d").tolist()
    columns = ["date", "ticker", "close_total", "mcap_pit", "shares_outstanding_pit"]
    prices = _frame(store.load(Tables.cube_part_prices, columns=columns, where={"date": days}))
    prices["date"] = pd.to_datetime(prices["date"])
    close, cap = (prices.pivot(index="date", columns="ticker", values=name).sort_index() for name in ("close_total", "mcap_pit"))
    shares = prices.pivot(index="date", columns="ticker", values="shares_outstanding_pit").sort_index()
    before = period - pd.offsets.QuarterEnd(1)
    vol = close.pct_change().rolling(250, min_periods=120).std()
    return pd.DataFrame(
        {
            "market_cap": _asof(cap, period),
            "market_cap_prev": _asof(cap, before),
            "shares_outstanding": _asof(shares, period),
            "volatility": _asof(vol, period),
        }
    )[list(MARKET_COLUMNS)]


def test_real_quarter_mapped_lines_cover_at_least_the_symbol_match_and_features_stay_in_bounds() -> None:
    store = _live_store()
    cfg = read_config(str(ROOT / "configs"))
    specs = cluster_specs(cfg.cluster)
    before, prev4 = PERIOD - pd.offsets.QuarterEnd(1), PERIOD - pd.offsets.QuarterEnd(4)

    # The labels of the quarter before name each cluster's managers; a small deterministic list per cluster.
    labels = _frame(store.load(Tables.sec_13f_manager_details, columns=["cik", "cluster"], where={"period": before.strftime("%Y-%m-%d")}))
    labels = labels[labels["cluster"].ge(0)].sort_values("cik")
    members = {int(cast(float, c)): sorted(group["cik"].head(10)) for c, group in labels.groupby("cluster")}
    ciks = sorted({cik for group in members.values() for cik in group})

    tickers = _frame(store.load(Tables.sharadar_tickers, columns=["permaticker", "ticker", "category", "sector", "lastpricedate"]))
    reference = sharadar_reference(tickers)
    prefixes = prefix_tickers(_frame(store.load(Tables.cusip_ticker_map)))
    tags = CusipTags.load(store)
    market = _market_at(store, PERIOD)
    universe = set(market.dropna(subset=["market_cap"]).index)

    deadlines = filing_deadlines(pd.DatetimeIndex([PERIOD, before, prev4]), load_trading_calendar(store))
    books: dict[pd.Timestamp, pd.DataFrame] = {}
    symbol_cover = mapped_cover = 0.0
    for period in (PERIOD, before, prev4):
        raw = read_lines(store, ciks, [period])
        assert list(raw.columns) == list(LINE_COLUMNS)
        classified = classify_securities(clean_book(raw, cfg.build_cube.manager_profile, prefixes), reference, cfg.build_cube.manager_profile)
        books[period] = company_books(classified, tags, universe, deadlines[period])
        if period == PERIOD:
            filed = pd.to_datetime(classified["filing_date"]) <= deadlines[PERIOD]
            common = classified[classified["kind"].eq("common") & filed]
            total = common["value"].sum()
            symbol_cover = common.loc[common["asset"].eq("equity_us") & common["ticker"].isin(universe), "value"].sum() / total
            mapped_cover = books[period].query("kind == 'common'")["value"].sum() / total

    quarter = QuarterBooks(books[PERIOD], books[before], books[prev4])
    bounded = {
        "net_breadth": 1.0,
        "new_position_rate": 1.0,
        "top10_chg": 1.0,
        "stickiness": 1.0,
        "cap_proximity": 1.0,
        "rank_gap": 1.0,
        "option_intensity": 1.0,
    }
    lines = []
    for spec in specs:
        out = cluster_feature_frame(quarter, members.get(spec.cluster_id, []), market, replace(spec, min_holders=2))
        for name, limit in bounded.items():
            if name in out:
                assert out[name].abs().max(skipna=True) <= limit + 1e-6 if len(out) else True, (spec.cluster_id, name)
        lines.append(
            f"cluster {spec.cluster_id:>2} ({spec.kind:<13}) {len(members.get(spec.cluster_id, [])):>2} CIKs {len(out):>4} tickers {out.notna().mean().min() if len(out) else 0.0:.2f} min column fill"
        )
    assert mapped_cover >= symbol_cover - 0.02, (
        f"mapped coverage {mapped_cover:.3f} is worse than the symbol match {symbol_cover:.3f} by more than 2 points"
    )
    print(f"\n=== SANITY CHECK: real quarter {PERIOD.date()} ({len(ciks)} CIKs) ===")
    print("\n".join(lines))
    print(
        f"Universe-line value coverage: mapped (CusipTags, company level) {mapped_cover:.4f} vs symbol match {symbol_cover:.4f}; features inside their bounds."
    )
