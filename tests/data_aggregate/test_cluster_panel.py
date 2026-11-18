"""Cluster panel: the publication clock, the daily expansion with a carry limit and the incremental `StepCubeSecThirteen` (AC-008, AC-010, AC-011).

The clock and the expansion are exact on hand-built frames. The step runs on an in-memory SQLite world of ten managers holding three tickers over
eight report quarters (known holdings: X flows +100 shares per manager and quarter), labelled cluster 0 except one manager who moves to cluster 1
after 2023-12-31, so the full build, both incremental nights, the column-set stop, the consistency stop and the missing-label stop are checked on
known values. No DAG, cube build, PostgreSQL or configs/ write is made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.sec_thirteen.cluster_config import cluster_specs, column_names
from src.data_aggregate.utils.sec_thirteen.cluster_panel import expand_daily, publication_dates
from src.data_store.schema import Tables
from src.utils.config import read_config
from src.utils.institutional_vintages import availability_date
from tests.canonical_schema import create_canonical

ROOT = Path(__file__).resolve().parents[2]
PERIODS = [pd.Timestamp(p) for p in ("2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31")]
MANAGERS = [f"{n:010d}" for n in range(1, 11)]
MOVER = MANAGERS[-1]  # cluster 0 through 2023-09-30, cluster 1 from 2023-12-31 (so out of the members of the quarters from 2024-03-31)
PRICES = {"X": 50.0, "Y": 20.0, "Z": 10.0}
SHARES_OUT = 100_000.0
FLOW = "f_ic_cluster_0_net_share_flow"
STICKY = "f_ic_cluster_0_stickiness"
DAYS = pd.bdate_range("2022-01-03", "2025-03-31")


def _pub(period: str) -> pd.Timestamp:
    return cast(pd.Timestamp, publication_dates(pd.DatetimeIndex([period]), DAYS, 7).iloc[0])


def test_publication_date_is_the_deadline_day_plus_the_buffer_on_a_trading_day() -> None:
    periods = pd.DatetimeIndex(pd.date_range("2022-03-31", "2025-03-31", freq="QE"))
    published = publication_dates(periods, DAYS, 7)
    expected = {"2022-03-31": "2022-05-23", "2022-06-30": "2022-08-22", "2024-09-30": "2024-11-21", "2024-12-31": "2025-02-21"}
    for period, day in expected.items():
        assert published[pd.Timestamp(period)] == pd.Timestamp(day), period
    # every published quarter is at least 7 days after its 45-day deadline and on a trading day; a deadline past the calendar is not published
    lag = (published.dropna() - pd.DatetimeIndex(published.dropna().index) - pd.Timedelta(days=45)).dt.days
    assert lag.min() >= 7 and published.dropna().isin(DAYS).all()
    assert pd.isna(published[pd.Timestamp("2025-03-31")]) and published.notna().sum() == len(periods) - 1
    # no buffer is exactly the shared 13F availability day
    plain = availability_date(periods, DAYS, settle_trading_days=0)
    pd.testing.assert_series_equal(publication_dates(periods, DAYS, 0), plain, check_freq=False)
    print(
        f"SANITY: 2024-12-31 is published on 2025-02-21 (deadline Friday 02-14 + 7), the smallest lag is {lag.min()} days, buffer 0 = availability_date."
    )


def _frame(values: dict[str, float]) -> pd.DataFrame:
    """One quarter's features: column `a` holds `values` per ticker, column `b` is a constant 0.5."""
    return pd.DataFrame({"a": np.array(list(values.values()), dtype="float32"), "b": np.float32(0.5)}, index=pd.Index(list(values), name="ticker"))


def test_expand_daily_carries_each_quarter_to_the_next_publication_within_the_age_limit() -> None:
    first, second = pd.Timestamp("2024-02-21"), pd.Timestamp("2024-05-22")
    quarters = {first: _frame({"X": 1.0, "Y": 2.0}), second: _frame({"X": 3.0})}
    daily = expand_daily(quarters, DAYS, 100).set_index(["date", "ticker"])["a"]
    # nothing before the first publication, the first quarter until the day before the second, then the second only (Y is not carried over it)
    assert daily.index.get_level_values("date").min() == first
    assert daily.loc[(DAYS[DAYS < second][-1], "X")] == 1.0 and daily.loc[(DAYS[DAYS < second][-1], "Y")] == 2.0
    assert daily.loc[(second, "X")] == 3.0 and (second, "Y") not in daily.index
    # the second quarter lives 100 calendar days: its last day is on or before 2024-08-30 and the day after has no row
    last = daily.index.get_level_values("date").max()
    assert last <= second + pd.Timedelta(days=100) and DAYS[DAYS > last][0] > second + pd.Timedelta(days=100)
    # a quarter older than the limit with no successor stops: first quarter alone carries 100 days only
    alone = expand_daily({first: _frame({"X": 1.0})}, DAYS, 100)
    assert alone["date"].max() <= first + pd.Timedelta(days=100) and alone["date"].min() == first
    print(
        f"SANITY: values stay put until the next publication, Y is not carried over quarter 2, a lone quarter stops {100} days after {first.date()}."
    )


def test_expand_daily_never_uses_a_value_before_its_publication_day() -> None:
    quarters = {pd.Timestamp("2024-05-22"): _frame({"X": 9.0}), pd.Timestamp("2024-02-21"): _frame({"X": 5.0})}
    daily = expand_daily(quarters, DAYS, 400).set_index(["date", "ticker"])["a"]
    day = pd.Timestamp("2024-05-21")
    assert daily.loc[(day, "X")] == 5.0 and daily.loc[(pd.Timestamp("2024-05-22"), "X")] == 9.0
    early = daily[daily.index.get_level_values("date") < pd.Timestamp("2024-02-21")]
    assert early.empty
    print("SANITY: unordered quarters are placed by publication date: 5.0 up to 2024-05-21, 9.0 from 2024-05-22, none before the first publication.")


# ---- the step on a planted SQLite world --------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    return read_config(str(ROOT / "configs"))


def _lines(period: pd.Timestamp, number: int) -> list[dict[str, Any]]:
    """Every manager files its book 40 days after the period: X (+100 shares a quarter, all ten), Y (flat, all ten), Z (five managers)."""
    rows: list[dict[str, Any]] = []
    for index, cik in enumerate(MANAGERS):
        filed, accession = period + pd.Timedelta(days=40), f"{cik}-{period:%y%m%d}"
        marker = {
            "cusip": "_empty",
            "ticker": "_empty",
            "issuer_name": None,
            "position_type": None,
            "shares": None,
            "value_usd": None,
            "form": "13F-HR",
        }
        base = {"cik": cik, "period": period, "filing_date": filed, "accession_number": accession, "parse_status": "complete"}
        rows.append(base | marker)
        held = {"X": 1000.0 + 100.0 * number, "Y": 500.0, "Z": 300.0 if index < 5 else None}
        for ticker, shares in held.items():
            if shares is not None:
                line = {"cusip": f"{ord(ticker):06d}101", "ticker": ticker, "issuer_name": f"{ticker} CORP", "position_type": "common", "form": None}
                rows.append(base | line | {"shares": shares, "value_usd": shares * PRICES[ticker]})
    return rows


def _labels() -> pd.DataFrame:
    rows = [(cik, period, 1 if cik == MOVER and period >= PERIODS[3] else 0) for period in PERIODS for cik in MANAGERS]
    return pd.DataFrame(rows, columns=["cik", "period", "cluster"])


def _prices(last: str) -> pd.DataFrame:
    days = pd.bdate_range("2023-01-02", last)
    draws = np.random.default_rng(0).normal(0.0, 0.01, size=(len(days), len(PRICES)))
    frame = pd.DataFrame({"date": np.repeat(days, len(PRICES)), "ticker": list(PRICES) * len(days)})
    cap = frame["ticker"].map({ticker: SHARES_OUT * price for ticker, price in PRICES.items()})
    return frame.assign(close_split=frame["ticker"].map(PRICES), ret=draws.ravel(), mcap_pit=cap, shares_outstanding_pit=SHARES_OUT)


class World:
    """The seeded store, the step on it and a spy on the sec13f_hr reads and the part writes."""

    def __init__(self, store: Any, cfg: DictConfig, last_price_day: str) -> None:
        create_canonical(store.engine, [Tables.sec13f_hr, Tables.cusip_ticker_map, Tables.sharadar_tickers])
        store.save(Tables.cusip_ticker_map, pd.DataFrame({"cusip": [f"{ord(t):06d}101" for t in PRICES], "ticker": list(PRICES)}))
        tickers = pd.DataFrame({"table": "SEP", "permaticker": range(1, 4), "ticker": list(PRICES), "category": "Domestic Common Stock"})
        store.save(Tables.sharadar_tickers, tickers.assign(sector="Technology", lastpricedate=pd.Timestamp("2026-01-02")))
        store.save(Tables.sec13f_hr, pd.DataFrame([row for number, period in enumerate(PERIODS) for row in _lines(period, number)]))
        store.create_declared(Tables.sec_13f_manager_details)
        store.save(Tables.sec_13f_manager_details, _labels())
        self.store, self.cfg = store, cfg
        self.set_prices(last_price_day)
        self.reads: list[list[str]] = []
        self.writes: list[tuple[str, pd.DataFrame]] = []
        self._spy()

    def _spy(self) -> None:
        load, replace, append_tail = self.store.load, self.store.replace, self.store.append_tail

        def _load(table: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(table, "name", table) == Tables.sec13f_hr.name:
                self.reads.append(list(kwargs["where"]["period"]))
            return load(table, *args, **kwargs)

        def _replace(table: Any, df: pd.DataFrame, *args: Any, **kwargs: Any) -> int:
            self.writes.append(("replace", df))
            return replace(table, df, *args, **kwargs)

        def _append(table: Any, df: pd.DataFrame, *args: Any, **kwargs: Any) -> int:
            self.writes.append(("append_tail", df))
            return append_tail(table, df, *args, **kwargs)

        self.store.load, self.store.replace, self.store.append_tail = _load, _replace, _append

    def set_prices(self, last_price_day: str) -> None:
        self.store.replace(Tables.cube_part_prices, _prices(last_price_day))

    def step(self) -> StepCubeSecThirteen:
        context = SimpleNamespace(
            store=self.store, log=logging.getLogger(__name__), config=self.cfg, config_dir=ROOT / "configs", paths={}, universe=SimpleNamespace()
        )
        return StepCubeSecThirteen(cast(Context, context), self.cfg)

    def part(self) -> pd.DataFrame:
        part = self.store.load(Tables.cube_part_sec_thirteen)
        part["date"] = pd.to_datetime(part["date"])
        numeric = [column for column in part.columns if column not in {"date", "ticker"}]
        return part.astype({column: "float64" for column in numeric}).sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture
def world(sqlite_store: Any, cfg: DictConfig) -> Any:
    return lambda last_price_day: World(sqlite_store, cfg, last_price_day)


def _value(part: pd.DataFrame, column: str, day: str, ticker: str = "X") -> float:
    row = part[(part["date"] == pd.Timestamp(day)) & part["ticker"].eq(ticker)]
    return float("nan") if row.empty else float(row[column].iloc[0])


def test_full_build_carries_the_80_columns_by_publication_with_membership_from_the_previous_label(world: Any, cfg: DictConfig) -> None:
    w = world("2025-03-31")
    w.step().run(full=True)
    part = w.part()
    # the 80 configured float32 columns, written by one replace
    columns = column_names(cluster_specs(cfg.cluster))
    assert len(columns) == 80 and set(part.columns) == {"date", "ticker", *columns}
    kind, frame = w.writes[0]
    assert kind == "replace" and len(w.writes) == 1 and all(frame[column].dtype == np.float32 for column in columns)
    # first publication 2023-08-21 (quarter 2023-06-30): nothing earlier; X flows 10 x 100 / 100,000 until the mover leaves the members
    assert part["date"].min() == _pub("2023-06-30") and part["date"].max() == pd.Timestamp("2025-03-31")
    assert _value(part, FLOW, "2023-08-18") != _value(part, FLOW, "2023-08-18") and _value(part, FLOW, "2023-08-21") == pytest.approx(0.01)
    assert _value(part, FLOW, "2024-05-21") == pytest.approx(0.01), "the quarter 2024-03-31 is not published before 2024-05-22"
    assert _value(part, FLOW, "2024-05-22") == pytest.approx(0.009), "labelled cluster 1 on 2023-12-31, so out of the members of 2024-03-31"
    assert _value(part, FLOW, "2025-03-31") == pytest.approx(0.009)
    # stickiness needs the book four quarters back: null until 2024-03-31 (2023-03-31 lines), then every holder is there
    assert np.isnan(_value(part, STICKY, "2023-08-21")) and _value(part, STICKY, "2024-05-22") == pytest.approx(1.0)
    # Z is held by 5 managers (< 8 min_holders): no row; clusters without members are null
    assert part[part["ticker"].eq("Z")][FLOW].isna().all() and part["f_ic_cluster_3_net_breadth"].isna().all()
    print(
        f"SANITY: {len(part)} rows, 80 float32 columns; X flow 0.01 until 2024-05-21 then 0.009 once the 2023-12-31 label moves the mover out; Z has no row."
    )


def test_night_without_a_due_quarter_carries_the_saved_values_and_reads_no_lines(world: Any) -> None:
    w = world("2025-02-18")  # 2024-12-31 is published on 2025-02-21
    w.step().run(full=True)
    before = w.part()
    assert before["date"].max() == pd.Timestamp("2025-02-18")

    # two new sessions, no quarter published: values carried, nothing read from sec13f_hr, only the tail rewritten
    w.set_prices("2025-02-20")
    w.reads.clear()
    w.writes.clear()
    w.step().run()
    after = w.part()
    assert w.reads == [] and [kind for kind, _ in w.writes] == ["append_tail"]
    assert after["date"].max() == pd.Timestamp("2025-02-20") and len(after) > len(before)
    assert _value(after, FLOW, "2025-02-20") == _value(after, FLOW, "2025-02-18") == pytest.approx(0.009)

    # the rows before the rewritten tail are the saved ones, the same night again changes nothing, and the full rebuild gives the same part
    cutoff = pd.Timestamp("2025-02-06")
    pd.testing.assert_frame_equal(before[before["date"] < cutoff].reset_index(drop=True), after[after["date"] < cutoff].reset_index(drop=True))
    w.step().run()
    pd.testing.assert_frame_equal(after, w.part())
    w.step().run(full=True)
    pd.testing.assert_frame_equal(after, w.part())
    print("SANITY: a night between publications reads 0 line chunks, carries 0.009 onto 2025-02-20, is idempotent and equals the full rebuild.")


def test_night_with_a_due_quarter_computes_it_with_the_consistency_quarter_and_equals_a_full_rebuild(world: Any) -> None:
    w = world("2025-02-18")
    w.step().run(full=True)
    before = w.part()
    w.set_prices("2025-03-31")
    w.reads.clear()
    w.writes.clear()
    w.step().run()
    after = w.part()
    # only the newest saved quarter (2024-09-30, recomputed to check) and the due one (2024-12-31) are read, with their previous and year-ago books
    periods = {period for read in w.reads for period in read}
    expected = {"2023-09-30", "2023-12-31", "2024-06-30", "2024-09-30", "2024-12-31"}
    assert periods == expected and w.writes[0][0] == "append_tail"
    assert _value(after, FLOW, "2025-02-20") == pytest.approx(0.009) and _value(after, STICKY, "2025-02-21") == pytest.approx(1.0)
    assert after["date"].max() == pd.Timestamp("2025-03-31")
    # the rows before the tail start are the saved ones
    start = pd.Timestamp("2025-02-06")
    pd.testing.assert_frame_equal(before[before["date"] < start].reset_index(drop=True), after[after["date"] < start].reset_index(drop=True))
    w.step().run(full=True)
    pd.testing.assert_frame_equal(after, w.part())
    print(f"SANITY: the due quarter 2024-12-31 read {sorted(periods)} only, appended 2025-02-19..2025-03-31 and equals the full rebuild exactly.")


def test_column_set_mismatch_stops_before_any_work_and_full_rebuilds(world: Any, cfg: DictConfig) -> None:
    w = world("2025-02-18")
    w.step().run(full=True)
    saved = w.store.load(Tables.cube_part_sec_thirteen)
    stale = saved.assign(f_stale_column=1.0)
    w.store.replace(Tables.cube_part_sec_thirteen, stale)
    w.reads.clear()
    w.writes.clear()
    with pytest.raises(RuntimeError, match="build-sec-thirteen -F"):
        w.step().run()
    assert w.reads == [] and w.writes == []
    assert "f_stale_column" in w.store.columns(Tables.cube_part_sec_thirteen)
    w.step().run(full=True)
    assert set(w.store.columns(Tables.cube_part_sec_thirteen)) == {"date", "ticker", *column_names(cluster_specs(cfg.cluster))}
    print("SANITY: a stored column set that differs from the 80 configured stops the incremental run before any read or write; -F restores the 80.")


def test_a_relabelled_newest_saved_quarter_stops_the_due_quarter_with_the_rebuild_message(world: Any) -> None:
    w = world("2025-02-18")
    w.step().run(full=True)
    before = w.part()
    # a refit-like relabel: the label of 2024-06-30 (the members of the newest saved quarter 2024-09-30) changes
    w.store.save(Tables.sec_13f_manager_details, pd.DataFrame({"cik": MANAGERS[:3], "period": PERIODS[5], "cluster": 2}))
    w.set_prices("2025-03-31")
    w.writes.clear()
    with pytest.raises(RuntimeError, match="build-sec-thirteen -F"):
        w.step().run()
    assert w.writes == []
    pd.testing.assert_frame_equal(before, w.part())
    print(
        "SANITY: the recomputed newest saved quarter no longer matches its saved values, so the due quarter stops with the -F message and no write."
    )


def test_a_quarter_without_labels_for_the_quarter_before_stops_the_build(world: Any) -> None:
    w = world("2025-03-31")
    w.store.delete(Tables.sec_13f_manager_details, where={"period": PERIODS[5]})
    w.reads.clear()
    with pytest.raises(RuntimeError, match="build-cluster"):
        w.step().run(full=True)
    assert w.reads == [] and not w.store.exists(Tables.cube_part_sec_thirteen) and w.writes == []
    print("SANITY: unlabelled 2024-06-30 stops the quarter 2024-09-30 before any line is read or any row written.")
