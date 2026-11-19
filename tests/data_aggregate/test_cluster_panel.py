"""Cluster panel: the publication clock, the daily expansion with a carry limit and the incremental `StepCubeSecThirteen` (AC-008, AC-010, AC-011).

The clock and the expansion are exact on hand-built frames. The step runs on an in-memory SQLite world of ten managers holding three tickers over
eight report quarters (known holdings: X flows +100 shares per manager and quarter), labelled cluster 0 except one manager who moves to cluster 1
after 2023-12-31, so the full build, both incremental nights, the column-set stop, the consistency stop and the missing-label stop are checked on
known values. No DAG, cube build, PostgreSQL or configs/ write is made.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.transformers import step_cube_sec_thirteen as step_module
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.price.frames import load_trading_calendar
from src.data_aggregate.utils.sec_thirteen.cluster_config import cluster_specs, column_names
from src.data_aggregate.utils.sec_thirteen.cluster_panel import expand_daily, filing_deadlines, publication_dates
from src.data_store.schema import Tables
from src.utils.config import read_config
from src.utils.window import WindowRun
from tests.canonical_schema import create_canonical
from tests.support.config import fixture_config
from tests.support.universe import universe_of

ROOT = Path(__file__).resolve().parents[2]
PERIODS = [pd.Timestamp(p) for p in ("2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31")]
MANAGERS = [f"{n:010d}" for n in range(1, 11)]
MOVER = MANAGERS[-1]  # cluster 0 through 2023-09-30, cluster 1 from 2023-12-31 (so out of the members of the quarters from 2024-03-31)
PRICES = {"X": 50.0, "Y": 20.0, "Z": 10.0}
SHARES_OUT = 100_000.0
FLOW = "f_ic_cluster_0_net_share_flow"
STICKY = "f_ic_cluster_0_stickiness"
DAYS = pd.bdate_range("2022-01-03", "2025-03-31")


LAG = int(fixture_config().data.tables.sec13f_hr.pit.lag_days)


def _pub(period: str) -> pd.Timestamp:
    return cast(pd.Timestamp, publication_dates(pd.DatetimeIndex([period]), DAYS, fixture_config()).iloc[0])


def test_publication_date_is_the_deadline_day_plus_the_configured_pit_lag_on_a_trading_day() -> None:
    periods = pd.DatetimeIndex(pd.date_range("2022-03-31", "2025-03-31", freq="QE"))
    published = publication_dates(periods, DAYS, fixture_config())
    # the deadline (day 45, a weekend rolls to Monday) plus the seven calendar days of data.tables.sec13f_hr.pit, then the first session
    expected = {"2022-03-31": "2022-05-23", "2022-06-30": "2022-08-22", "2024-09-30": "2024-11-21", "2024-12-31": "2025-02-21"}
    assert LAG == 7
    for period, day in expected.items():
        assert published[pd.Timestamp(period)] == pd.Timestamp(day), period
    # every published quarter is at least the lag after its 45-day deadline and on a trading day; a deadline past the calendar is not published
    lag = (published.dropna() - pd.DatetimeIndex(published.dropna().index) - pd.Timedelta(days=45)).dt.days
    assert lag.min() >= LAG and published.dropna().isin(DAYS).all()
    assert pd.isna(published[pd.Timestamp("2025-03-31")]) and published.notna().sum() == len(periods) - 1
    # a zero lag is exactly the rolled filing deadline
    no_lag = fixture_config(data={"tables": {"sec13f_hr": {"pit": {"lag_days": 0}}}})
    pd.testing.assert_series_equal(publication_dates(periods, DAYS, no_lag), filing_deadlines(periods, DAYS), check_freq=False)
    print(f"SANITY: 2024-12-31 is published on Friday 2025-02-21 (deadline Friday 02-14 + {LAG}), the smallest lag is {lag.min()} days.")


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


def test_filing_deadline_is_the_trading_day_on_or_after_day_45_and_the_publication_is_the_pit_lag_after_it() -> None:
    periods = pd.DatetimeIndex(pd.date_range("2022-03-31", "2025-03-31", freq="QE"))
    deadlines = filing_deadlines(periods, DAYS)
    published = publication_dates(periods, DAYS, fixture_config())
    # a weekend day 45 rolls to the next trading day (2022-06-30 + 45 = Sunday 08-14), a weekday stays
    assert deadlines[pd.Timestamp("2022-06-30")] == pd.Timestamp("2022-08-15") and deadlines[pd.Timestamp("2022-09-30")] == pd.Timestamp("2022-11-14")
    assert deadlines[pd.Timestamp("2024-12-31")] == pd.Timestamp("2025-02-14")
    day_45 = pd.Series(periods + pd.Timedelta(days=45), index=periods)
    rolled = deadlines.dropna() > day_45.reindex(deadlines.dropna().index)
    # the only clock: the deadline is a trading day, never before day 45, and the publication day is at least the lag after it
    assert deadlines.dropna().isin(DAYS).all() and (deadlines.dropna() >= day_45.reindex(deadlines.dropna().index)).all() and rolled.any()
    inside = published.dropna().index
    assert ((published[inside] - pd.Timedelta(days=LAG)) >= deadlines[inside]).all()
    print(
        f"SANITY: {int(rolled.sum())} of {len(periods)} quarters roll a weekend day 45; the publication day is always at least {LAG} days after the deadline."
    )


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
    return pd.DataFrame(rows, columns=["cik", "period", "cluster"]).assign(first_filed_date=lambda labels: labels["period"] + pd.Timedelta(days=40))


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
        self.read_ciks: list[str] = []
        self.writes: list[tuple[str, pd.DataFrame, bool]] = []
        self._spy()

    def _spy(self) -> None:
        load, replace_window = self.store.load, self.store.replace_window

        def _load(table: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(table, "name", table) == Tables.sec13f_hr.name:
                self.reads.append(list(kwargs["where"]["period"]))
                self.read_ciks.extend(kwargs["where"]["cik"])
            return load(table, *args, **kwargs)

        def _replace_window(table: Any, df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
            self.writes.append(("replace_window", df, bool(kwargs["schema_rebuild"])))
            return replace_window(table, df, *args, **kwargs)

        self.store.load, self.store.replace_window = _load, _replace_window

    def set_prices(self, last_price_day: str) -> None:
        self.as_of = pd.Timestamp(last_price_day)
        self.store.replace(Tables.cube_part_prices, _prices(last_price_day))

    def step(self, full: bool = False) -> StepCubeSecThirteen:
        run = WindowRun("full" if full else "incremental", "known-truth cluster fixture", self.as_of)
        context = SimpleNamespace(
            store=self.store,
            log=logging.getLogger(__name__),
            config=self.cfg,
            config_dir=ROOT / "configs",
            paths={},
            universe=universe_of(list(PRICES)),
            run=run,
        )
        return StepCubeSecThirteen(cast(Context, context), self.cfg)

    def run(self, full: bool = False) -> Any:
        """One run of the step under the root-run contract of an incremental or full run."""
        return self.step(full).run()

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
    receipt = w.run(full=True)
    part = w.part()
    # the 80 configured float32 columns, written by one full-schema window replacement, and the receipt of the resolved window
    columns = column_names(cluster_specs(cfg.cluster))
    assert len(columns) == 80 and set(part.columns) == {"date", "ticker", *columns}
    kind, frame, rebuilt = w.writes[0]
    assert kind == "replace_window" and rebuilt and len(w.writes) == 1 and all(frame[column].dtype == np.float32 for column in columns)
    assert list(receipt) == [Tables.cube_part_sec_thirteen] and receipt[Tables.cube_part_sec_thirteen].is_full
    # first publication (quarter 2023-06-30): nothing earlier; X flows 10 x 100 / 100,000 until the mover leaves the members
    first, newer = _pub("2023-06-30"), _pub("2024-03-31")
    assert first == pd.Timestamp("2023-08-21") and part["date"].min() == first and part["date"].max() == pd.Timestamp("2025-03-31")
    assert np.isnan(_value(part, FLOW, str((DAYS[DAYS < first][-1]).date()))) and _value(part, FLOW, str(first.date())) == pytest.approx(0.01)
    assert _value(part, FLOW, str((DAYS[DAYS < newer][-1]).date())) == pytest.approx(0.01), "the quarter 2024-03-31 is not published before its day"
    assert _value(part, FLOW, str(newer.date())) == pytest.approx(0.009), "labelled cluster 1 on 2023-12-31, so out of the members of 2024-03-31"
    assert _value(part, FLOW, "2025-03-31") == pytest.approx(0.009)
    # stickiness needs the book four quarters back: null until 2024-03-31 (2023-03-31 lines), then every holder is there
    assert np.isnan(_value(part, STICKY, str(first.date()))) and _value(part, STICKY, str(newer.date())) == pytest.approx(1.0)
    # Z is held by 5 managers (< 8 min_holders): no row; clusters without members are null
    assert part[part["ticker"].eq("Z")][FLOW].isna().all() and part["f_ic_cluster_3_net_breadth"].isna().all()
    print(
        f"SANITY: {len(part)} rows, 80 float32 columns; X flow 0.01 until {newer.date()} then 0.009 once the 2023-12-31 label moves the mover out; Z has no row."
    )


def test_managers_of_tracked_only_clusters_and_the_sentinels_are_neither_read_nor_members(world: Any) -> None:
    w = world("2025-03-31")
    w.run(full=True)
    first = w.part()
    extra = [("9000000011", 11), ("9000000012", 12), ("9000000013", 13), ("9000000015", 15), ("9000000090", -1), ("9000000091", -2)]
    rows = [(cik, period, cluster, period + pd.Timedelta(days=40)) for period in PERIODS for cik, cluster in extra]
    w.store.save(Tables.sec_13f_manager_details, pd.DataFrame(rows, columns=["cik", "period", "cluster", "first_filed_date"]))
    w.reads.clear()
    w.read_ciks.clear()

    w.run(full=True)

    assert w.read_ciks and not set(w.read_ciks) & {cik for cik, _ in extra}, "no line of a tracked-only or sentinel manager is read"
    pd.testing.assert_frame_equal(w.part(), first)
    print(
        f"SANITY: {len(extra)} managers labelled 11, 12, 13, 15, -1 and -2 in every quarter: {len(set(w.read_ciks))} CIKs read (the clusters with features only) and the 80 columns equal the build without them."
    )


def test_night_without_a_due_quarter_carries_the_saved_values_and_reads_no_lines(world: Any) -> None:
    w = world("2025-02-12")  # 2024-12-31 is published on Friday 2025-02-21
    w.run(full=True)
    before = w.part()
    assert before["date"].max() == pd.Timestamp("2025-02-12")

    # two new sessions, no quarter published: values carried, nothing read from sec13f_hr, only the configured overlap rewritten
    w.set_prices("2025-02-14")
    w.reads.clear()
    w.writes.clear()
    plan = w.run()[Tables.cube_part_sec_thirteen]
    overlap = int(w.cfg.data.tables.cube_part_sec_thirteen.overlap.days)
    assert plan.frontier == pd.Timestamp("2025-02-12") and plan.write.start == plan.frontier - pd.Timedelta(days=overlap) and not plan.is_full
    assert w.reads == [] and [(kind, rebuilt) for kind, _, rebuilt in w.writes] == [("replace_window", False)]
    after = w.part()
    assert after["date"].max() == pd.Timestamp("2025-02-14") and len(after) > len(before)
    assert _value(after, FLOW, "2025-02-14") == _value(after, FLOW, "2025-02-12") == pytest.approx(0.009)

    # the rewritten window re-emits the saved rows exactly, the same night again changes nothing, and the full rebuild gives the same part
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after[after["date"] <= plan.frontier].reset_index(drop=True))
    w.run()
    pd.testing.assert_frame_equal(after, w.part())
    w.run(full=True)
    pd.testing.assert_frame_equal(after, w.part())
    print(
        f"SANITY: a night between publications reads 0 line chunks, rewrites {overlap} days from {plan.write.start.date()} as saved, is idempotent and equals the full rebuild."
    )


def test_night_with_a_due_quarter_computes_it_with_the_consistency_quarter_and_equals_a_full_rebuild(world: Any) -> None:
    w = world("2025-02-14")
    w.run(full=True)
    before = w.part()
    w.set_prices("2025-03-31")
    w.reads.clear()
    w.writes.clear()
    plan = w.run()[Tables.cube_part_sec_thirteen]
    after = w.part()
    # only the newest saved quarter (2024-09-30, recomputed to check) and the due one (2024-12-31) are read, with their previous and year-ago books
    periods = {period for read in w.reads for period in read}
    expected = {"2023-09-30", "2023-12-31", "2024-06-30", "2024-09-30", "2024-12-31"}
    assert periods == expected and [(kind, rebuilt) for kind, _, rebuilt in w.writes] == [("replace_window", False)]
    assert plan.frontier == pd.Timestamp("2025-02-14") and plan.write.end == pd.Timestamp("2025-03-31")
    assert _value(after, FLOW, "2025-02-14") == pytest.approx(0.009) and _value(after, STICKY, "2025-02-21") == pytest.approx(1.0)
    assert after["date"].max() == pd.Timestamp("2025-03-31")
    # the rows up to the frontier, rewritten inside the overlap or not, are the saved ones
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after[after["date"] <= plan.frontier].reset_index(drop=True))
    w.run(full=True)
    pd.testing.assert_frame_equal(after, w.part())
    print(
        f"SANITY: the due quarter 2024-12-31 read {sorted(periods)} only, rewrote {plan.write.start.date()}..2025-03-31 and equals the full rebuild exactly."
    )


def test_column_set_mismatch_stops_before_any_work_and_full_rebuilds(world: Any, cfg: DictConfig) -> None:
    w = world("2025-02-14")
    w.run(full=True)
    saved = w.store.load(Tables.cube_part_sec_thirteen)
    stale = saved.assign(f_stale_column=1.0)
    w.store.replace(Tables.cube_part_sec_thirteen, stale)
    w.reads.clear()
    w.writes.clear()
    with pytest.raises(RuntimeError, match="build-sec-thirteen -F"):
        w.run()
    assert w.reads == [] and w.writes == []
    assert "f_stale_column" in w.store.columns(Tables.cube_part_sec_thirteen)
    w.run(full=True)
    assert set(w.store.columns(Tables.cube_part_sec_thirteen)) == {"date", "ticker", *column_names(cluster_specs(cfg.cluster))}
    print("SANITY: a stored column set that differs from the 80 configured stops the incremental run before any read or write; -F restores the 80.")


def test_a_relabelled_newest_saved_quarter_stops_the_due_quarter_with_the_rebuild_message(world: Any) -> None:
    w = world("2025-02-14")
    w.run(full=True)
    before = w.part()
    # a refit-like relabel: the label of 2024-06-30 (the members of the newest saved quarter 2024-09-30) changes
    relabelled = pd.DataFrame({"cik": MANAGERS[:3], "period": PERIODS[5], "cluster": 2}).assign(first_filed_date=PERIODS[5] + pd.Timedelta(days=40))
    w.store.save(Tables.sec_13f_manager_details, relabelled)
    w.set_prices("2025-03-31")
    w.writes.clear()
    with pytest.raises(RuntimeError, match="build-sec-thirteen -F"):
        w.run()
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
        w.run(full=True)
    assert w.reads == [] and not w.store.exists(Tables.cube_part_sec_thirteen) and w.writes == []
    print("SANITY: unlabelled 2024-06-30 stops the quarter 2024-09-30 before any line is read or any row written.")


def test_report_periods_before_the_declared_source_start_do_not_open_the_plan(world: Any) -> None:
    w = world("2025-03-31")
    w.run(full=True)
    clean = w.part()
    # a typo-dated or very late filed old period (1987, 1997) whose first filing falls inside the source window must not pull the plan back
    strays = pd.DataFrame(
        [
            (MANAGERS[0], pd.Timestamp("1987-03-31"), -1, pd.Timestamp("2003-05-01")),
            (MANAGERS[0], pd.Timestamp("1997-06-30"), 0, pd.Timestamp("2003-05-01")),
        ],
        columns=["cik", "period", "cluster", "first_filed_date"],
    )
    w.store.save(Tables.sec_13f_manager_details, strays)
    w.writes.clear()
    w.run(full=True)
    pd.testing.assert_frame_equal(w.part(), clean)
    print("SANITY: labels of periods 1987-03-31 and 1997-06-30 before the 1999-07-01 source start leave the quarters planned and the part unchanged.")


LATE_MANAGER = "0000000011"  # labelled cluster 0 and files every book 60 days after its period end, after the deadline of any quarter


def _add_a_late_manager(store: Any) -> None:
    rows = [
        row
        | {"cik": LATE_MANAGER, "accession_number": f"{LATE_MANAGER}-{row['period']:%y%m%d}", "filing_date": row["period"] + pd.Timedelta(days=60)}
        for number, period in enumerate(PERIODS)
        for row in _lines(period, number)
        if row["cik"] == MANAGERS[0]
    ]
    store.save(Tables.sec13f_hr, pd.DataFrame(rows))
    late = pd.DataFrame(
        [(LATE_MANAGER, period, 0, period + pd.Timedelta(days=60)) for period in PERIODS], columns=["cik", "period", "cluster", "first_filed_date"]
    )
    store.save(Tables.sec_13f_manager_details, late)


def test_the_step_cuts_every_book_at_the_deadline_the_publication_clock_uses(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    read_newest: dict[pd.Timestamp, pd.Timestamp] = {}
    cleaned_newest: dict[pd.Timestamp, pd.Timestamp] = {}
    real_read, real_clean = step_module.read_lines, step_module.clean_book

    def read(store: Any, ciks: Any, periods: Any, filed: Any) -> pd.DataFrame:
        raw = real_read(store, ciks, periods, filed)
        read_newest[periods[0]] = pd.to_datetime(raw["filing_date"]).max()
        return raw

    def clean(raw: pd.DataFrame, config: Any, prefixes: Any) -> pd.DataFrame:
        if len(raw):
            cleaned_newest[pd.Timestamp(raw["period"].iloc[0])] = pd.to_datetime(raw["filing_date"]).max()
        return real_clean(raw, config, prefixes)

    monkeypatch.setattr(step_module, "read_lines", read)
    monkeypatch.setattr(step_module, "clean_book", clean)
    w = world("2025-03-31")
    _add_a_late_manager(w.store)
    w.run(full=True)

    periods = pd.DatetimeIndex(sorted(cleaned_newest))
    calendar = load_trading_calendar(w.step()._context)
    clock = filing_deadlines(periods, calendar)
    published = publication_dates(periods, calendar, w.cfg)
    assert len(periods) >= 6 and all(read_newest[t] == t + pd.Timedelta(days=60) for t in periods), "the late manager's lines are read"
    assert all(cleaned_newest[t] == t + pd.Timedelta(days=40) and cleaned_newest[t] <= clock[t] for t in periods), "no late line reaches the cleaning"
    assert ((published.dropna() - pd.Timedelta(days=LAG)) >= clock[published.dropna().index]).all()
    print(
        f"SANITY: the {len(periods)} quarters read lines filed 60 days after their period end and hand the cleaning only those filed by the "
        f"deadline the publication clock uses (newest 40 days), each deadline at least {LAG} days before its publication."
    )


def test_the_step_validates_the_cluster_block_before_any_read(world: Any, cfg: DictConfig) -> None:
    w = world("2025-03-31")
    bad = copy.deepcopy(cfg)
    bad.cluster.clusters.bank.id = bad.cluster.clusters.insurance.id
    w.cfg = bad
    with pytest.raises(ValueError, match="distinct"):
        w.run(full=True)

    assert w.reads == [] and w.writes == []
    print("SANITY: two categories sharing an id stop the step in the plan, before any line is read or any row written.")
