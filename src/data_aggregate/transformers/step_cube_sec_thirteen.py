"""Per-cluster 13F cube features: plan the due report quarters, compute each from the stored cluster labels, carry them daily and save one window."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PART_REFRESH_TRADING_DAYS, PartWindow, plan_window, write_part
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.manager_profile.lines import clean_book, read_lines
from src.data_aggregate.utils.manager_profile.reference import classify_securities, prefix_tickers, sharadar_reference
from src.data_aggregate.utils.price.frames import load_price_frames, load_trading_calendar
from src.data_aggregate.utils.sec_thirteen.cluster_config import COLUMN_PREFIX, ClusterSpec, cluster_specs, column_budget, column_names
from src.data_aggregate.utils.sec_thirteen.cluster_features import MARKET_COLUMNS, QuarterBooks, cluster_feature_frame
from src.data_aggregate.utils.sec_thirteen.cluster_lines import company_books
from src.data_aggregate.utils.sec_thirteen.cluster_panel import expand_daily, publication_dates
from src.data_store.schema import Tables
from src.utils.identity_tables import CusipTags
from src.utils.step import Step

_REBUILD = "run `data_aggregate build-sec-thirteen -F`"


@dataclass(frozen=True)
class Quarters:
    """The report quarters of one run: the publication day of each, those published after the saved part (`due`), those to compute and the newest saved one."""

    publication: pd.Series
    due: list[pd.Timestamp]
    compute: list[pd.Timestamp]
    saved_quarter: pd.Timestamp | None


@dataclass(frozen=True)
class Saved:
    """The stored part: its column set and window, the refresh tail and the saved values of the newest day, carried onto the days after it."""

    columns: list[str]
    window: PartWindow
    tail: pd.DataFrame | None
    carried: pd.DataFrame | None


@dataclass(frozen=True)
class Plan:
    """What one run does: the configured clusters and columns, the trading calendar, the manager labels, the quarters and the saved part."""

    specs: list[ClusterSpec]
    columns: list[str]
    calendar: pd.DatetimeIndex
    labels: pd.DataFrame
    quarters: Quarters
    saved: Saved


@dataclass(frozen=True)
class References:
    """The identifier references of a line: CUSIP tags, the Sharadar reference and the CUSIP-prefix tickers."""

    tags: CusipTags
    reference: pd.DataFrame
    prefixes: pd.Series


@dataclass(frozen=True)
class Market:
    """The quarter-end market frames (index date, columns ticker): market cap, shares outstanding and 250-day volatility."""

    mcap: pd.DataFrame
    shares: pd.DataFrame
    volatility: pd.DataFrame


@dataclass(frozen=True)
class Resources:
    """What the quarters read once: members by (label period, cluster), the labelled managers by period, the universe, references and market."""

    members: dict[tuple[pd.Timestamp, int], list[str]]
    labelled: dict[pd.Timestamp, frozenset[str]]
    universe: tuple[str, ...]
    references: References
    market: Market


def _market_at(market: Market, period: pd.Timestamp) -> pd.DataFrame:
    """`MARKET_COLUMNS` per ticker at the quarter end `period` (market cap also a quarter back), the last observation on or before the day."""
    back = period - pd.offsets.QuarterEnd(1)
    frames_days = ((market.mcap, period), (market.mcap, back), (market.shares, period), (market.volatility, period))
    values = [frame.reindex([day], method="ffill").iloc[0] for frame, day in frames_days]
    return pd.DataFrame(dict(zip(MARKET_COLUMNS, values, strict=True)))


class StepCubeSecThirteen(Step):
    _FIELDS = ("mcap_pit", "shares_outstanding_pit", "ret")

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._part = part_for(Tables.cube_part_sec_thirteen)

    def run(self, full: bool = False) -> None:
        """Plan, read the due quarters' books, compute the per-cluster features, carry them over the trading days and save the window."""
        started = time.monotonic()

        # load and plan: the window, the labels and the saved tail, then the references and market of the quarters to compute
        plan = self._plan(full)
        resources = self._load(plan) if plan.quarters.compute else None

        # compute, check, merge and save
        computed = {} if resources is None else self._compute_quarters(plan, resources)
        self._check(plan, computed)
        rows = self._expand(plan, computed)
        self._save(plan, rows)
        self._log.info(
            "sec thirteen clusters: %d quarters computed %s, %d rows written (%.1f s)",
            len(plan.quarters.due),
            [t.date().isoformat() for t in plan.quarters.due],
            len(rows),
            time.monotonic() - started,
        )

    def _plan(self, full: bool) -> Plan:
        """The window, the quarters to compute and the saved rows to carry; raises before any quarter is read when the part cannot be extended."""
        # config: the clusters and their cube columns; the column budget warns above its limits
        cfg = self._config.cluster
        specs = cluster_specs(cfg)
        columns = column_names(specs)
        self._log.info("sec thirteen clusters: %d cube columns", column_budget(specs, cfg))

        # load: the calendar, the saved column set and window, the manager labels and the saved tail
        calendar = load_trading_calendar(self._store)
        saved_columns = self._store.columns(self._part.table)
        window = plan_window(self._store, self._part.table, full=full, warmup=0, trading_index=calendar, refresh=PART_REFRESH_TRADING_DAYS)
        df_labels = self._store.load(Tables.sec_13f_manager_details, columns=["cik", "period", "cluster"], optional=True)
        tail = None if window.is_full else self._store.load(self._part.table, since=window.refresh_from, optional=True)

        # check: a stored column set other than the configured one needs the full rebuild, never a hidden one; the labels exist
        if not window.is_full and saved_columns and set(saved_columns) != {"date", "ticker", *columns}:
            raise RuntimeError(f"{self._part.table} has {len(saved_columns) - 2} feature columns, the config {len(columns)}: {_REBUILD}")
        if df_labels is None or df_labels["cluster"].isna().all():
            raise RuntimeError("no cluster labels in sec_13f_manager_details: run `data_aggregate build-cluster` first")
        labels = df_labels.assign(period=pd.to_datetime(df_labels["period"]).dt.normalize())

        # plan: the publication day of every quarter with a previous label, the saved quarter and the quarters to compute
        one, age = pd.offsets.QuarterEnd(1), pd.Timedelta(days=int(cfg.max_age_calendar_days))
        periods = pd.DatetimeIndex(pd.date_range(labels["period"].min() + one, calendar[-1], freq="QE"))
        published = publication_dates(periods, calendar, int(cfg.availability_buffer_days)).dropna()
        published = published[published + age >= pd.Timestamp(cfg.history_start)]
        saved = None if window.last is None else published[published <= window.last]
        saved_quarter = None if saved is None or saved.empty else cast(pd.Timestamp, saved.index[-1])
        due = list(published.index if window.last is None else published[published > window.last].index)
        compute = [saved_quarter, *due] if due and saved_quarter is not None else due
        if window.is_full and not due:
            raise RuntimeError("no report quarter is published yet in the price calendar; nothing to build")

        # check: every quarter needs the labels of the quarter before it
        labelled = set(labels.loc[labels["cluster"].notna(), "period"])
        missing = [(t - one).date().isoformat() for t in compute if t - one not in labelled]
        if missing:
            raise RuntimeError(f"cluster labels missing for the report quarters {missing}: run `data_aggregate build-cluster` first")

        # merge: the saved values of the newest date, carried onto the days after it
        carried = None
        if tail is not None:
            latest = tail[pd.to_datetime(tail["date"]) == window.last]
            carried = latest.set_index("ticker")[columns].astype("float32")
        return Plan(specs, columns, calendar, labels, Quarters(published, due, compute, saved_quarter), Saved(saved_columns, window, tail, carried))

    def _load(self, plan: Plan) -> Resources:
        """The references, the labels as member lists and the quarter-end market frames of the quarters to compute, each read once."""
        # load: the reference tables and the price fields over the span the quarters need
        # (volatility: std of the last 250 daily returns, at least 120 observed; prices start 400 calendar days before the oldest quarter)
        vol_window, min_periods, lookback_days = 250, 120, 400
        columns = ["permaticker", "ticker", "category", "sector", "lastpricedate"]
        df_tickers = cast(pd.DataFrame, self._store.load(Tables.sharadar_tickers, columns=columns))
        tags = CusipTags.load(self._store)
        oldest = min(plan.quarters.compute) - pd.offsets.QuarterEnd(1) - pd.Timedelta(days=lookback_days)
        frames = load_price_frames(self._store, peers={}, fields=self._FIELDS, since=None if plan.saved.window.is_full else oldest)
        frames.require(*self._FIELDS)

        # clean: the labelled managers by label period and by cluster
        labelled = plan.labels[plan.labels["cluster"].ge(0)]
        members = {
            (cast(pd.Timestamp, period), int(cast(float, cluster))): group["cik"].tolist()
            for (period, cluster), group in labelled.groupby(["period", "cluster"])
        }
        by_period = {cast(pd.Timestamp, period): frozenset(group["cik"]) for period, group in labelled.groupby("period")}

        # compute: the volatility frame
        returns = cast(pd.DataFrame, frames.ret)
        volatility = returns.rolling(vol_window, min_periods=min_periods).std()
        reference, prefixes = sharadar_reference(df_tickers), prefix_tickers(tags.cmap)
        mcap, shares = cast(pd.DataFrame, frames.mcap_pit), cast(pd.DataFrame, frames.shares_outstanding_pit)
        return Resources(members, by_period, frames.universe, References(tags, reference, prefixes), Market(mcap, shares, volatility))

    def _compute_quarters(self, plan: Plan, res: Resources) -> dict[pd.Timestamp, pd.DataFrame]:
        """The feature frame (index ticker, the configured float32 columns) of each quarter to compute, oldest first; each book is built once."""
        # clean: the managers each report period must provide, over every quarter that reads it (itself, the next one and four ahead)
        one, four = pd.offsets.QuarterEnd(1), pd.offsets.QuarterEnd(4)
        needs: dict[pd.Timestamp, frozenset[str]] = {}
        for t in plan.quarters.compute:
            for period in (t, t - one, t - four):
                needs[period] = needs.get(period, frozenset()) | res.labelled.get(t - one, frozenset())

        # compute: the books of the quarter, the quarter before and the year before, then every cluster's features
        profile_cfg, universe = self._config.build_cube.manager_profile, set(res.universe)
        books: dict[pd.Timestamp, pd.DataFrame] = {}
        computed: dict[pd.Timestamp, pd.DataFrame] = {}
        for t in plan.quarters.compute:
            started = time.monotonic()
            for period in sorted({t - four, t - one, t} - books.keys()):
                raw = read_lines(self._store, sorted(needs[period]), [period])
                classified = classify_securities(clean_book(raw, profile_cfg, res.references.prefixes), res.references.reference, profile_cfg)
                books[period] = company_books(classified, res.references.tags, universe)
            quarter, market = QuarterBooks(books[t], books[t - one], books[t - four]), _market_at(res.market, t)
            frames = [
                cluster_feature_frame(quarter, res.members.get((t - one, spec.cluster_id), []), market, spec).add_prefix(
                    f"{COLUMN_PREFIX}_{spec.cluster_id}_"
                )
                for spec in plan.specs
            ]
            computed[t] = pd.concat(frames, axis=1).reindex(columns=plan.columns).astype("float32")
            books = {period: book for period, book in books.items() if period > t - four}
            self._log.info("sec thirteen clusters: %s %d tickers (%.1f s)", t.date(), len(computed[t]), time.monotonic() - started)
        return computed

    def _check(self, plan: Plan, computed: dict[pd.Timestamp, pd.DataFrame]) -> None:
        """Stop unless the newest saved quarter, recomputed, equals its saved values within the tolerance."""
        saved = plan.quarters.saved_quarter
        if saved is None or saved not in computed or plan.saved.carried is None:
            return

        # compare: the recomputed quarter against the saved rows of the newest saved day, a missing row reads as null
        fresh = computed[saved]
        index = fresh.index.union(plan.saved.carried.index)
        new, old = (frame.reindex(index=index, columns=plan.columns).to_numpy(dtype="float64") for frame in (fresh, plan.saved.carried))
        tolerance = float(self._config.cluster.consistency_tolerance)
        differs = ~((np.isnan(new) & np.isnan(old)) | (np.abs(new - old) <= tolerance))
        if differs.any():
            gaps = np.abs(new - old)[differs]
            finite = gaps[np.isfinite(gaps)]
            largest = f"{finite.max():.3g}" if finite.size else "none, only a null against a value"
            raise RuntimeError(
                f"the saved quarter {saved.date()} no longer reproduces ({int(differs.sum())} of {differs.size} cells differ, "
                f"largest {largest}): the labels or centroids changed; {_REBUILD}"
            )

    def _expand(self, plan: Plan, computed: dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
        """The rows to write: the saved tail as saved, then the trading days after it from the due quarters and the carried saved quarter."""
        # merge: the frames by publication day; the newest saved quarter is the saved rows of the newest day
        cfg = self._config.cluster
        quarters = {plan.quarters.publication[t]: computed[t] for t in plan.quarters.due}
        if plan.saved.carried is not None and plan.quarters.saved_quarter is not None:
            quarters[plan.quarters.publication[plan.quarters.saved_quarter]] = plan.saved.carried

        # compute: the days to fill, from the history start or after the newest saved day
        calendar = plan.calendar
        days = (
            calendar[calendar >= pd.Timestamp(cfg.history_start)] if plan.saved.window.last is None else calendar[calendar > plan.saved.window.last]
        )
        daily = expand_daily(quarters, days, int(cfg.max_age_calendar_days))
        if plan.saved.tail is None:
            return daily

        # post-process: the saved rows in the part's column order, then the new days
        tail = plan.saved.tail[["date", "ticker", *plan.columns]].assign(date=pd.to_datetime(plan.saved.tail["date"]))
        return tail if daily.empty else pd.concat([tail.astype({column: "float32" for column in plan.columns}), daily], ignore_index=True)

    def _save(self, plan: Plan, rows: pd.DataFrame) -> None:
        """One replace (full) or one inclusive tail append of `rows`; rows with no feature value are dropped."""
        written = write_part(self._store, self._part.table, rows, plan.saved.window, drop_empty=True, existing_columns=plan.saved.columns)
        if written == COLUMNS_CHANGED:
            raise RuntimeError(f"{self._part.table} column set changed during the run: {_REBUILD}")
