"""Manager profile: enrich `sec_13f_manager_details` with the category and the behavioural profile of each (cik, period) row.

`run_profile` writes the profile columns; `run_clustering` writes the `cluster` label (fitted fund clusters, categories, -1); `run` does both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.context import Context
from src.data_aggregate.utils.manager_profile.categories import apply_overrides, classify_names, load_category_overrides
from src.data_aggregate.utils.manager_profile.clustering import (
    assemble_labels,
    due_rows,
    eligible_fund_rows,
    keep_clusters,
    read_mapping,
    write_centroids,
)
from src.data_aggregate.utils.manager_profile.lines import clean_book, read_lines
from src.data_aggregate.utils.manager_profile.profile import profile_rows
from src.data_aggregate.utils.manager_profile.quarter_stats import compact_book, quarter_stats
from src.data_aggregate.utils.manager_profile.reference import classify_securities, prefix_tickers, quarter_aggregate, sharadar_reference
from src.data_aggregate.utils.manager_profile.refresh import PROFILE_COLUMNS, changed_rows, plan_refresh, profile_frame, stored_details
from src.data_aggregate.utils.sec_thirteen.cluster_config import FUND, validate_cluster_config
from src.data_store.schema import Tables
from src.modelling.transformers.cluster import ClusterTransformer
from src.utils.step import Step
from src.utils.string import run_date


@dataclass(frozen=True)
class Carry:
    """What a report quarter passes to the next: its compact book and per-ticker aggregate."""

    period: pd.Timestamp
    book: pd.DataFrame
    aggregate: pd.DataFrame


class StepManagerProfile(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube.manager_profile
        self._rules = config.manager_categories
        self._overrides = load_category_overrides(context.config_dir, self._rules)
        self._store = context.store

    def run(self, full: bool = False) -> int:
        """The profile, then the cluster labels (each rebuilt in full when `full`); returns the rows written by both."""
        return self.run_profile(full) + self.run_clustering(full)

    def run_clustering(self, full: bool = False) -> int:
        """Label the due rows of `sec_13f_manager_details` with their manager cluster; returns the rows written.

        `cluster.fit: false` only reads the saved centroids and labels the due rows (`full` relabels every due row, no refit); `fit: true`
        first fits k-means on the eligible fund rows of every quarter, keeps `keep` clusters, saves the JSON and relabels every due row.
        """
        started = time.monotonic()
        cfg = self._config.cluster
        validate_cluster_config(cfg)

        # load: the manager rows once, only the columns the labels need
        features = OmegaConf.to_container(cfg.features)
        names = [entry if isinstance(entry, str) else str(entry["name"]) for entry in cast(list, features)]
        wanted = ["cik", "period", "category", "cluster", "n_positions", *dict.fromkeys(["book_value_usd", *names])]
        df_stored = self._store.load(Tables.sec_13f_manager_details, columns=wanted, optional=True)
        if df_stored is None:
            self._log.info("manager clustering: no manager rows stored yet")
            return 0
        details = df_stored.assign(period=pd.to_datetime(df_stored["period"]).dt.normalize())

        # plan: the rows due a label; a fit relabels every due row because it moves every assignment
        due = due_rows(details, int(cfg.cluster_label_delay_days), run_date(None), relabel=full or bool(cfg.fit))
        if due.empty and not cfg.fit:
            self._log.info(
                "manager clustering: %d rows, none due; centroids not read, nothing written (%.1f s)", len(details), time.monotonic() - started
            )
            return 0

        # compute: fit and keep the clusters, or load them; then the labels of the due rows
        path = self._config_dir / str(cfg.centroids_path)
        if cfg.fit:
            spec = {"features": features, "group_by": "period", "k": cfg.k, "seed": cfg.seed, "n_init": cfg.n_init}
            transformer = ClusterTransformer(OmegaConf.create({**spec, "max_rows_per_group": cfg.max_rows_per_quarter}))
            transformer.fit(eligible_fund_rows(details, cfg.fit_filters))
            mapping = keep_clusters(transformer, cfg.exclude, int(cfg.keep))
            write_centroids(transformer, mapping, path)
        else:
            transformer, mapping = ClusterTransformer.load(path), read_mapping(path)
        categories = {kind: int(entry.id) for kind, entry in cfg.clusters.items() if kind != FUND}
        funds = eligible_fund_rows(details[details["period"].isin(due["period"].unique())], cfg.fit_filters)
        labels = assemble_labels(due, transformer.transform(funds), mapping, categories, cfg.fit_filters)

        # check: a fit leaves `fit: true` behind and moves every historical label
        if cfg.fit:
            self._log.warning("manager clustering: the fit is saved to %s but cluster.fit is still true in cluster.yml; set it back to false", path)
            self._log.warning("manager clustering: every label changed; rebuild the cube features with `build-sec-thirteen -F`")

        # save: the key and the label of each due row
        written = self._save(labels)
        self._log.info(
            "manager clustering: %d of %d rows due and labelled, counts by cluster %s (%.1f s)",
            written,
            len(details),
            labels["cluster"].value_counts().sort_index().to_dict(),
            time.monotonic() - started,
        )
        return written

    def run_profile(self, full: bool = False) -> int:
        """Profile the stale quarters of `sec_13f_manager_details` (all of them when `full`); returns the rows written."""
        started = time.monotonic()

        # load: the key, name and profile columns of the manager rows once; nothing else is read while every row is profiled
        wanted = ["cik", "period", "source_period", "manager_name", *PROFILE_COLUMNS]
        df_stored = self._store.load(Tables.sec_13f_manager_details, columns=wanted, optional=True)
        if df_stored is None:
            self._log.info("manager profile: no manager rows stored yet")
            return 0
        details = stored_details(df_stored)

        # plan: the quarters whose rows lack a profile or whose name changed
        category = apply_overrides(classify_names(details["manager_name"], self._rules), details["cik"], self._overrides)
        refresh = plan_refresh(details, category, full, int(self._cfg.window_quarters))
        if refresh is None:
            self._log.info("manager profile: %d rows, all profiled; 0 quarters read (%.1f s)", len(details), time.monotonic() - started)
            return 0

        # compute: one chronological pass over the quarters, then the 4-quarter profile of the rewritten rows
        qstats = self._quarter_pass(details, refresh.compute)
        profile = profile_frame(details, category, profile_rows(qstats, self._cfg), refresh.write)

        # save: only the rows that differ
        changed = self._save(changed_rows(profile, details))
        self._log.info(
            "manager profile: %d quarters read, %d rows rewritten of %d, %d changed (%.1f s)",
            len(refresh.compute),
            len(profile),
            len(details),
            changed,
            time.monotonic() - started,
        )
        return changed

    def _quarter_pass(self, details: pd.DataFrame, quarters: list[pd.Timestamp]) -> pd.DataFrame:
        """Single-quarter statistics of every manager over `quarters` in order; a quarter's book and aggregate feed only the next quarter end."""
        # load: the reference tables once
        ticker_columns = ["permaticker", "ticker", "category", "sector", "lastpricedate"]
        df_tickers = cast(pd.DataFrame, self._store.load(Tables.sharadar_tickers, columns=ticker_columns))
        df_map = cast(pd.DataFrame, self._store.load(Tables.cusip_ticker_map))
        reference, prefixes = sharadar_reference(df_tickers), prefix_tickers(df_map)

        # compute: quarter by quarter, carrying the previous quarter forward
        rows = {period: group[["cik", "source_period"]] for period, group in details.groupby("period")}
        stats: list[pd.DataFrame] = []
        carry: Carry | None = None
        for period in quarters:
            started = time.monotonic()
            raw = self._read_quarter(period, rows[period])
            if raw.empty:
                self._log.warning("manager profile: no 13F lines stored for %s", period.date())
                continue
            cleaned = clean_book(raw, self._cfg, prefixes)
            if cleaned.empty:
                self._log.warning("manager profile: no book for %s (every filer has only a short amendment)", period.date())
                continue
            lines = classify_securities(cleaned, reference, self._cfg)
            previous = carry if carry is not None and carry.period == period - pd.offsets.QuarterEnd(1) else None
            stats.append(quarter_stats(lines, previous.book if previous else None, previous.aggregate if previous else None, self._cfg))
            carry = Carry(period, compact_book(lines), quarter_aggregate(lines, self._cfg.cap_breakpoints))
            self._log.info(
                "manager profile: %s %d managers, %d lines (%.1f s)", period.date(), len(rows[period]), len(lines), time.monotonic() - started
            )
        return pd.concat(stats, ignore_index=True) if stats else pd.DataFrame(columns=["cik", "period"])

    def _read_quarter(self, period: pd.Timestamp, rows: pd.DataFrame) -> pd.DataFrame:
        """Markers and lines of the quarter for its managers (`cik`, `source_period`); a snapped row is read under its own period and labelled `period`."""
        snapped = rows["source_period"].ne(period)
        raw = read_lines(self._store, rows.loc[~snapped, "cik"].tolist(), [period])
        if not snapped.any():
            return raw

        # The snapped rows: one bounded read over their CIKs and periods, then only their exact (cik, period) pairs.
        pairs = rows.loc[snapped, ["cik", "source_period"]].rename(columns={"source_period": "period"})
        extra = read_lines(self._store, pairs["cik"].tolist(), sorted(pairs["period"].unique()))
        extra = extra.assign(period=pd.to_datetime(extra["period"])).merge(pairs, on=["cik", "period"]).assign(period=period)
        return pd.concat([raw, extra], ignore_index=True)

    def _save(self, df_changed: pd.DataFrame) -> int:
        """Upsert the key and the columns of `df_changed` only (the profile columns or the cluster label), returns the rows written."""
        if df_changed.empty:
            return 0
        return self._store.save(Tables.sec_13f_manager_details, df_changed)
