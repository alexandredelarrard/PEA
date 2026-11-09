"""One sparse, public-information feature history for both fixed manager cohorts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import PublicBookSnapshot
from src.data_aggregate.utils.sec_thirteen.superinvestor_metrics import (
    BehaviorMarket,
    comparable_pairs,
    holding_duration,
    join_pairs,
    level_metrics,
    prepare_level_sample,
    prepare_pair_sample,
    prepare_whole_book_sample,
    quantity_metrics,
    summarize_books,
    whole_book_metrics,
)
from src.data_aggregate.utils.sec_thirteen.value_basis import nullable_value_to_daily


def feature_names() -> tuple[str, ...]:
    """The reviewed raw-feature contract, independent of the configured K."""
    metrics = (
        "holders",
        "holders_yoy",
        "conviction_weight",
        "max_conviction",
        "top10_holders",
        "breadth_chg",
        "shares_chg",
        "shares_chg_yoy",
        "quarters_held",
        "trade_balance",
        "relative_position_chg",
        "pooled_weight",
        "pooled_weight_chg",
    )
    return tuple(f"ic_super_{prefix}{metric}" for prefix in ("", "topk_") for metric in metrics)


EMISSION = dict.fromkeys(feature_names(), "raw")


@dataclass(frozen=True)
class SuperinvestorFeatureHistory:
    levels: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


def compute_superinvestor_feature_history(
    snapshots: Sequence[PublicBookSnapshot], cohorts: Mapping[str, frozenset[str]], universe: tuple[str, ...], market: BehaviorMarket
) -> SuperinvestorFeatureHistory:
    """Summarize every revision and join every compared pair once, then reduce both masks per event with identical equations."""
    index = pd.Index(universe, dtype=object)
    event_rows: list[dict[str, pd.Series]] = []
    diagnostics = []
    books = dict.fromkeys(book for snapshot in snapshots for book in (*snapshot.archive.values(), *snapshot.level_books.values()))
    table = summarize_books(list(books), index)
    selections = [{lag: comparable_pairs(snapshot, lag) for lag in (1, 4)} for snapshot in snapshots]
    compared = dict.fromkeys((current, prior) for selected in selections for lag in (1, 4) for _, current, prior in selected[lag])
    pairs = join_pairs(list(compared), table)
    for snapshot, selected in zip(snapshots, selections, strict=True):
        durations = {cik: holding_duration(book, snapshot.archive, table) for cik, book in snapshot.level_books.items() if book.support.comparable}
        level = prepare_level_sample(snapshot, table, durations, index)
        quarter, year = prepare_pair_sample(selected[1], pairs, market), prepare_pair_sample(selected[4], pairs, market)
        whole_sample = prepare_whole_book_sample(quarter)
        for name, cohort in cohorts.items():
            levels, coverage = level_metrics(level, cohort, index)
            changes = quantity_metrics(quarter, cohort, index)
            annual = quantity_metrics(year, cohort, index)
            whole, priced = whole_book_metrics(whole_sample, cohort, index)
            metrics = {**levels, **changes, **whole, "holders_yoy": annual["breadth_chg"], "shares_chg_yoy": annual["shares_chg"]}
            prefix = f"ic_super_{name}_" if name else "ic_super_"
            event_rows.append({prefix + metric: values.replace([np.inf, -np.inf], np.nan) for metric, values in metrics.items()})
            diagnostics.append(
                {
                    "date": snapshot.day,
                    "cohort": name or "full",
                    "common_period": snapshot.common_period,
                    "configured": len(cohort),
                    **coverage,
                    "quarter_pairs": len(quarter.managers & cohort),
                    "year_pairs": len(year.managers & cohort),
                    "whole_book_pairs": priced,
                }
            )

    days = pd.DatetimeIndex([snapshot.day for snapshot in snapshots], name="as_of")
    series = {name: [row[name] for row in event_rows if name in row] for name in EMISSION}
    levels = {name: pd.DataFrame(values, index=days, columns=universe, dtype="float32") for name, values in series.items()}
    return SuperinvestorFeatureHistory(levels, pd.DataFrame(diagnostics))


def project_superinvestor_features(history: SuperinvestorFeatureHistory, frames: PriceFrames, source_start: pd.Timestamp) -> pd.DataFrame:
    """Carry observations without decay, retaining explicit expiry/unknown barriers."""
    df_panel = frames.skeleton().set_index(["date", "ticker"])
    for name, sparse in history.levels.items():
        if sparse.empty:
            df_panel["f_" + name] = np.nan
            continue

        # An explicit unknown supersedes known state, including at a scheduled expiry.
        values = cast(pd.Series, sparse.rename_axis(index="as_of", columns="ticker").stack(future_stack=True))
        df_history = values.rename(name).reset_index()
        daily = nullable_value_to_daily(df_history, name, frames.trading_index)
        daily = daily.reindex(index=frames.trading_index, columns=frames.universe)
        daily.loc[daily.index < source_start] = np.nan
        daily = daily.where(frames.availability) if frames.availability is not None else daily
        df_panel["f_" + name] = daily.stack(future_stack=True).reindex(df_panel.index).astype("float32")

    # Keep all 26 declared columns even if a metric or the entire tail is unknown.
    columns = ["f_" + name for name in EMISSION]
    return df_panel.reindex(columns=columns).astype("float32").reset_index()
