"""One sparse, public-information feature history for both fixed manager cohorts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import PublicBookSnapshot
from src.data_aggregate.utils.sec_thirteen.superinvestor_metrics import (
    BehaviorMarket,
    BookSummaries,
    level_metrics,
    prepare_level_sample,
    prepare_pair_sample,
    prepare_whole_book_sample,
    quantity_metrics,
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
    """Prepare matched endpoint facts once and reduce both masks with identical equations."""
    rows: dict[str, list[pd.Series]] = {name: [] for name in EMISSION}
    diagnostics = []
    summaries = BookSummaries()
    days = []
    for snapshot in snapshots:
        quarter = prepare_pair_sample(snapshot, 1, summaries, market)
        year = prepare_pair_sample(snapshot, 4, summaries, market)
        summaries.prune_pairs()
        level = prepare_level_sample(snapshot, universe, summaries)
        whole_sample = prepare_whole_book_sample(quarter)
        days.append(snapshot.day)
        for name, cohort in cohorts.items():
            levels, coverage = level_metrics(level, cohort, universe)
            changes = quantity_metrics(quarter, cohort, universe)
            annual = quantity_metrics(year, cohort, universe)
            whole, priced = whole_book_metrics(whole_sample, cohort, universe)
            metrics = {**levels, **changes, **whole, "holders_yoy": annual["breadth_chg"], "shares_chg_yoy": annual["shares_chg"]}
            prefix = f"ic_super_{name}_" if name else "ic_super_"
            for metric, values in metrics.items():
                rows[prefix + metric].append(values.replace([np.inf, -np.inf], np.nan))
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

    levels = {
        name: pd.DataFrame(values, index=pd.DatetimeIndex(days, name="as_of"), columns=universe, dtype="float32") for name, values in rows.items()
    }
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
