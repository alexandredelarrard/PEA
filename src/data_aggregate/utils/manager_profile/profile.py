"""Profile of a manager at a report quarter: the mean of its single-quarter statistics over the trailing calendar quarters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_aggregate.utils.manager_profile.quarter_stats import STAT_COLUMNS


def profile_rows(qstats: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Per (cik, period) row of `qstats`: each statistic averaged over the `window_quarters` quarter ends ending at the period.

    A statistic is null with fewer than `min_quarters` known values in the window; `book_value_usd` is the median; `n_quarters_used` counts the
    window's quarters that have a usable book. A quarter the manager did not file is not filled. A window quarter whose book was first
    filed (`filed_date`) after the row's own book is left out of that row's window; the row's own quarter always stays.
    """
    # Calendar quarter number of each row.
    window, minimum = int(cfg.window_quarters), int(cfg.min_quarters)
    stats = qstats.assign(quarter=qstats["period"].dt.year * 4 + qstats["period"].dt.quarter)

    # Every quarter belongs to the windows of the next `window` quarter ends; join each row to its own window.
    spread = pd.concat([stats.assign(target=stats["quarter"] + step) for step in range(window)]).drop(columns="period")
    spread = spread.rename(columns={"filed_date": "filed_date_member"})
    targets = stats[["cik", "period", "quarter", "filed_date"]].rename(columns={"quarter": "target"})
    members = targets.merge(spread, on=["cik", "target"])
    members = members[~members["filed_date_member"].gt(members["filed_date"])]

    # Mean (median for the book value) of the known values, null below the minimum.
    columns = [name for name in STAT_COLUMNS if name != "book_value_usd"]
    grouped = members.groupby(["cik", "period"])
    mean = grouped[columns].mean().where(grouped[columns].count() >= minimum)
    median = grouped["book_value_usd"].median().where(grouped["book_value_usd"].count() >= minimum)
    used = grouped.size().rename("n_quarters_used")
    out = pd.concat([mean, median, used], axis=1).reindex(columns=[*STAT_COLUMNS, "n_quarters_used"])

    # Below the minimum number of usable quarters nothing is reported.
    out.loc[out["n_quarters_used"] < minimum, list(STAT_COLUMNS)] = np.nan
    return out.reset_index()
