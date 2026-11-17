"""Single-quarter statistics of each manager's cleaned book, against the preceding quarter's book and aggregate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_aggregate.utils.manager_profile.reference import ASSETS, SECTOR_COLUMNS, UNMAPPED

_CAPS = ("small", "mid", "large")
#: Columns of one quarter's statistics, in the order of the table's profile block.
STAT_COLUMNS = (
    "n_positions",
    "hhi",
    "turnover",
    "new_position_rate",
    "book_return",
    *(f"pct_{asset}" for asset in ASSETS),
    *(f"pct_cap_{cap}" for cap in _CAPS),
    *SECTOR_COLUMNS,
    "active_share",
    "book_value_usd",
    "mapped_share",
)


def compact_book(lines: pd.DataFrame) -> pd.DataFrame:
    """Each manager's equity positions as `cik, key, asset, shares, value, weight` (weight over the equity total), kept for the next quarter."""
    equity = lines[lines["kind"].eq("common")]
    weight = equity["value"] / equity.groupby("cik")["value"].transform("sum")
    return equity[["cik", "key", "asset", "shares", "value"]].assign(weight=weight).reset_index(drop=True)


def _equity(lines: pd.DataFrame, prev_agg: pd.DataFrame | None) -> pd.DataFrame:
    """Equity lines with `ew` (equity weight), the previous aggregate's `cap` and `agg_weight` of the key, and `agg_known`."""
    equity = lines[lines["kind"].eq("common")].copy()
    equity["ew"] = equity["value"] / equity.groupby("cik")["value"].transform("sum")
    equity["agg_known"] = prev_agg is not None
    if prev_agg is None:
        return equity.assign(cap=np.nan, agg_weight=np.nan)
    return equity.assign(cap=equity["key"].map(prev_agg["cap"]), agg_weight=equity["key"].map(prev_agg["weight"]))


def _asset_mix(lines: pd.DataFrame) -> pd.DataFrame:
    """Share of the book's total reported value by asset type (a type the book lacks is a true 0)."""
    value = lines.pivot_table(index="cik", columns="asset", values="value", aggfunc="sum", fill_value=0.0).reindex(
        columns=list(ASSETS), fill_value=0.0
    )
    return value.div(value.sum(axis=1), axis=0).rename(columns=lambda asset: f"pct_{asset}")


def _concentration(equity: pd.DataFrame) -> pd.DataFrame:
    """Number of equity positions and the Herfindahl index of their weights."""
    grouped = equity.assign(sq=equity["ew"] ** 2).groupby("cik")
    return pd.DataFrame({"n_positions": grouped.size().astype(float), "hhi": grouped["sq"].sum()})


def _sector_cap(equity: pd.DataFrame) -> pd.DataFrame:
    """Sector shares of equity weight, cap shares of equity weight (null without a previous aggregate) and `mapped_share`."""
    # Sector mix sums to 1 over the twelve columns.
    sector = equity.pivot_table(index="cik", columns="sector_col", values="ew", aggfunc="sum", fill_value=0.0)
    sector = sector.reindex(columns=list(SECTOR_COLUMNS), fill_value=0.0)

    # Cap mix and mapped share exist only where the previous quarter's aggregate sizes the key.
    known = equity.groupby("cik")["agg_known"].first()
    cap = equity.pivot_table(index="cik", columns="cap", values="ew", aggfunc="sum", fill_value=0.0).reindex(columns=list(_CAPS), fill_value=0.0)
    cap = cap.rename(columns=lambda name: f"pct_cap_{name}").where(known, np.nan)
    sized = equity["cap"].notna() & equity["sector_col"].ne(UNMAPPED)
    mapped = equity["ew"].where(sized, 0.0).groupby(equity["cik"]).sum().where(known, np.nan).rename("mapped_share")
    return pd.concat([sector, cap, mapped], axis=1)


def _turnover(equity: pd.DataFrame, prev_book: pd.DataFrame | None, threshold: float) -> pd.DataFrame:
    """Turnover and new-position rate against the immediately preceding quarter's book (rows only for managers in it).

    Turnover is the previous weight of the positions that exited or whose shares moved by more than `threshold`, over the previous weight of
    the positions that can be compared (an exit, or shares known in both quarters).
    """
    # No preceding quarter, no comparison.
    if prev_book is None:
        return pd.DataFrame(columns=["turnover", "new_position_rate"], dtype=float)

    # Previous positions against the current ones.
    merged = prev_book.merge(equity[["cik", "key", "shares"]], on=["cik", "key"], how="left", suffixes=("_prev", "_cur"), indicator=True)
    exited = merged["_merge"].eq("left_only")
    comparable = exited | (merged["shares_prev"].gt(0) & merged["shares_cur"].gt(0))
    moved = (merged["shares_cur"] - merged["shares_prev"]).abs().div(merged["shares_prev"]).gt(threshold)
    changed = comparable & (exited | moved)
    turnover = (
        merged["weight"].where(changed, 0.0).groupby(merged["cik"]).sum() / merged["weight"].where(comparable, 0.0).groupby(merged["cik"]).sum()
    )

    # Share of the current positions that the preceding book did not hold.
    held = prev_book[["cik", "key"]].assign(held=True)
    current = equity[["cik", "key"]].merge(held, on=["cik", "key"], how="left")
    new_rate = (~current["held"].fillna(False).astype(bool)).groupby(current["cik"]).mean()
    rate = new_rate.reindex(turnover.index)
    return pd.DataFrame({"turnover": turnover.where(np.isfinite(turnover)), "new_position_rate": rate})


def _book_return(equity: pd.DataFrame, prev_book: pd.DataFrame | None, cfg: DictConfig) -> pd.Series:
    """Price return of the preceding quarter's US equity positions held fixed in shares, by previous value, over the names still in the book.

    Each name's return is its implied price (value / shares) ratio, so a trade does not count as a gain. A name whose price ratio is outside
    `return_price_band`, or whose shares moved by more than `split_share_move` while its value moved by at most `split_value_tolerance` (a
    split), is left out.
    """
    # No preceding quarter, no return.
    if prev_book is None:
        return pd.Series(dtype=float, name="book_return")

    # Positions held in both quarters with a usable price.
    held = prev_book[prev_book["asset"].eq("equity_us") & prev_book["shares"].gt(0) & prev_book["value"].gt(0)]
    both = held.merge(equity[["cik", "key", "shares", "value"]], on=["cik", "key"], suffixes=("_prev", "_cur"))
    both = both[both["shares_cur"].gt(0) & both["value_cur"].gt(0)]

    # Drop price outliers and splits.
    shares_ratio, value_ratio = both["shares_cur"] / both["shares_prev"], both["value_cur"] / both["value_prev"]
    price_ratio = value_ratio / shares_ratio
    low, high = (float(bound) for bound in cfg.return_price_band)
    split = np.log(shares_ratio).abs().gt(np.log1p(float(cfg.split_share_move))) & np.log(value_ratio).abs().le(
        np.log1p(float(cfg.split_value_tolerance))
    )
    kept = both[price_ratio.between(low, high) & ~split]

    # Previous-value weighted price return.
    gain = (kept["value_prev"] * (price_ratio[kept.index] - 1.0)).groupby(kept["cik"]).sum()
    return (gain / kept.groupby("cik")["value_prev"].sum()).rename("book_return")


def _active_share(equity: pd.DataFrame) -> pd.Series:
    """Half the summed absolute weight gap to the previous aggregate book (names the manager does not hold count in full)."""
    gap = (equity["ew"] - equity["agg_weight"].fillna(0.0)).abs()
    held = equity["agg_weight"].fillna(0.0)
    share = 0.5 * (gap.groupby(equity["cik"]).sum() + 1.0 - held.groupby(equity["cik"]).sum())
    return share.where(equity.groupby("cik")["agg_known"].first(), np.nan).rename("active_share")


def quarter_stats(lines: pd.DataFrame, prev_book: pd.DataFrame | None, prev_agg: pd.DataFrame | None, cfg: DictConfig) -> pd.DataFrame:
    """One row per manager (`cik`, `period`, `filed_date` of its book, STAT_COLUMNS) of the quarter's cleaned `lines`.

    `prev_book` (`compact_book`) and `prev_agg` (`quarter_aggregate`) are the immediately preceding quarter's, None when it is missing;
    the statistics that need them read null.
    """
    # Equity weights with the previous aggregate's evidence.
    equity = _equity(lines, prev_agg)

    # Statistics by family, joined on the manager.
    book = lines.groupby("cik")
    parts = [
        _asset_mix(lines),
        _concentration(equity),
        _sector_cap(equity),
        _turnover(equity, prev_book, float(cfg.turnover_threshold)),
        _book_return(equity, prev_book, cfg),
        _active_share(equity),
        book["value"].sum().rename("book_value_usd"),
    ]
    stats = pd.concat(parts, axis=1).reindex(book.size().index)

    # A book without equity holds zero positions (known); its other equity statistics stay null.
    stats["n_positions"] = stats["n_positions"].fillna(0.0)
    out = stats.reindex(columns=list(STAT_COLUMNS)).astype(float)
    out.insert(0, "period", book["period"].first())
    out.insert(1, "filed_date", book["filing_date"].first())
    return out.reset_index()
