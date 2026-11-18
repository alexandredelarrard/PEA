"""Per-cluster 13F features of one report quarter, from the company books of the cluster's managers (REQ-011, REQ-012).

`cluster_feature_frame` serves every cluster: it keeps the managers of the member list that filed a common book in both quarters and
turns their lines into the cluster's configured features per ticker. Each helper is flat (pandas and numpy only).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_aggregate.utils.sec_thirteen.cluster_config import ClusterSpec

#: Columns of the market frame (index ticker): quarter-end market cap now and one quarter back, shares outstanding, 250-day volatility.
MARKET_COLUMNS = ("market_cap", "market_cap_prev", "shares_outstanding", "volatility")
#: A share count moved by a split when holders' ratios have a median beyond 1.3x (up or down) and 60% of them lie within 5% of it.
_SPLIT_MIN_LOG = float(np.log(1.3))
_SPLIT_AGREEMENT = 0.6
_SPLIT_TOLERANCE = 0.05
#: A line is near its cap at 4% of its manager's common book; a top-10 line is one of the manager's ten largest common companies.
_NEAR_CAP_WEIGHT = 0.04
_TOP_RANK = 10
_OPTION_KINDS = ["call", "put"]


@dataclass(frozen=True)
class QuarterBooks:
    """The company books (`cluster_lines.BOOK_COLUMNS`) of the quarter, the quarter before and the same quarter a year before."""

    cur: pd.DataFrame
    prev: pd.DataFrame
    prev4: pd.DataFrame


def _split_factors(cur: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    """Per ticker, the factor a split put on every holder's share count between the quarters (1 where the holders do not agree on one)."""
    held = [book.loc[book["kind"].eq("common") & book["shares"].gt(0), ["cik", "ticker", "shares"]] for book in (cur, prev)]
    pairs = held[0].merge(held[1], on=["cik", "ticker"], suffixes=("", "_p"))
    ratio = pairs["shares"] / pairs["shares_p"]
    median = ratio.groupby(pairs["ticker"]).median()
    near = ((ratio / pairs["ticker"].map(median) - 1).abs() < _SPLIT_TOLERANCE).astype(float).groupby(pairs["ticker"]).mean()
    is_split = (median.map(np.log).abs() > _SPLIT_MIN_LOG) & near.ge(_SPLIT_AGREEMENT)
    return median.where(is_split, 1.0)


def _matched(books: QuarterBooks, ciks: list[str], split: pd.Series) -> pd.DataFrame:
    """One row per (manager, ticker) held by a matched manager now or a quarter back, with the flags and amounts the features sum.

    `shares_prev` is in today's share units (split factor applied); a line without a share count is not held; `held4` is null when no
    matched manager filed a book four quarters back.
    """
    # The two quarters side by side; a side the manager lacks reads 0 shares and no rank.
    cur, prev, old = (book[book["kind"].eq("common") & book["cik"].isin(ciks)] for book in (books.cur, books.prev, books.prev4))
    pairs = cur[["cik", "ticker", "shares", "value", "rank", "book_total"]].merge(
        prev[["cik", "ticker", "shares", "value", "rank"]], on=["cik", "ticker"], how="outer", suffixes=("", "_p")
    )
    shares = pairs["shares"].fillna(0.0)
    shares_prev = pairs["shares_p"].fillna(0.0) * pairs["ticker"].map(split).fillna(1.0)
    held, held_prev = shares.gt(0), shares_prev.gt(0)

    # Flags and amounts.
    value = pairs["value"].fillna(0.0)
    keys = pd.MultiIndex.from_frame(pairs[["cik", "ticker"]])
    held4 = pd.Series(keys.isin(pd.MultiIndex.from_frame(old[["cik", "ticker"]])), index=pairs.index) & held
    return pd.DataFrame(
        {
            "ticker": pairs["ticker"],
            "held": held,
            "new": held & ~held_prev,
            "exit": ~held & held_prev,
            "flow": shares - shares_prev,
            "top": pairs["rank"].le(_TOP_RANK),
            "top_prev": pairs["rank_p"].le(_TOP_RANK),
            "value": value,
            "value_prev": pairs["value_p"].fillna(0.0),
            "near_cap": (value / pairs["book_total"]).ge(_NEAR_CAP_WEIGHT) & held,
            "held4": held4.astype(float) if not old.empty else np.nan,
        }
    )


def _flows(pairs: pd.DataFrame, n_managers: int, shares_out: pd.Series) -> pd.DataFrame:
    """Per ticker: the holder counts and sums (`n_hold`, `exit`, `value`, `value_prev`, `near_cap`) and the first-four and stickiness features."""
    total = pairs.drop(columns="ticker").groupby(pairs["ticker"]).sum(min_count=1)
    flows = pd.DataFrame(
        {
            "n_hold": total["held"].astype(float),
            "exit": total["exit"].astype(float),
            "value": total["value"],
            "value_prev": total["value_prev"],
            "near_cap": total["near_cap"].astype(float),
        }
    )
    shares_out = shares_out.where(shares_out.gt(0)).reindex(flows.index)
    flows["net_breadth"] = (total["new"] - total["exit"]) / (flows["n_hold"] + flows["exit"]).replace(0, np.nan)
    flows["new_position_rate"] = total["new"] / n_managers
    flows["net_share_flow"] = total["flow"] / shares_out
    flows["top10_chg"] = (total["top"] - total["top_prev"]) / n_managers
    flows["stickiness"] = total["held4"] / flows["n_hold"].replace(0, np.nan)
    return flows


def _rank_gap(flows: pd.DataFrame, books: QuarterBooks, ciks: list[str], market: pd.DataFrame) -> pd.DataFrame:
    """`rank_gap` (percentile rank of the cluster's aggregate weight minus that of market cap) and its change from the same managers a quarter back."""
    # Aggregate weight over the managers' whole common books; a quarter back only the tickers they held then.
    mine = [book[book["kind"].eq("common") & book["cik"].isin(ciks)] for book in (books.cur, books.prev)]
    totals = [rows.drop_duplicates("cik")["book_total"].sum() for rows in mine]
    quarters = (
        (flows["value"] / totals[0], market["market_cap"], pd.Series(True, index=flows.index)),
        (flows["value_prev"] / totals[1], market["market_cap_prev"], flows["value_prev"].gt(0)),
    )
    gaps = []
    for weight, cap, listed in quarters:
        rank_weight = weight.where(listed).rank(pct=True)
        rank_cap = cap.reindex(flows.index).where(listed).rank(pct=True)
        gaps.append(rank_weight - rank_cap)
    return pd.DataFrame({"rank_gap": gaps[0], "rank_gap_chg": gaps[0] - gaps[1]})


def _slot(flows: pd.DataFrame, books: QuarterBooks, ciks: list[str], market: pd.DataFrame) -> pd.DataFrame:
    """The cluster-type features: `cap_proximity`, `option_intensity` and `flow_x_vol` (flow times volatility rank minus one half)."""
    options = books.cur[books.cur["kind"].isin(_OPTION_KINDS) & books.cur["cik"].isin(set(ciks))]
    by_kind = options.pivot_table(index="ticker", columns="kind", values="value", aggfunc="sum")
    option_value = by_kind.reindex(index=flows.index, columns=_OPTION_KINDS).fillna(0.0).sum(axis=1)
    volatility_rank = market["volatility"].reindex(flows.index).rank(pct=True)
    return pd.DataFrame(
        {
            "cap_proximity": flows["near_cap"] / flows["n_hold"].replace(0, np.nan),
            "option_intensity": option_value / (flows["value"] + option_value).replace(0, np.nan),
            "flow_x_vol": flows["net_share_flow"] * (volatility_rank - 0.5),
        }
    )


def cluster_feature_frame(books: QuarterBooks, members: Collection[str], market: pd.DataFrame, spec: ClusterSpec) -> pd.DataFrame:
    """The `spec.features` of one cluster for the quarter of `books.cur`: index ticker, float32, null where unknown.

    Matched managers are the `members` with a common book in both quarters. A ticker needs `spec.min_holders` managers that hold or
    exited it, else it has no row (null); `stickiness` is null without a matched manager's book four quarters back. `market` has the
    `MARKET_COLUMNS` per ticker (quarter-end values; ranks of weight, market cap and volatility run over the tickers of the matched books).
    """
    # clean: managers with a common book in both quarters
    filed = [set(book.loc[book["kind"].eq("common"), "cik"]) for book in (books.cur, books.prev)]
    matched = sorted(set(members) & filed[0] & filed[1])
    if not matched:
        return pd.DataFrame(columns=list(spec.features), index=pd.Index([], name="ticker"), dtype="float32")

    # compute: split factors, the matched lines, then the features per ticker
    pairs = _matched(books, matched, _split_factors(books.cur, books.prev))
    flows = _flows(pairs, len(matched), market["shares_outstanding"])
    frame = pd.concat([flows, _rank_gap(flows, books, matched, market), _slot(flows, books, matched, market)], axis=1)

    # post-process: no value below the minimum number of managers
    enough = (frame["n_hold"] + frame["exit"]) >= spec.min_holders
    return frame.loc[enough, list(spec.features)].replace([np.inf, -np.inf], np.nan).astype("float32")
