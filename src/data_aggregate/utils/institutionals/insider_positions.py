"""Cumulative direct-account share ratios with independent disclosure clocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.quotes import lookup_asof
from src.data_aggregate.utils.institutionals.insider_quality import account_other_flows, security_class

MICRO = 1_000_000


def position_units(trades: pd.DataFrame, df_factor: pd.DataFrame | None) -> pd.DataFrame:
    """Put unclassed transaction-day shares and holdings in terminal units, `q * F_T / F_d`; keep source execution USD.

    `df_factor` is `terminal_basis_factor` of the saved basis. Class-titled rows and rows without F keep source
    quantities for scale-free same-day ratios, marked unsupported for cross-day and cross-account comparisons.
    """
    df_rows = trades.copy()
    df_rows["klass"] = security_class(df_rows["security_title"])
    df_rows["units_known"] = False
    if df_factor is None or df_factor.empty:
        return df_rows
    factor = lookup_asof(df_factor, df_rows["ticker"], df_rows["transaction_day"])

    # Convert only unclassed common rows with a known basis.
    df_rows["units_known"] = factor.gt(0) & np.isfinite(factor) & df_rows["transaction_day"].notna() & df_rows["klass"].eq("COMMON")
    for column in ("shares_n", "shares_owned_after"):
        df_rows[column] = (df_rows[column] * factor).where(df_rows["units_known"], df_rows[column])
    return df_rows


def cumulative_positions(
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    other_flows: pd.DataFrame,
    idx: pd.DatetimeIndex,
    days: int,
    populations: Mapping[str, int | None],
    excluded: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Net shares bought minus sold over holdings before the first window trade, one panel per output name.

    `populations` maps each output name to its population code (`None` = all insiders); `excluded` lists
    `ticker, owner_cik, start, stop` intervals in which an account's quantities are disputed. A cell sums, per ticker
    and over every share class, the direct accounts that traded in the window and have a supported balance chain;
    an unsupported account is left out of numerator, denominator and counts, so one bad account never blanks a
    ticker. Each lot keeps its own original disclosure expiry. The denominator is the latest included economic-day
    terminal minus active net shares, after balance-chain checks.
    """
    if trades.empty or idx.empty:
        return {}
    multiplicity = pd.to_numeric(trades.get("n_reporting_owners", pd.Series(np.nan, index=trades.index)), errors="coerce")
    account_known = (
        trades["direct_indirect"].eq("D")
        & trades["owner_cik"].ne("")
        & trades["transaction_day"].notna()
        & multiplicity.eq(1)
        & trades["security_title"].fillna("").str.strip().ne("")
        & ~trades.get("scope_unknown", pd.Series(False, index=trades.index)).fillna(False)
    )
    rows = trades[account_known].copy()
    rows["stop"] = rows["anchor"] + pd.Timedelta(days=days)
    rows["stop"] = rows["stop"].where(rows["visible_until"].isna() | rows["stop"].le(rows["visible_until"]), rows["visible_until"])
    rows["delta"] = rows["shares_n"].where(rows["code"].eq("P"), -rows["shares_n"])
    keys = ["ticker", "owner_cik", "klass"]
    day_groups = {} if positions.empty else dict(tuple(positions.groupby(keys, sort=False)))
    df_flows = pd.concat([other_flows, trades.loc[trades["transaction_day"].isna()]], ignore_index=True)
    flows = {} if df_flows.empty else dict(tuple(df_flows.dropna(subset=["day"]).groupby("ticker", sort=False)))
    disputed = dict(tuple(excluded.groupby(["ticker", "owner_cik"], sort=False))) if not excluded.empty else {}
    intervals: list[dict[str, object]] = []
    for key, group in rows.groupby(keys, sort=False):
        account_days = day_groups.get(key, positions.iloc[:0])
        df_flows = flows.get(key[0], pd.DataFrame(columns=["ticker", "owner_cik", "transaction_day", "day"]))
        blocked = account_other_flows(df_flows, key[0], key[1])
        intervals.extend(
            _account_intervals(cast(tuple[str, str, str], key), group, account_days, blocked, idx, disputed.get(key[:2], excluded.iloc[:0]))
        )
    columns = pd.Index(sorted(trades["ticker"].unique()), name="ticker")
    out = {name: pd.DataFrame(np.nan, index=idx, columns=columns) for name in populations} if not rows.empty else {}
    if intervals:
        events = pd.DataFrame(intervals)
        for name, code in populations.items():
            events_in = events if code is None else events[events["population"].eq(code)]
            out[name] = _pool(events_in, idx).reindex(columns=columns)
    return out


def _account_intervals(
    key: tuple[str, str, str], lots: pd.DataFrame, positions: pd.DataFrame, flows: pd.DataFrame, idx: pd.DatetimeIndex, excluded: pd.DataFrame
) -> list[dict[str, object]]:
    """Sweep public version/expiry boundaries instead of allocating account-by-day frames."""
    time_cols = ["visible_from", "stop", "visible_until", "transaction_day"]
    lot = {column: lots[column].to_numpy(dtype="datetime64[ns]") for column in time_cols}
    positions = positions.reindex(columns=["visible_from", "visible_until", "transaction_day", "starting", "net_shares"])
    pos = {column: positions[column].to_numpy(dtype="datetime64[ns]") for column in ("visible_from", "visible_until", "transaction_day")}
    flow_day = flows["transaction_day"].to_numpy(dtype="datetime64[ns]")
    flow_filed = flows["day"].to_numpy(dtype="datetime64[ns]")
    flow_public = flows["visible_from"].to_numpy(dtype="datetime64[ns]")
    flow_stop = flows["visible_until"].to_numpy(dtype="datetime64[ns]")
    dispute_start = excluded["start"].to_numpy(dtype="datetime64[ns]")
    dispute_stop = excluded["stop"].to_numpy(dtype="datetime64[ns]")
    points = np.unique(
        np.concatenate(
            [
                lot["visible_from"],
                lot["stop"],
                pos["visible_from"],
                pos["visible_until"],
                flow_public,
                flow_stop,
                dispute_start,
                dispute_stop,
                np.array([idx.min(), idx.max() + pd.Timedelta(days=1)], dtype="datetime64[ns]"),
            ]
        )
    )
    points = points[~np.isnat(points)]
    points = points[(points >= idx.min().to_datetime64()) & (points <= (idx.max() + pd.Timedelta(days=1)).to_datetime64())]
    delta = lots["delta"].to_numpy(dtype=float)
    population = lots["population"].to_numpy(dtype=int)
    units = lots["units_known"].to_numpy(dtype=bool)
    uncertain = lots["quality_uncertain"].fillna(False).to_numpy(dtype=bool)
    starting = positions["starting"].to_numpy(dtype=float)
    terminal = starting + positions["net_shares"].to_numpy(dtype=float)
    output = []
    for point, stop in zip(points[:-1], points[1:], strict=True):
        active = (lot["visible_from"] <= point) & (lot["stop"] > point)
        if not active.any():
            continue
        first, last = lot["transaction_day"][active].min(), lot["transaction_day"][active].max()
        event = {
            "ticker": key[0],
            "owner_cik": key[1],
            "klass": key[2],
            "start": point,
            "stop": stop,
            "net": np.nan,
            "prior": np.nan,
            "unit_basis": "NORMALIZED" if units[active].all() else np.datetime_as_string(first, unit="D"),
            "population": int(population[active].min()),
        }
        output.append(event)
        if ((dispute_start <= point) & (point < dispute_stop)).any():
            continue
        public = (lot["visible_from"] <= point) & (np.isnat(lot["visible_until"]) | (lot["visible_until"] > point))
        span = public & (lot["transaction_day"] >= first) & (lot["transaction_day"] <= last)
        visible = (pos["visible_from"] <= point) & (np.isnat(pos["visible_until"]) | (pos["visible_until"] > point))
        visible &= (pos["transaction_day"] >= first) & (pos["transaction_day"] <= last)
        ordered = np.flatnonzero(visible)[np.argsort(pos["transaction_day"][visible])]
        # Expired trades interleaved between active economic days make a literal start ambiguous.
        possible_day = ((flow_day >= first) & (flow_day <= last)) | (np.isnat(flow_day) & (flow_filed >= first))
        blocked = (flow_public <= point) & (np.isnat(flow_stop) | (flow_stop > point)) & possible_day
        interleaved = span & ~active & (lot["transaction_day"] > first) & (lot["transaction_day"] < last)
        if blocked.any() or uncertain[span].any() or interleaved.any():
            continue
        economic_days = np.unique(lot["transaction_day"][active])
        if len(ordered) != len(economic_days) or not np.array_equal(pos["transaction_day"][ordered], economic_days):
            continue
        if len(economic_days) > 1 and not units[span].all():
            continue
        if len(ordered) > 1 and not np.allclose(starting[ordered[1:]], terminal[ordered[:-1]], rtol=0, atol=1e-6):
            continue
        net = float(delta[active].sum())
        prior = float(terminal[ordered[-1]] - net)
        if not np.isfinite(prior) or prior < 0 or not np.isfinite(delta[active]).all():
            continue
        event.update(net=net, prior=prior)
    return output


def _pool(events: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Add supported account totals once per ticker; require compatible units and positive pooled holdings.

    An event without a supported balance chain contributes to nothing: not to the numerator, the holdings, the unit-basis
    count or the support count.
    """
    if events.empty:
        return pd.DataFrame(index=idx)
    columns = pd.Index(sorted(events["ticker"].unique()), name="ticker")
    supported = np.isfinite(events["net"]) & np.isfinite(events["prior"]) & events["prior"].ge(0)
    events = events[supported]
    if events.empty:
        return pd.DataFrame(np.nan, index=idx, columns=columns)
    col = columns.get_indexer(pd.Index(events["ticker"]))
    start = idx.searchsorted(events["start"].to_numpy(), side="left")
    stop = idx.searchsorted(events["stop"].to_numpy(), side="left")
    shape = (len(idx) + 1, len(columns))

    # Sum account intervals in the same order for numerator, holdings and support count.
    numerator = _accumulate_intervals(events["net"].to_numpy(dtype=float), start, stop, col, shape)
    denominator = _accumulate_intervals(events["prior"].to_numpy(dtype=float), start, stop, col, shape)
    bases = np.zeros(shape)
    # Union each basis's intervals before counting; never allocate a panel per economic date.
    for (ticker, _), group in events.assign(_start=start, _stop=stop).groupby(["ticker", "unit_basis"], sort=False):
        ordered = group.sort_values("_start")
        begins = ordered["_start"].to_numpy(dtype=np.intp)
        ends = np.maximum.accumulate(ordered["_stop"].to_numpy(dtype=np.intp))
        segments = np.r_[0, np.flatnonzero(begins[1:] > ends[:-1]) + 1]
        last = np.r_[segments[1:] - 1, len(ends) - 1]
        column = cast(int, columns.get_loc(ticker))
        np.add.at(bases, (begins[segments], column), 1)
        np.add.at(bases, (ends[last], column), -1)
    known = (denominator > 0) & np.isfinite(denominator) & np.isfinite(numerator) & (bases.cumsum(axis=0)[:-1] == 1)
    return pd.DataFrame(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=known), index=idx, columns=columns)


def _accumulate_intervals(values: np.ndarray, start: np.ndarray, stop: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Sum each value on its half-open grid interval using start/stop differences, exactly.

    Values are accumulated as integer micro-shares: a float add-then-subtract leaves ~1e-6 residue where the
    true sum is 0, and a positive-denominator test then reads that residue as holdings.
    """
    scaled = np.rint(values * MICRO).astype(np.int64)
    differences = np.zeros(shape, dtype=np.int64)
    np.add.at(differences, (start, columns), scaled)
    np.add.at(differences, (stop, columns), -scaled)
    return differences.cumsum(axis=0)[:-1] / MICRO
