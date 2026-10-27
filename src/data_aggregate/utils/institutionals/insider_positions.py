"""Cumulative direct-account share ratios with independent disclosure clocks."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.inputs import QuantityBasis
from src.data_aggregate.utils.institutionals.insider_quality import account_other_flows, security_class


def position_units(trades: pd.DataFrame, basis: QuantityBasis | None) -> pd.DataFrame:
    """Rebase transaction-day shares and holdings together; preserve clean source execution USD.

    Unknown class/event evidence keeps source quantities for scale-free same-day ratios,
    while marking their cross-day and cross-account comparability unsupported.
    """
    df_rows = trades.copy()
    df_rows["klass"] = security_class(df_rows["security_title"])
    df_rows["declared_units_known"] = False
    if basis is None:
        df_rows["units_known"] = False
        return df_rows
    declared = basis.events.declared_units(df_rows["ticker"], df_rows["transaction_day"], df_rows["security_title"])
    df_rows["declared_units_known"] = declared.gt(0) & np.isfinite(declared) & df_rows["transaction_day"].notna()
    target = pd.Series(basis.target_date, index=df_rows.index)
    factor = basis.events.rebase(pd.Series(1.0, index=df_rows.index), df_rows["ticker"], df_rows["transaction_day"], target) * declared
    factor = factor.where(df_rows["klass"].eq("COMMON"))
    df_rows["units_known"] = factor.gt(0) & np.isfinite(factor) & df_rows["transaction_day"].notna()
    for column in ("shares_n", "shares_owned_after"):
        rebased = basis.events.rebase(df_rows[column] * declared, df_rows["ticker"], df_rows["transaction_day"], target)
        df_rows[column] = rebased.where(df_rows["units_known"], df_rows[column])
    return df_rows


def cumulative_positions(
    trades: pd.DataFrame, positions: pd.DataFrame, other_flows: pd.DataFrame, idx: pd.DatetimeIndex, days: int
) -> dict[str, pd.DataFrame]:
    """Pool window shares only when every eligible management account is supported.

    Each lot retains its own original disclosure expiry. The denominator is the latest
    included economic-day terminal minus active net shares, after balance-chain checks.
    Silent executives are absent; unsupported or possibly eligible management stays unknown.
    """
    if trades.empty or idx.empty:
        return {}
    multiplicity = pd.to_numeric(trades.get("n_reporting_owners", pd.Series(np.nan, index=trades.index)), errors="coerce")
    role_unknown = trades.get("role_unknown", trades["officer_title_clean"].eq("") & trades["is_officer"].eq(1)).fillna(False)
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
    possible = (trades["is_management"] | role_unknown) & ~trades["direct_indirect"].eq("I")
    possible &= ~(multiplicity.gt(1) & multiplicity.mod(1).eq(0)) & (~account_known | role_unknown)
    intervals = _possible_intervals(trades[possible], days)
    for key, group in rows.groupby(keys, sort=False):
        account_days = day_groups.get(key, positions.iloc[:0])
        df_flows = flows.get(key[0], pd.DataFrame(columns=["ticker", "owner_cik", "transaction_day", "day"]))
        blocked = account_other_flows(df_flows, key[0], key[1])
        intervals.extend(_account_intervals(cast(tuple[str, str, str], key), group, account_days, blocked, idx))
    columns = pd.Index(sorted(trades["ticker"].unique()), name="ticker")
    unknown = pd.DataFrame(np.nan, index=idx, columns=columns)
    out = {}
    if rows["code"].eq("P").any():
        out["ic_insider_purchase_pct_prior"] = unknown.copy()
    if trades["is_management"].any() or possible.any():
        out["ic_insider_management_position_change_60d"] = unknown.copy()
    if intervals:
        events = pd.DataFrame(intervals)
        if "ic_insider_purchase_pct_prior" in out:
            out["ic_insider_purchase_pct_prior"] = _pool(events[events["buy"].gt(0)], idx, "buy").reindex(columns=columns)
        if "ic_insider_management_position_change_60d" in out:
            out["ic_insider_management_position_change_60d"] = _pool(events[events["management"]], idx, "net").reindex(columns=columns)
    return out


def _possible_intervals(trades: pd.DataFrame, days: int) -> list[dict[str, object]]:
    """Keep unresolved management candidates visible until each original lot expires."""
    stop = trades["anchor"] + pd.Timedelta(days=days)
    stop = stop.where(trades["visible_until"].isna() | stop.le(trades["visible_until"]), trades["visible_until"])
    return trades.assign(start=trades["visible_from"], stop=stop, net=np.nan, buy=np.nan, prior=np.nan, management=True, unit_basis="UNKNOWN")[
        ["ticker", "owner_cik", "klass", "start", "stop", "net", "buy", "prior", "management", "unit_basis"]
    ].to_dict("records")


def _account_intervals(
    key: tuple[str, str, str], lots: pd.DataFrame, positions: pd.DataFrame, flows: pd.DataFrame, idx: pd.DatetimeIndex
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
    points = np.unique(
        np.concatenate(
            [
                lot["visible_from"],
                lot["stop"],
                pos["visible_from"],
                pos["visible_until"],
                flow_public,
                flow_stop,
                np.array([idx.min(), idx.max() + pd.Timedelta(days=1)], dtype="datetime64[ns]"),
            ]
        )
    )
    points = points[~np.isnat(points)]
    points = points[(points >= idx.min().to_datetime64()) & (points <= (idx.max() + pd.Timedelta(days=1)).to_datetime64())]
    delta = lots["delta"].to_numpy(dtype=float)
    purchase = lots["code"].eq("P").to_numpy()
    manager = lots["is_management"].to_numpy(dtype=bool)
    blank_officer = (lots["officer_title_clean"].eq("") & lots["is_officer"].eq(1)).to_numpy()
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
            "buy": np.nan,
            "prior": np.nan,
            "unit_basis": "NORMALIZED" if units[active].all() else np.datetime_as_string(first, unit="D"),
            "management": bool(manager[active].any()),
        }
        output.append(event)
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
        event.update(net=net if (manager[active] | blank_officer[active]).all() else np.nan, buy=float(delta[active & purchase].sum()), prior=prior)
    return output


def _pool(events: pd.DataFrame, idx: pd.DatetimeIndex, value: str) -> pd.DataFrame:
    """Add complete account totals once; require compatible units and positive pooled holdings."""
    if events.empty:
        return pd.DataFrame(index=idx)
    columns = pd.Index(sorted(events["ticker"].unique()), name="ticker")
    col = columns.get_indexer(pd.Index(events["ticker"]))
    start = idx.searchsorted(events["start"].to_numpy(), side="left")
    stop = idx.searchsorted(events["stop"].to_numpy(), side="left")
    shape = (len(idx) + 1, len(columns))

    # Sum account intervals in the same order for numerator, holdings and support count.
    supported = np.isfinite(events[value]) & np.isfinite(events["prior"]) & events["prior"].ge(0)
    numerator = _accumulate_intervals(events[value].where(supported, 0).to_numpy(dtype=float), start, stop, col, shape)
    denominator = _accumulate_intervals(events["prior"].where(supported, 0).to_numpy(dtype=float), start, stop, col, shape)
    count = _accumulate_intervals(np.ones(len(events)), start, stop, col, shape)
    supported_count = _accumulate_intervals(supported.to_numpy(dtype=float), start, stop, col, shape)
    bases = np.zeros(shape)
    # Union each basis's intervals before counting; never allocate a panel per economic date.
    for (ticker, _, _), group in events.assign(_start=start, _stop=stop).groupby(["ticker", "klass", "unit_basis"], sort=False):
        ordered = group.sort_values("_start")
        begins = ordered["_start"].to_numpy(dtype=np.intp)
        ends = np.maximum.accumulate(ordered["_stop"].to_numpy(dtype=np.intp))
        segments = np.r_[0, np.flatnonzero(begins[1:] > ends[:-1]) + 1]
        last = np.r_[segments[1:] - 1, len(ends) - 1]
        column = cast(int, columns.get_loc(ticker))
        np.add.at(bases, (begins[segments], column), 1)
        np.add.at(bases, (ends[last], column), -1)
    known = (
        (count > 0)
        & (count == supported_count)
        & (denominator > 0)
        & np.isfinite(denominator)
        & np.isfinite(numerator)
        & (bases.cumsum(axis=0)[:-1] == 1)
    )
    return pd.DataFrame(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=known), index=idx, columns=columns)


def _accumulate_intervals(values: np.ndarray, start: np.ndarray, stop: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Sum each value on its half-open grid interval using start/stop differences."""
    differences = np.zeros(shape)
    np.add.at(differences, (start, columns), values)
    np.add.at(differences, (stop, columns), -values)
    return differences.cumsum(axis=0)[:-1]
