"""Cumulative direct-account share ratios with independent disclosure clocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.insider_quality import security_class
from src.utils.price_adjustments import ShareBasis


def position_units(trades: pd.DataFrame, frames: PriceFrames, quotes: pd.DataFrame | None, share_basis: ShareBasis | None = None) -> pd.DataFrame:
    """Normalize quantities and holdings together using genuine splits, never level actions.

    A single economic day needs no cross-day conversion. Multi-day balances require a
    known common unit: the as-traded and adjusted price legs must share their quote date.
    """
    rows = trades.copy()
    rows["klass"] = security_class(rows["security_title"])
    factor = pd.Series(np.nan, index=rows.index)
    if quotes is not None and not quotes.empty and frames.close_split is not None:
        basis = frames.close_split
        if frames.level_factor is not None:
            basis = basis * frames.level_factor.reindex_like(basis)
        factors = quotes / basis.reindex_like(quotes).where(basis.reindex_like(quotes).gt(0))
        row = factors.index.get_indexer(pd.DatetimeIndex(rows["transaction_day"]))
        col = factors.columns.get_indexer(rows["ticker"])
        known = (row >= 0) & (col >= 0)
        factor.iloc[known] = factors.to_numpy()[row[known], col[known]]
        factor = factor.where(rows["klass"].eq("COMMON"))
    if share_basis is not None:
        configured = rows["ticker"].isin(share_basis.policies)
        declared = share_basis.declared_units(rows["ticker"], rows["transaction_day"], rows["security_title"])
        actual = share_basis.factor(rows["ticker"], rows["transaction_day"], "actual")
        factor.loc[configured] = (actual * declared).loc[configured].where(rows.loc[configured, "klass"].eq("COMMON"))
    rows["units_known"] = factor.gt(0) & np.isfinite(factor)
    for column in ("shares_n", "shares_owned_after"):
        rows[column] = rows[column] * factor.where(rows["units_known"], 1.0)
    return rows


def cumulative_positions(
    trades: pd.DataFrame, positions: pd.DataFrame, other_flows: pd.DataFrame, idx: pd.DatetimeIndex, days: int
) -> dict[str, pd.DataFrame]:
    """Pool window shares over one starting holding per supported active account.

    Each lot retains its own original disclosure expiry. The denominator is the latest
    included economic-day terminal minus active net shares, after balance-chain checks.
    Silent executives are absent; unsupported accounts are excluded from both legs.
    """
    if trades.empty or positions.empty:
        return {}
    joint = pd.to_numeric(trades.get("n_reporting_owners", pd.Series(1, index=trades.index)), errors="coerce").fillna(1).gt(1)
    rows = trades[trades["direct_indirect"].eq("D") & trades["owner_cik"].ne("") & trades["transaction_day"].notna() & ~joint].copy()
    rows["stop"] = rows["anchor"] + pd.Timedelta(days=days)
    rows["stop"] = rows["stop"].where(rows["visible_until"].isna() | rows["stop"].le(rows["visible_until"]), rows["visible_until"])
    rows["delta"] = rows["shares_n"].where(rows["code"].eq("P"), -rows["shares_n"])
    keys = ["ticker", "owner_cik", "klass"]
    day_groups = dict(tuple(positions.groupby(keys, sort=False)))
    flows = (
        {} if other_flows.empty else dict(tuple(other_flows.dropna(subset=["transaction_day", "day"]).groupby(["ticker", "owner_cik"], sort=False)))
    )
    intervals = []
    for key, group in rows.groupby(keys, sort=False):
        account_days = day_groups.get(key)
        if account_days is None:
            continue
        blocked = flows.get(key[:2], pd.DataFrame(columns=["transaction_day", "day"]))
        intervals.extend(_account_intervals(key, group, account_days, blocked, idx))
    columns = pd.Index(sorted(rows["ticker"].unique()), name="ticker")
    unknown = pd.DataFrame(np.nan, index=idx, columns=columns)
    out = {}
    if rows["code"].eq("P").any():
        out["ic_insider_purchase_pct_prior"] = unknown.copy()
    if rows["is_management"].any():
        out["ic_insider_management_position_change_60d"] = unknown.copy()
    if intervals:
        events = pd.DataFrame(intervals)
        if "ic_insider_purchase_pct_prior" in out:
            out["ic_insider_purchase_pct_prior"] = _pool(events[events["buy"].gt(0)], idx, "buy").reindex(columns=columns)
        if "ic_insider_management_position_change_60d" in out:
            out["ic_insider_management_position_change_60d"] = _pool(events[events["management"]], idx, "net").reindex(columns=columns)
    return out


def _account_intervals(
    key: tuple[str, str, str], lots: pd.DataFrame, positions: pd.DataFrame, flows: pd.DataFrame, idx: pd.DatetimeIndex
) -> list[dict[str, object]]:
    """Sweep public version/expiry boundaries instead of allocating account-by-day frames."""
    time_cols = ["visible_from", "stop", "visible_until", "transaction_day"]
    lot = {column: lots[column].to_numpy(dtype="datetime64[ns]") for column in time_cols}
    pos = {column: positions[column].to_numpy(dtype="datetime64[ns]") for column in ("visible_from", "visible_until", "transaction_day")}
    flow_day = flows["transaction_day"].to_numpy(dtype="datetime64[ns]")
    flow_public = flows["day"].to_numpy(dtype="datetime64[ns]")
    points = np.unique(
        np.concatenate(
            [
                lot["visible_from"],
                lot["stop"],
                pos["visible_from"],
                pos["visible_until"],
                flow_public,
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
        public = (lot["visible_from"] <= point) & (np.isnat(lot["visible_until"]) | (lot["visible_until"] > point))
        span = public & (lot["transaction_day"] >= first) & (lot["transaction_day"] <= last)
        visible = (pos["visible_from"] <= point) & (np.isnat(pos["visible_until"]) | (pos["visible_until"] > point))
        visible &= (pos["transaction_day"] >= first) & (pos["transaction_day"] <= last)
        ordered = np.flatnonzero(visible)[np.argsort(pos["transaction_day"][visible])]
        # Expired trades interleaved between active economic days make a literal start ambiguous.
        blocked = (flow_public <= point) & (flow_day >= first) & (flow_day <= last)
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
        if not np.isfinite(prior) or prior <= 0 or not np.isfinite(delta[active]).all():
            continue
        output.append(
            {
                "ticker": key[0],
                "klass": key[2],
                "start": point,
                "stop": stop,
                "net": net,
                "buy": float(delta[active & purchase].sum()),
                "prior": prior,
                "unit_basis": "NORMALIZED" if units[active].all() else np.datetime_as_string(first, unit="D"),
                "management": bool(manager[active].any() and (manager[active] | blank_officer[active]).all()),
            }
        )
    return output


def _pool(events: pd.DataFrame, idx: pd.DatetimeIndex, value: str) -> pd.DataFrame:
    """Add account totals once; require one compatible share class and unit basis."""
    if events.empty:
        return pd.DataFrame(index=idx)
    columns = pd.Index(sorted(events["ticker"].unique()), name="ticker")
    col = columns.get_indexer(events["ticker"])
    start = idx.searchsorted(events["start"].to_numpy(), side="left")
    stop = idx.searchsorted(events["stop"].to_numpy(), side="left")
    shape = (len(idx) + 1, len(columns))

    def accumulated(values: np.ndarray) -> np.ndarray:
        differences = np.zeros(shape)
        np.add.at(differences, (start, col), values)
        np.add.at(differences, (stop, col), -values)
        return differences.cumsum(axis=0)[:-1]

    numerator = accumulated(events[value].to_numpy(dtype=float))
    denominator = accumulated(events["prior"].to_numpy(dtype=float))
    count = accumulated(np.ones(len(events)))
    bases = np.zeros(shape)
    # Union each basis's intervals before counting; never allocate a panel per economic date.
    for (ticker, _, _), group in events.assign(_start=start, _stop=stop).groupby(["ticker", "klass", "unit_basis"], sort=False):
        ordered = group.sort_values("_start")
        begins = ordered["_start"].to_numpy()
        ends = np.maximum.accumulate(ordered["_stop"].to_numpy())
        segments = np.r_[0, np.flatnonzero(begins[1:] > ends[:-1]) + 1]
        last = np.r_[segments[1:] - 1, len(ends) - 1]
        column = columns.get_loc(ticker)
        np.add.at(bases, (begins[segments], column), 1)
        np.add.at(bases, (ends[last], column), -1)
    known = (count > 0) & (denominator > 0) & (bases.cumsum(axis=0)[:-1] == 1)
    return pd.DataFrame(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=known), index=idx, columns=columns)
