"""Cumulative direct-account share ratios with independent disclosure clocks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.quotes import lookup_asof
from src.data_aggregate.utils.institutionals.insider_quality import FLOW_SIGN, common_stock_mask, security_class

MICRO = 1_000_000

# Holdings are unknown until a ticker has this many days of balance rows: before that, accounts that have not yet reported are missing.
HOLDINGS_WARMUP_DAYS = 365


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
    other_flows: pd.DataFrame,
    idx: pd.DatetimeIndex,
    days: int,
    populations: Mapping[str, int | None],
    excluded: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Net shares bought minus sold in the window over the ticker's insider holdings `days` calendar days earlier.

    Holdings are, per ticker, the sum over direct accounts (owner x class) of the latest post-transaction balance public
    on the day, carried forward; `populations` maps each output name to its population code (`None` = all insiders).
    Accounts listed in `excluded` (disputed quantities) are left out of numerator and holdings. A cell is NaN
    without an active supported trade or without positive holdings `days` earlier.
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
    disputed = pd.MultiIndex.from_frame(excluded[["ticker", "owner_cik"]].drop_duplicates()) if not excluded.empty else None
    uncertain = trades.get("quality_uncertain", pd.Series(False, index=trades.index)).fillna(False).astype(bool)
    lots = trades[account_known & ~uncertain].copy()
    if disputed is not None:
        lots = lots[~pd.MultiIndex.from_frame(lots[["ticker", "owner_cik"]]).isin(disputed)]
    columns = pd.Index(sorted(trades["ticker"].unique()), name="ticker")
    balances = _balance_events(trades.loc[account_known], other_flows, disputed)
    shape = (len(idx) + 1, len(columns))
    start = idx.searchsorted(lots["visible_from"].to_numpy(), side="left")
    stop_day = lots["anchor"] + pd.Timedelta(days=days)
    stop_day = stop_day.where(lots["visible_until"].isna() | stop_day.le(lots["visible_until"]), lots["visible_until"])
    stop = idx.searchsorted(stop_day.to_numpy(), side="left")
    lot_col = columns.get_indexer(pd.Index(lots["ticker"]))
    net = lots["shares_n"].where(lots["code"].eq("P"), -lots["shares_n"]).to_numpy(dtype=float)
    asof = idx.searchsorted((idx - pd.Timedelta(days=days)).to_numpy(), side="right") - 1
    first = balances.groupby("ticker")["visible_from"].min().reindex(columns)
    warm = ((idx - pd.Timedelta(days=days)).to_numpy()[:, None] >= (first + pd.Timedelta(days=HOLDINGS_WARMUP_DAYS)).to_numpy()[None, :]).astype(bool)
    out: dict[str, pd.DataFrame] = {}
    for name, code in populations.items():
        in_pop = np.ones(len(lots), dtype=bool) if code is None else lots["population"].eq(code).to_numpy()
        numerator = _accumulate_intervals(net[in_pop], start[in_pop], stop[in_pop], lot_col[in_pop], shape)
        active = _accumulate_intervals(np.ones(int(in_pop.sum())), start[in_pop], stop[in_pop], lot_col[in_pop], shape)
        holdings = _holdings(balances, code, idx, columns)
        lagged = np.where((asof >= 0)[:, None], holdings[np.maximum(asof, 0)], np.nan)
        known = (lagged > 0) & (active > 0) & warm
        out[name] = pd.DataFrame(np.divide(numerator, lagged, out=np.full_like(numerator, np.nan), where=known), index=idx, columns=columns)
    return out


def _balance_events(trades: pd.DataFrame, other_flows: pd.DataFrame, disputed: pd.MultiIndex | None) -> pd.DataFrame:
    """Each direct account's balance-setting rows, one per account, transaction day and publication day.

    Rows of one account, transaction day and publication day reduce to the row whose balance no sibling row
    starts from (the end of that day's chain), the highest accession breaking ties.
    """
    keep = ["ticker", "owner_cik", "klass", "order_day", "visible_from", "after", "delta", "population", "accession_number"]
    parts = [_balance_rows(trades, trades["transaction_day"])]
    if not other_flows.empty:
        flows = other_flows[
            other_flows["security_type"].isin(["nonderiv", "nonderiv_holding"])
            & other_flows["direct_indirect"].eq("D")
            & other_flows["owner_cik"].ne("")
            & pd.to_numeric(other_flows["n_reporting_owners"], errors="coerce").eq(1)
            & other_flows["security_title"].fillna("").str.strip().ne("")
            & common_stock_mask(other_flows["security_title"])
        ]
        parts.append(_balance_rows(flows, flows["transaction_day"].fillna(flows["day"])))
    df_rows = pd.concat([part[keep] for part in parts], ignore_index=True)
    df_rows = df_rows[np.isfinite(df_rows["after"]) & df_rows["after"].ge(0) & df_rows["order_day"].notna()]
    if disputed is not None:
        df_rows = df_rows[~pd.MultiIndex.from_frame(df_rows[["ticker", "owner_cik"]]).isin(disputed)]
    group = ["ticker", "owner_cik", "klass", "order_day", "visible_from"]
    gid = df_rows.groupby(group, sort=False).ngroup().to_numpy()
    has_delta = df_rows["delta"].notna().to_numpy()
    before = pd.MultiIndex.from_arrays([gid[has_delta], np.round((df_rows["after"] - df_rows["delta"]).to_numpy()[has_delta], 6)])
    ends = pd.MultiIndex.from_arrays([gid, np.round(df_rows["after"].to_numpy(), 6)])
    df_rows = df_rows[~ends.isin(before)]
    df_rows = df_rows.sort_values([*group, "accession_number"]).drop_duplicates(group, keep="last")
    return df_rows.sort_values(["ticker", "owner_cik", "klass", "visible_from", "order_day"]).reset_index(drop=True)


def _balance_rows(df_rows: pd.DataFrame, order_day: pd.Series) -> pd.DataFrame:
    """Rows reduced to account, ordering day, publication day, post-transaction balance and signed change (NaN if undeclared)."""
    sign = df_rows["code"].fillna("").map({"P": 1, "S": -1, **FLOW_SIGN}).astype(float)
    return pd.DataFrame(
        {
            "ticker": df_rows["ticker"],
            "owner_cik": df_rows["owner_cik"],
            "klass": security_class(df_rows["security_title"]),
            "order_day": order_day,
            "visible_from": df_rows["visible_from"],
            "after": pd.to_numeric(df_rows["shares_owned_after"], errors="coerce"),
            "delta": df_rows["shares_n"] * sign,
            "population": df_rows["population"],
            "accession_number": df_rows.get("accession_number", pd.Series("", index=df_rows.index)).fillna("").astype(str),
        }
    )


def _holdings(balances: pd.DataFrame, code: int | None, idx: pd.DatetimeIndex, columns: pd.Index) -> np.ndarray:
    """Sum of the accounts' latest public balances per grid day and ticker; `code` keeps rows of that population."""
    shape = (len(idx) + 1, len(columns))
    if balances.empty:
        return np.zeros(shape)[:-1]
    account = [balances["ticker"], balances["owner_cik"], balances["klass"]]
    key = balances["order_day"].astype("int64")
    wins = key.ge(key.groupby(account).cummax())
    df_win = balances[wins]
    value = df_win["after"] if code is None else df_win["after"].where(df_win["population"].eq(code), 0.0)
    step = value - value.groupby([df_win["ticker"], df_win["owner_cik"], df_win["klass"]]).shift(1).fillna(0.0)
    start = idx.searchsorted(df_win["visible_from"].to_numpy(), side="left")
    col = columns.get_indexer(pd.Index(df_win["ticker"]))
    inside = (start < len(idx)) & (col >= 0)
    scaled = np.rint(step.to_numpy(dtype=float)[inside] * MICRO).astype(np.int64)
    differences = np.zeros(shape, dtype=np.int64)
    np.add.at(differences, (start[inside], col[inside]), scaled)
    return differences.cumsum(axis=0)[:-1] / MICRO


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
