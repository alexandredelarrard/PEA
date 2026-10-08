"""Traded-security check: does each canonical `security_master` line trade at the ticker's Yahoo price?

Per line, every FTD price inside the line's window is compared with Yahoo's raw close of the prior trading day
(`prices.close_split` times the `prices_splits` ratios dated after that day). A line with too few matched days
within the tolerance is a warning item, one per ticker; a line with too few matched days, or no Yahoo price, is
counted as unverified. Secondary classes are not judged: `prices_splits` holds no splits for their Yahoo symbols.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils.identity_flags import FLAG_COLUMNS
from src.utils.string import normalise_ticker, pad_cik

KIND = "traded_security_mismatch"
ROLES = ("canonical_current", "canonical_predecessor")
TOLERANCE = 0.03
MIN_SHARE = 0.90
MIN_DAYS = 20
CHUNK = 50
_CONFIG = "configs/sec/security_master_manual.json"
_LINE = ["ticker", "cusip", "lineage_role", "symbol"]
_MASTER_COLUMNS = ["canonical_company", "issuer_cik", "cusip", "lineage_role", "valid_from", "valid_to"]
_FAR = pd.Timestamp("2262-01-01")
STAT_COLUMNS = (*_LINE, "issuer_cik", "n_ftd", "n", "share", "median", "d0", "d1")


def _ts(values: pd.Series) -> pd.Series:
    """Datetimes at one resolution (a DATE column reads back as `datetime.date`)."""
    return pd.to_datetime(values).astype("datetime64[ns]")


def _lines(master: pd.DataFrame, scope: Sequence[str]) -> pd.DataFrame:
    """The scope's canonical rows, each with the Yahoo symbol it is priced under (its ticker)."""
    rows = master[master["lineage_role"].isin(ROLES) & master["cusip"].notna()].copy()
    rows["ticker"] = rows["canonical_company"].map(normalise_ticker)
    rows = rows[rows["ticker"].isin(set(scope))]
    rows["symbol"] = rows["ticker"]
    rows["issuer_cik"] = rows["issuer_cik"].map(lambda c: pad_cik(c) if pd.notna(c) and str(c).strip() else "")
    rows["valid_from"] = _ts(rows["valid_from"])
    rows["valid_to"] = _ts(rows["valid_to"]).fillna(_FAR)
    return rows.reset_index(drop=True)


def _split_factor(days: pd.DataFrame, splits: pd.DataFrame) -> pd.Series:
    """Per row, the product of its symbol's split ratios dated strictly after `pdate` (1 when none)."""
    factor = pd.Series(1.0, index=days.index)
    for symbol, group in splits.groupby("symbol", sort=False):
        ordered = group.sort_values("date")
        logs = np.log(ordered["ratio"].to_numpy(dtype=float))
        after = np.concatenate([np.cumsum(logs[::-1])[::-1], [0.0]])
        mine = days["symbol"].eq(symbol) & days["pdate"].notna()
        if mine.any():
            at = np.searchsorted(ordered["date"].to_numpy(), days.loc[mine, "pdate"].to_numpy(), side="right")
            factor[mine] = np.exp(after[at])
    return factor


def line_stats(lines: pd.DataFrame, ftd: pd.DataFrame, prices: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """`STAT_COLUMNS` per line: FTD days in its windows, matched days, share within the tolerance, median ratio, span."""
    ftd = ftd.assign(
        date=_ts(ftd["date"]), trade_date=_ts(ftd["trade_date"]).fillna(_ts(ftd["date"])), price=pd.to_numeric(ftd["price"], errors="coerce")
    )
    ftd = ftd[ftd["price"].gt(0)]
    days = lines.merge(ftd, on="cusip")
    days = days[days["trade_date"].ge(days["valid_from"]) & days["trade_date"].lt(days["valid_to"])]
    days = days.drop_duplicates([*_LINE, "date"]).sort_values("date", kind="mergesort").reset_index(drop=True)
    days["symbol"] = days["symbol"].astype(str)  # one key dtype on both sides, also when a read came back empty
    quotes = prices.assign(
        pdate=_ts(prices["date"]), symbol=prices["ticker"].map(normalise_ticker).astype(str), close_split=pd.to_numeric(prices["close_split"])
    )
    quotes = quotes[quotes["close_split"].notna()].sort_values("pdate", kind="mergesort")[["symbol", "pdate", "close_split"]]
    days = pd.merge_asof(days, quotes, left_on="date", right_on="pdate", by="symbol", allow_exact_matches=False, direction="backward")
    valid = splits[pd.to_numeric(splits["ratio"], errors="coerce").gt(0)]
    valid = valid.assign(date=_ts(valid["date"]), symbol=valid["ticker"].map(normalise_ticker), ratio=pd.to_numeric(valid["ratio"]))
    days["raw"] = days["close_split"] * _split_factor(days, valid)
    days["ratio"] = days["price"] / days["raw"].where(days["raw"].gt(0))
    days["inside"] = (days["ratio"] - 1).abs().le(TOLERANCE)
    matched = days[days["ratio"].notna()]
    keys = lines[[*_LINE, "issuer_cik"]].drop_duplicates(_LINE)
    out = keys.merge(days.groupby(_LINE).size().rename("n_ftd").reset_index(), on=_LINE, how="left")
    stats = matched.groupby(_LINE).agg(
        n=("ratio", "size"), share=("inside", "mean"), median=("ratio", "median"), d0=("date", "min"), d1=("date", "max")
    )
    out = out.merge(stats.reset_index(), on=_LINE, how="left")
    out[["n_ftd", "n"]] = out[["n_ftd", "n"]].fillna(0).astype(int)
    return out[list(STAT_COLUMNS)]


def _line_text(row: Any) -> str:
    return f"{row.ticker} {row.cusip} {row.lineage_role} ({row.n} matched of {row.n_ftd} FTD days)"


def mismatch_items(stats: pd.DataFrame) -> tuple[list[dict[str, object]], list[str]]:
    """One warning item per ticker with a flagged line (the worst line in the evidence), and the unverified lines."""
    checked = stats[stats["n"].ge(MIN_DAYS)]
    flagged = checked[checked["share"].lt(MIN_SHARE)].assign(_gap=lambda f: np.log(f["median"]).abs())
    items: list[dict[str, object]] = []
    for ticker, group in flagged.groupby("ticker", sort=True):
        worst = group.sort_values(["share", "_gap"], ascending=[True, False], kind="mergesort").iloc[0]
        others = sorted(set(group["cusip"]) - {worst["cusip"]})
        evidence = (
            f"{worst['cusip']} ({worst['lineage_role']}, {worst['d0'].date()}..{worst['d1'].date()}): {worst['share']:.0%} of {worst['n']} "
            f"matched days within +-{TOLERANCE:.0%} of Yahoo's raw prior close, median ratio {worst['median']:.3f}; "
            f"{len(group)} flagged line(s)" + (f" (also {', '.join(others)})" if others else "")
        )
        items.append(
            {
                "kind": KIND,
                "action": True,
                "ticker": str(ticker),
                "ciks": str(worst["issuer_cik"]),
                "evidence": evidence,
                "suggested_action": "the FTD line trades at another price than the ticker's Yahoo series: check which security its filings and prices follow",
                "config_file": _CONFIG,
            }
        )
    unverified = stats[stats["n"].lt(MIN_DAYS)].sort_values(_LINE, kind="mergesort")
    return items, [_line_text(r) for r in unverified.itertuples(index=False)]


def _load(context: Context, table: Any, columns: list[str], where: dict[str, Any]) -> pd.DataFrame:
    rows = context.store.load(table, columns=columns, where=where, optional=True)
    return rows if rows is not None else pd.DataFrame(columns=columns)


def traded_security_stats(context: Context, scope: Sequence[str]) -> pd.DataFrame:
    """`line_stats` over the scope, read per chunk of tickers through the store."""
    needed = (Tables.security_master, Tables.sec_fails_to_deliver_security)
    if not all(context.store.exists(t) for t in needed):
        return pd.DataFrame(columns=list(STAT_COLUMNS))
    master = _load(context, Tables.security_master, _MASTER_COLUMNS, {"lineage_role": list(ROLES)})
    lines = _lines(master, scope)
    tickers = sorted(set(lines["ticker"]))
    parts = []
    for start in range(0, len(tickers), CHUNK):
        chunk = lines[lines["ticker"].isin(tickers[start : start + CHUNK])]
        symbols = sorted(set(chunk["symbol"]))
        ftd = _load(context, Tables.sec_fails_to_deliver_security, ["cusip", "date", "trade_date", "price"], {"cusip": sorted(set(chunk["cusip"]))})
        prices = _load(context, Tables.prices, ["ticker", "date", "close_split"], {"ticker": symbols})
        splits = _load(context, Tables.prices_splits, ["ticker", "date", "ratio"], {"ticker": symbols})
        parts.append(line_stats(chunk, ftd, prices, splits))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(STAT_COLUMNS))


def traded_security_flags(context: Context, scope: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The `traded_security_mismatch` flag items (`FLAG_COLUMNS`) and the rule's metrics."""
    stats = traded_security_stats(context, scope)
    items, unverified = mismatch_items(stats)
    metrics = {
        "traded_security_lines": len(stats),
        "traded_security_mismatch": [str(i["ticker"]) for i in items],
        "traded_security_unverified": len(unverified),
        "traded_security_unverified_lines": unverified,
    }
    return pd.DataFrame(items, columns=list(FLAG_COLUMNS)), metrics
