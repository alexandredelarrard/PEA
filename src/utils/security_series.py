"""The FTD line rules, one copy: the settlement-cycle trade date, the FTD `universe_cusip` lines keyed for stamping, each
stored line's security on its trade date, and the ticker series built from the stored lines.

`sec_fails_to_deliver_security` stores the FTD lines as filed. The institutionals cube maps them to companies here: per
(ticker, settlement date) the sum over the canonical and secondary-class lines of quantity x conversion ratio, a class
quantity rebased trade -> settlement through the shared share basis, and of quantity x the file's PRICE (NULL when a
summed line has no price); a served date with no line of a supported company is an explicit zero, unless an unresolved
line could be the company's. The FTD ingest and the identity step read `trade_dates`; the short-volume tape reads
`summed_lines`.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS, SHARADAR_ACTION_SPLIT
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.price.adjustments import ShareBasis, resolve_share_events
from src.utils.price.basis import prepare_quote_anchors
from src.utils.string import normalise_ticker, yahoo_symbol

#: The `source` of a line read off the SEC fails-to-deliver files.
SOURCE_FTD = "ftd"
#: First trade date of each settlement cycle (T+2 from 2017-09-05, T+1 from 2024-05-28).
T2_FROM = pd.Timestamp("2017-09-05")
T1_FROM = pd.Timestamp("2024-05-28")
#: The `universe_cusip` columns read for stamping FTD lines.
FTD_MASTER_COLUMNS = [
    "security_id",
    "canonical_company",
    "cusip",
    "source_symbol",
    "market_symbol",
    "security_class",
    "conversion_ratio",
    "lineage_role",
    "valid_from",
    "valid_to",
    "scope_changed_at",
]
#: Roles whose lines are summed into a company's series.
SUMMED_ROLES = (*CANONICAL_ROLES, SECONDARY_CLASS)
#: The FTD ticker series: one row per (ticker, settlement date).
SERIES_COLUMNS = ("ticker", "date", "fails_quantity", "fails_value", "period")
#: The stored line columns the series reads.
LINE_COLUMNS = ("date", "trade_date", "cusip", "source_symbol", "fails_quantity", "fails_value", "period")

_FAR = pd.Timestamp("2262-01-01")
_HIT_COLUMNS = ["canonical_company", "lineage_role", "security_class", "conversion_ratio", "market_symbol"]


def trade_dates(settlement: pd.Series) -> pd.Series:
    """Trade dates of FTD settlement dates by the cycle in force (T+3, T+2 from 2017-09-05, T+1 from 2024-05-28)."""
    days = pd.to_datetime(settlement)
    bday = pd.offsets.BDay
    t1, t2, t3 = days - bday(1), days - bday(2), days - bday(3)
    out = t3.where(t2 < T2_FROM, t2).where(t1 < T1_FROM, t1)
    return out.where(days.notna())


def prepare_master(master: pd.DataFrame) -> pd.DataFrame:
    """The FTD rows of `universe_cusip`, keyed for stamping: upper-case CUSIP and symbol, open ends at a far date."""
    rows = master[master["source"].eq(SOURCE_FTD)] if "source" in master.columns else master
    out = rows.reindex(columns=FTD_MASTER_COLUMNS).copy()
    out["market_symbol"] = out["market_symbol"].fillna(out["source_symbol"]).map(yahoo_symbol)
    out["cusip"] = out["cusip"].astype(str).str.strip().str.upper()
    out["source_symbol"] = out["source_symbol"].fillna("").map(normalise_ticker)
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"]).fillna(_FAR)
    out["conversion_ratio"] = pd.to_numeric(out["conversion_ratio"], errors="coerce").astype("float64")
    company = out["canonical_company"].astype(object)
    out["canonical_company"] = company.where(company.notna() & company.astype(str).ne(""), None)
    out["scope_changed_at"] = pd.to_datetime(out["scope_changed_at"], errors="coerce")
    return out.reset_index(drop=True)


def nullable(values: pd.Series) -> pd.Series:
    """`values` as objects with missing values as None (the store's NULL)."""
    return values.astype(object).where(values.notna(), None)


def summed_lines(stamped: pd.DataFrame, tickers: Collection[str] | None) -> pd.DataFrame:
    """The stamped lines summed into a company's series: a summed role and a ticker, limited to `tickers` when given."""
    summed = stamped[stamped["lineage_role"].isin(SUMMED_ROLES) & stamped["ticker"].notna()]
    return summed if tickers is None else summed[summed["ticker"].isin(set(tickers))]


def _covering(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["trade_date"] >= frame["valid_from"]) & (frame["trade_date"] < frame["valid_to"])]


def stamp_lines(lines: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """`lines` with `security_id`, `ticker`, `lineage_role`, `security_class`, `conversion_ratio` and `market_symbol`.

    The answer is the master row of the line's (CUSIP, symbol) covering its trade date, else the one answer every
    row of its CUSIP covering that date agrees on. A CUSIP the master lacks keeps a NULL `security_id` (out of
    scope); a known CUSIP with no answer on that date keeps a NULL role and is never summed.
    """
    out = lines.reset_index(drop=True).copy()
    trade = pd.to_datetime(out["trade_date"]) if "trade_date" in out.columns else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    out["trade_date"] = trade.fillna(trade_dates(out["date"]))
    keys = pd.DataFrame(
        {
            "_row": out.index,
            "cusip": out["cusip"].astype(str).str.strip().str.upper(),
            "source_symbol": out["source_symbol"].fillna("").map(normalise_ticker),
            "trade_date": out["trade_date"],
        }
    )
    exact = _covering(keys.merge(master, on=["cusip", "source_symbol"]))
    exact = exact.sort_values(["_row", "valid_from"], ascending=[True, False]).drop_duplicates("_row")
    rest = keys[~keys["_row"].isin(exact["_row"])].drop(columns="source_symbol")
    loose = _covering(rest.merge(master.drop(columns="source_symbol"), on="cusip")).drop_duplicates(["_row", *_HIT_COLUMNS])
    agreed = loose[~loose["_row"].duplicated(keep=False)]
    hits = pd.concat([exact[["_row", *_HIT_COLUMNS]], agreed[["_row", *_HIT_COLUMNS]]], ignore_index=True).set_index("_row")
    out["security_id"] = nullable(("C" + keys["cusip"]).where(keys["cusip"].isin(set(master["cusip"]))))
    out["ticker"] = nullable(hits["canonical_company"].reindex(out.index))
    out["lineage_role"] = nullable(hits["lineage_role"].reindex(out.index))
    out["security_class"] = nullable(hits["security_class"].reindex(out.index))
    out["conversion_ratio"] = hits["conversion_ratio"].reindex(out.index).astype("float64")
    out["market_symbol"] = hits["market_symbol"].reindex(out.index)
    return out


def ticker_rows(stamped: pd.DataFrame, basis: ShareBasis) -> pd.DataFrame:
    """Per (ticker, settlement date) over the canonical and secondary-class lines, the sum of quantity x conversion ratio
    in settlement units and of quantity x PRICE; the dollars are NULL when a summed line has no price."""
    summed = summed_lines(stamped, None)
    if summed.empty:
        return pd.DataFrame({column: pd.Series(dtype="datetime64[ns]" if column == "date" else object) for column in SERIES_COLUMNS})
    quantity = _settlement_quantity(summed, basis)
    if not np.isfinite(quantity).all():
        raise ValueError("FTD settlement quantity basis is unavailable or nonfinite")
    value = pd.to_numeric(summed["fails_value"], errors="coerce")
    work = pd.DataFrame(
        {
            "ticker": summed["ticker"].astype(str),
            "date": pd.to_datetime(summed["date"]).dt.normalize(),
            "fails_quantity": quantity,
            "fails_value": value,
            "unpriced": value.isna() & quantity.notna(),
            "period": summed["period"].astype(str),
        }
    )
    grouped = work.groupby(["ticker", "date"], as_index=False, sort=True).agg(
        fails_quantity=("fails_quantity", "sum"), fails_value=("fails_value", "sum"), unpriced=("unpriced", "any"), period=("period", "min")
    )
    grouped["fails_value"] = grouped["fails_value"].where(~grouped["unpriced"])
    if not np.isfinite(grouped["fails_quantity"]).all() or np.isinf(grouped["fails_value"]).any():
        raise ValueError("FTD class conversion or aggregate quantity/value is nonfinite")
    return grouped[list(SERIES_COLUMNS)]


def _settlement_quantity(summed: pd.DataFrame, basis: ShareBasis) -> pd.Series:
    """Raw outstanding quantities are settlement units; identity/class ratios use the inferred trade date."""
    quantity = pd.to_numeric(summed["fails_quantity"], errors="coerce")
    ratio = pd.to_numeric(summed["conversion_ratio"], errors="coerce")
    if not (np.isfinite(quantity) & quantity.ge(0) & np.isfinite(ratio) & ratio.gt(0)).all():
        raise ValueError("FTD quantities and class ratios require finite valid source units")
    symbols = summed["market_symbol"].map(yahoo_symbol)
    canonical = summed["ticker"]
    trade = pd.to_datetime(summed["trade_date"])
    settlement = pd.to_datetime(summed["date"])
    class_quantity = basis.rebase(quantity, symbols, settlement, trade)
    converted = basis.rebase(class_quantity * ratio, canonical, trade, settlement)
    same_class = symbols.eq(canonical) | summed["lineage_role"].isin(CANONICAL_ROLES)
    return converted.where(~same_class, quantity * ratio)


def uncertain_dates(stamped: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """`[ticker, date]` an unresolved stamped line could belong to: a master row of the company carrying the line's CUSIP
    or its symbol (X93). The series blocks these dates; a line linked to a company only by a lineage symbol is not the
    company's."""
    # the unresolved lines, keyed as the master is
    unresolved = stamped.loc[stamped["lineage_role"].isna() | stamped["ticker"].isna()]
    keys = pd.DataFrame(
        {
            "cusip": unresolved["cusip"].astype(str).str.strip().str.upper(),
            "source_symbol": unresolved["source_symbol"].fillna("").map(normalise_ticker),
            "date": pd.to_datetime(unresolved["date"]).dt.normalize(),
        }
    )

    # the companies whose master rows carry the CUSIP or the symbol
    owners = master.dropna(subset=["canonical_company"])
    by_cusip = keys.merge(owners[["cusip", "canonical_company"]].drop_duplicates(), on="cusip")
    named = owners.loc[owners["source_symbol"].ne(""), ["source_symbol", "canonical_company"]].drop_duplicates()
    by_symbol = keys.loc[keys["source_symbol"].ne("")].merge(named, on="source_symbol")
    touched = pd.concat([by_cusip, by_symbol], ignore_index=True).rename(columns={"canonical_company": "ticker"})
    return touched[["ticker", "date"]].drop_duplicates().reset_index(drop=True)


def _supported_dates(dates: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """`[ticker, date, period, conversion_ratio]`: each served date x each summed master row covering its trade date."""
    rows = master.loc[master["canonical_company"].notna() & master["lineage_role"].isin(SUMMED_ROLES)].reset_index(drop=True)
    days = dates.sort_values("trade_date", kind="mergesort", ignore_index=True)
    trade = days["trade_date"].to_numpy(dtype="datetime64[ns]")
    lo = np.searchsorted(trade, rows["valid_from"].to_numpy(dtype="datetime64[ns]"), side="left")
    hi = np.searchsorted(trade, rows["valid_to"].to_numpy(dtype="datetime64[ns]"), side="left")
    counts = np.maximum(hi - lo, 0)
    which = np.repeat(np.arange(len(rows)), counts)
    at = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi, strict=True) if b > a]) if counts.sum() else np.array([], dtype=int)
    return pd.DataFrame(
        {
            "ticker": rows["canonical_company"].to_numpy(dtype=object)[which],
            "date": days["date"].to_numpy(dtype="datetime64[ns]")[at],
            "period": days["period"].to_numpy(dtype=object)[at],
            "conversion_ratio": rows["conversion_ratio"].to_numpy(dtype="float64")[which],
        }
    )


def fails_series(lines: pd.DataFrame, master: pd.DataFrame, basis: ShareBasis) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(series, coverage)` of the stored FTD lines: the (ticker, date) series over the served dates of each supported
    company (an explicit zero where no line of it was filed), and the served `(date, period)` calendar.

    A served date is a date of a stored line (the stored scope keeps a line on every served date). Dates an unresolved
    line could belong to, or whose covering line has an invalid ratio, are left out.
    """
    stamped = stamp_lines(lines, master)
    grain = ticker_rows(stamped, basis)
    dates = stamped[["date", "trade_date", "period"]].assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize()).drop_duplicates()
    dates["period"] = dates["period"].astype(str)
    candidates = _supported_dates(dates, master)

    # blocked: an invalid ratio on a covering line, or an unresolved line the company's master rows could own
    invalid = candidates.loc[~np.isfinite(candidates["conversion_ratio"]) | candidates["conversion_ratio"].le(0), ["ticker", "date"]]
    blocked = pd.concat([invalid, uncertain_dates(stamped, master)], ignore_index=True).drop_duplicates()
    candidates = candidates[["ticker", "date", "period"]].drop_duplicates()
    candidates = candidates.merge(blocked.assign(_blocked=True), on=["ticker", "date"], how="left")
    candidates = candidates.loc[candidates.pop("_blocked").isna()]

    # the series: each supported date's sum, else an explicit zero
    series = candidates.merge(grain, on=["ticker", "date", "period"], how="left", indicator=True)
    absent = series.pop("_merge").eq("left_only")
    series.loc[absent, ["fails_quantity", "fails_value"]] = 0.0
    series[["fails_quantity", "fails_value"]] = series[["fails_quantity", "fails_value"]].astype("float64")
    coverage = dates[["date", "period"]].drop_duplicates().sort_values(["date", "period"], ignore_index=True)
    return series.sort_values(["ticker", "date"], ignore_index=True)[list(SERIES_COLUMNS)], coverage


def load_fails_basis(store: DataStore, bugfix: dict, master: pd.DataFrame) -> ShareBasis:
    """The genuine-event share basis of the master's class symbols and canonical companies (the FTD trade -> settlement rebase)."""
    names = sorted(set(master["market_symbol"].dropna().astype(str)) | set(master["canonical_company"].dropna().astype(str)))
    policies = {ticker: policy for ticker, policy in bugfix.get("share_events", {}).items() if ticker in names}
    anchors = [(ticker, anchor) for ticker, policy in policies.items() for anchor in policy["anchors"] if "quoted_price" in anchor]
    actions = store.load(
        Tables.sharadar_actions,
        columns=["ticker", "date", "action", "value"],
        where={"ticker": names, "action": [SHARADAR_ACTION_SPLIT]},
        optional=True,
    )
    splits = store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": names}, optional=True)
    vendor = (
        store.load(
            Tables.sharadar_fundamentals_facts,
            columns=["ticker", "date", "sharesbas", "price"],
            where={"ticker": sorted(policies), "dimension": "ARQ"},
            optional=True,
        )
        if policies
        else None
    )
    quotes = (
        store.load(
            Tables.prices, columns=["ticker", "date", "close_split", "volume"], where={"ticker": sorted({t for t, _ in anchors})}, optional=True
        )
        if anchors
        else None
    )
    if splits is None and store.exists(Tables.prices_splits):
        splits = pd.DataFrame(columns=["ticker", "date", "ratio"])
    if actions is None and store.exists(Tables.sharadar_actions):
        actions = pd.DataFrame(columns=["ticker", "date", "action", "value"])
    checked_quotes = prepare_quote_anchors(quotes, vendor, anchors, bugfix) if anchors else None
    return resolve_share_events(actions, splits, bugfix, shares=vendor, quotes=checked_quotes, tickers=names)
