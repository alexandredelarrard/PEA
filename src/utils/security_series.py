"""The FTD and FINRA short-volume line rules, one copy: the settlement-cycle trade date, the FTD `universe_cusip` lines
keyed for stamping, each stored line's security on its date, and the ticker series built from the stored lines.

`sec_fails_to_deliver_security` stores the FTD lines as filed. The institutionals cube maps them to companies here: per
(ticker, settlement date) the sum over the canonical and secondary-class lines of quantity x conversion ratio, a class
quantity rebased trade -> settlement through the shared share basis, and of quantity x the file's PRICE (NULL when a
summed line has no price); a served date with no line of a supported company is an explicit zero, unless an unresolved
line could be the company's. The FTD ingest and the identity step read `trade_dates`.

`sec_short_volume_security` stores the FINRA RegSHO lines as filed. The cube maps each by its FTD spelling
(`finra_key`) to the `universe_cusip` line covering its date, else, for a symbol FTD never carries, through the lineage
symbol of a window CIK inside its window (`S<cik>:<symbol>`); per (ticker, date) volume x conversion ratio is summed over
the canonical and secondary-class lines, and a served date a company's symbol supports with no line of it is a supported
absence (short volume 0, total unknown), unless an unresolved line could be the company's or a covering ratio is invalid.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_CURRENT, CANONICAL_PREDECESSOR, CANONICAL_ROLES, SECONDARY_CLASS, SHARADAR_ACTION_SPLIT
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.price.adjustments import ShareBasis, resolve_share_events
from src.utils.price.basis import prepare_quote_anchors
from src.utils.string import normalise_ticker, pad_cik_series, squash, yahoo_symbol

#: The `source` of a line read off the SEC fails-to-deliver files.
SOURCE_FTD = "ftd"
#: `universe_cusip` roles never summed: an acquired company's line, a non-equity or other excluded line.
ACQUIRED_CONSTITUENT = "acquired_constituent"
EXCLUDED = "excluded"
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


# --------------------------------------------------------------------------- FINRA RegSHO short volume

#: FINRA lower-case markers: their FTD spelling and the kind of line they mark.
_MARKER_SPELLING = {"p": "PR", "r": "RT", "w": "WI"}
_MARKER_KIND = {"p": "preferred", "r": "right", "w": "when_issued"}
#: Suffixes after a `/`: the kind of line they mark (a single class letter marks a common class).
_SUFFIX_KIND = (("WS", "warrant"), ("CL", "called"), ("U", "unit"))
_CLASS_SUFFIX = re.compile(r"^[A-TV-Z]$")
#: The stored FINRA line columns the short-volume series reads.
SHORT_LINE_COLUMNS = ("date", "source_symbol", "short_volume", "total_volume")
#: The short-volume series: one row per (ticker, date).
SHORT_SERIES_COLUMNS = ("date", "ticker", "short_volume", "total_volume")
#: The lineage-fallback stamps the tenure mask reads.
FALLBACK_COLUMNS = ("date", "ticker", "security_id", "lineage_role", "security_class")
_STAMP_COLUMNS = ["security_id", "ticker", "lineage_role", "security_class", "conversion_ratio"]
#: A window or interval with no start (the lineage writes 1900-01-01 for an open start).
_OPEN = pd.Timestamp("1900-01-01")


def finra_key(symbol: object) -> str:
    """The FTD spelling of a FINRA symbol, the key `universe_cusip` rows are found by (`BACpB` -> `BACPRB`, `BRK/A` -> `BRKA`)."""
    return squash("".join(_MARKER_SPELLING.get(char, char) for char in str(symbol)))


def finra_marker_kind(symbol: object) -> str | None:
    """The non-common kind a FINRA symbol's own markers name (`preferred`, `right`, `when_issued`, `warrant`, `unit`, `called`), or None."""
    text = str(symbol)
    for marker, kind in _MARKER_KIND.items():
        if marker in text:
            return kind
    for part in text.split("/")[1:]:
        for prefix, kind in _SUFFIX_KIND:
            if part.startswith(prefix) and not _CLASS_SUFFIX.match(part):
                return kind
    return None


def _class_letter(symbol: object) -> str | None:
    parts = str(symbol).split("/")
    return parts[1] if len(parts) == 2 and _CLASS_SUFFIX.match(parts[1]) else None


def equity_symbol_rows(lineage: pd.DataFrame) -> pd.DataFrame:
    """The lineage symbol rows a tape may key on: not noise, and not evidenced by the Notes cover page (`dei`) alone, which
    is how a company's debt and preferred listings (ABBV28B, BRK59) enter the lineage."""
    sources = lineage["sources"].fillna("").astype(str).str.lower()
    return lineage[lineage["role"].eq("symbol") & lineage["status"].ne("noise") & sources.ne("dei")]


def _by_symbol(symbols: pd.Series, rule: Callable[[object], object]) -> pd.Series:
    """`rule` applied once per distinct symbol."""
    unique = symbols.unique()
    return symbols.map(dict(zip(unique, map(rule, unique), strict=True)))


def _short_master(lines: pd.DataFrame) -> pd.DataFrame:
    """The FTD `universe_cusip` lines keyed by their squashed symbol, open ends far, newest start first (the order the
    one answer of a date is taken in)."""
    rows = lines[lines["source"].eq(SOURCE_FTD)] if "source" in lines.columns else lines
    company = rows["canonical_company"].astype(object)
    out = pd.DataFrame(
        {
            "key": rows["source_symbol"].map(squash).astype(str),
            "source_symbol": rows["source_symbol"].astype(str),
            "security_id": rows["security_id"].astype(str),
            "ticker": company.where(company.notna() & company.astype(str).ne(""), None),
            "lineage_role": rows["lineage_role"].astype(str),
            "security_class": rows["security_class"].astype(str),
            "conversion_ratio": pd.to_numeric(rows["conversion_ratio"], errors="coerce").astype("float64"),
            "valid_from": pd.to_datetime(rows["valid_from"]),
            "valid_to": pd.to_datetime(rows["valid_to"]).fillna(_FAR),
        }
    )
    return out.sort_values(["valid_from", "security_id", "source_symbol"], ascending=[False, True, True], kind="mergesort", ignore_index=True)


def _lineage_tape(lineage: pd.DataFrame) -> pd.DataFrame:
    """`[key, company_id, ticker, cik, status, start, end, window_from, window_to, current]`: each equity lineage symbol
    interval with each window of its own CIK (a tape reads a symbol inside its CIK's declared window, P21); `current`
    marks a CIK holding the company's open window."""
    rows = lineage.assign(cik=pad_cik_series(lineage["cik"]), company_id=lineage["company_id"].astype(str))
    tickers = rows.dropna(subset=["canonical_ticker"]).drop_duplicates("company_id").set_index("company_id")["canonical_ticker"]
    symbols = equity_symbol_rows(rows)
    tape = pd.DataFrame(
        {
            "key": symbols["symbol"].map(yahoo_symbol).astype(str),
            "company_id": symbols["company_id"],
            "cik": symbols["cik"],
            "status": symbols["status"].astype(str),
            "start": pd.to_datetime(symbols["valid_from"]).fillna(_OPEN),
            "end": pd.to_datetime(symbols["valid_to"]).fillna(_FAR),
        }
    )
    windows = rows.loc[rows["role"].eq("cik_window")]
    windows = pd.DataFrame(
        {
            "company_id": windows["company_id"],
            "cik": windows["cik"],
            "window_from": pd.to_datetime(windows["valid_from"]).fillna(_OPEN),
            "window_to": pd.to_datetime(windows["valid_to"]).fillna(_FAR),
            "current": windows["valid_to"].isna(),
        }
    )
    windows["current"] = windows.groupby(["company_id", "cik"])["current"].transform("any")
    tape = tape.loc[tape["key"].ne("")].merge(windows, on=["company_id", "cik"])
    return tape.assign(ticker=tape["company_id"].map(tickers)).dropna(subset=["ticker"]).reset_index(drop=True)


def _security_hits(pairs: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Per `_row` of `pairs` (`key`, `day`), the one `universe_cusip` line of its key covering its day; several lines all
    summed or acquired keep the summed ones (a CUSIP-change day under one symbol); still several -> no answer."""
    hits = pairs[["_row", "key", "day"]].merge(master, on="key")
    hits = hits.loc[hits["day"].ge(hits["valid_from"]) & hits["day"].lt(hits["valid_to"])]
    summed = hits["lineage_role"].isin(SUMMED_ROLES)
    several = hits.groupby("_row")["security_id"].transform("nunique").gt(1)
    benign = hits["lineage_role"].isin((*SUMMED_ROLES, ACQUIRED_CONSTITUENT)).groupby(hits["_row"]).transform("all")
    hits = hits.loc[~(several & benign) | summed]
    hits = hits.loc[hits.groupby("_row")["security_id"].transform("nunique").eq(1)].drop_duplicates("_row")
    return hits.set_index("_row")[_STAMP_COLUMNS]


def _fallback_hits(pairs: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    """Per `_row` of `pairs` (`symbol`, `day`), the lineage interval of the symbol inside its CIK's window as an
    `S<cik>:<symbol>` security (current role on the open window's CIK, ratio 1); a `conflict` interval or two companies
    on that day -> no answer."""
    work = pairs[["_row", "symbol", "day"]].assign(key=_by_symbol(pairs["symbol"], yahoo_symbol).astype(str)).merge(tape, on="key")
    day = work["day"]
    work = work.loc[day.ge(work["start"]) & day.lt(work["end"]) & day.ge(work["window_from"]) & day.lt(work["window_to"])]
    conflict = work["status"].eq("conflict").groupby(work["_row"]).transform("any")
    blocked = conflict | work.groupby("_row")["company_id"].transform("nunique").ne(1)
    work = work.loc[~blocked].sort_values(["_row", "start", "cik"], ascending=[True, False, False]).drop_duplicates("_row")
    letters = _by_symbol(work["symbol"], _class_letter)
    hits = pd.DataFrame(
        {
            "security_id": [f"S{cik}:{key}" for cik, key in zip(work["cik"], work["key"], strict=True)],
            "ticker": work["ticker"],
            "lineage_role": np.where(work["current"], CANONICAL_CURRENT, CANONICAL_PREDECESSOR),
            "security_class": letters.map(lambda letter: f"class_{letter}" if isinstance(letter, str) else "common"),
            "conversion_ratio": 1.0,
        }
    )
    return hits.set_axis(work["_row"].to_numpy())


def _stamp_short(rows: pd.DataFrame, master: pd.DataFrame, tape: pd.DataFrame, universe: Collection[str]) -> pd.DataFrame:
    """The lines a company's series reads, stamped (`_STAMP_COLUMNS`): a symbol whose FTD key is a line of a `universe`
    company is kept, stamped by the line covering its date (NULL when none does, or when two symbols of the date share
    the key); a symbol FTD never carries is kept only through the lineage fallback; a line marked non-common by its own
    FINRA symbology is never summed."""
    requested = set(universe)
    symbol = rows["source_symbol"].astype(str)
    keys, marked = _by_symbol(symbol, finra_key).astype(str), _by_symbol(symbol, finra_marker_kind)
    clash = symbol.groupby([rows["date"], keys]).transform("nunique").gt(1)
    known = keys.isin(set(master.loc[master["ticker"].isin(requested), "key"]))
    pairs = pd.DataFrame({"_row": rows.index, "key": keys, "symbol": symbol, "day": rows["date"]})
    on_master = _security_hits(pairs.loc[known & ~clash], master)
    fallback = _fallback_hits(pairs.loc[~known & ~clash & marked.isna() & symbol.eq(symbol.str.upper())], tape)
    hits = pd.concat([on_master, fallback])
    hits = hits.loc[hits["ticker"].isin(requested)]
    out = rows.join(hits)
    unsafe = marked.notna() & out["lineage_role"].isin(SUMMED_ROLES)
    out.loc[unsafe, "lineage_role"] = EXCLUDED
    out.loc[unsafe, "security_class"] = marked[unsafe]
    return out.loc[known | out.index.isin(hits.index)]


def _short_sums(stamped: pd.DataFrame, universe: Collection[str]) -> pd.DataFrame:
    """Per (ticker, date) over the canonical and secondary-class lines, the volumes x conversion ratio."""
    summed = summed_lines(stamped, universe)
    ratio = pd.to_numeric(summed["conversion_ratio"], errors="coerce")
    if not (np.isfinite(ratio) & ratio.gt(0)).all():
        raise ValueError("RegSHO needs finite positive class conversion ratios")
    work = summed[["ticker", "date"]].assign(short_volume=summed["short_volume"] * ratio, total_volume=summed["total_volume"] * ratio)
    result = work.groupby(["ticker", "date"], as_index=False, sort=True)[["short_volume", "total_volume"]].sum()
    if not np.isfinite(result[["short_volume", "total_volume"]].to_numpy(dtype=float)).all():
        raise ValueError("RegSHO class-converted aggregate quantities overflowed")
    return result


def _day_spans(spans: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    """`[ticker, date]`: each `[start, stop)` span of `spans` over the sorted served `days`."""
    lo = np.searchsorted(days, spans["start"].to_numpy(dtype="datetime64[ns]"), side="left")
    hi = np.searchsorted(days, spans["stop"].to_numpy(dtype="datetime64[ns]"), side="left")
    counts = np.maximum(hi - lo, 0)
    at = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi, strict=True) if b > a]) if counts.sum() else np.array([], dtype=int)
    return pd.DataFrame({"ticker": np.repeat(spans["ticker"].to_numpy(dtype=object), counts), "date": days[at]})


def _supported_short(universe: Collection[str], days: np.ndarray, master: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    """`[ticker, date]` a summed security supports: the universe tickers and their companies' windowed lineage symbols,
    each by its `universe_cusip` line on the date, else its lineage fallback. Evaluated once per span between the
    symbol's interval bounds, then spread over the served days."""
    # the symbols: the tickers and every equity lineage symbol of their companies overlapping one of its CIK's windows
    overlap = tape["ticker"].isin(set(universe)) & tape["start"].lt(tape["window_to"]) & tape["window_from"].lt(tape["end"])
    symbols = pd.DataFrame({"symbol": sorted(set(universe) | set(tape.loc[overlap, "key"]))})
    symbols = symbols.assign(key=symbols["symbol"].map(finra_key).astype(str), tkey=symbols["symbol"].map(yahoo_symbol).astype(str))

    # the spans: between every bound of the symbol's lines and lineage intervals, inside the served days
    first, after = pd.Timestamp(days[0]), pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
    by_line = symbols.merge(master[["key", "valid_from", "valid_to"]], on="key").melt(["symbol"], ["valid_from", "valid_to"], value_name="at")
    by_tape = symbols.merge(tape.rename(columns={"key": "tkey"}), on="tkey")
    by_tape = by_tape.melt(["symbol"], ["start", "end", "window_from", "window_to"], value_name="at")
    bounds = pd.concat([by_line[["symbol", "at"]], by_tape[["symbol", "at"]], symbols[["symbol"]].assign(at=first)], ignore_index=True)
    bounds = bounds.assign(at=pd.to_datetime(bounds["at"]).clip(first, after)).drop_duplicates().sort_values(["symbol", "at"], ignore_index=True)
    bounds["stop"] = bounds.groupby("symbol")["at"].shift(-1).fillna(after)
    spans = bounds.loc[bounds["at"].lt(bounds["stop"])].merge(symbols, on="symbol").reset_index(drop=True)
    pairs = spans.assign(_row=spans.index, day=spans["at"])

    # each span's answer: the line, else the lineage fallback; a summed role supports its company
    on_master = _security_hits(pairs, master)
    hits = pd.concat([on_master, _fallback_hits(pairs.loc[~pairs["_row"].isin(on_master.index)], tape)])
    hits = hits.loc[hits["lineage_role"].isin(SUMMED_ROLES) & hits["ticker"].notna()]
    covered = spans.loc[hits.index, ["at", "stop"]].rename(columns={"at": "start"}).assign(ticker=hits["ticker"].to_numpy())
    return _day_spans(covered, days)


def _blocked_short(stamped: pd.DataFrame, master: pd.DataFrame, universe: Collection[str], days: np.ndarray) -> pd.DataFrame:
    """`[ticker, date]` with no checked absence: an unresolved unmarked line whose key is any summed line of the company,
    and the dates a summed line of a `universe` company covers with an invalid ratio."""
    unresolved = stamped.loc[stamped["lineage_role"].isna()]
    symbol = unresolved["source_symbol"].astype(str)
    keyed = pd.DataFrame({"key": _by_symbol(symbol, finra_key).astype(str), "date": unresolved["date"]}).loc[_by_symbol(symbol, finra_marker_kind).isna()]
    owners = master.loc[master["ticker"].notna() & master["lineage_role"].isin(SUMMED_ROLES), ["key", "ticker"]].drop_duplicates()
    incomplete = keyed.merge(owners, on="key")[["ticker", "date"]]
    ratio = master["conversion_ratio"]
    bad = master.loc[master["ticker"].isin(set(universe)) & master["lineage_role"].isin(SUMMED_ROLES) & ~(np.isfinite(ratio) & ratio.gt(0))]
    invalid = _day_spans(bad.rename(columns={"valid_from": "start", "valid_to": "stop"}), days)
    return pd.concat([incomplete, invalid], ignore_index=True).drop_duplicates()


def short_volume_series(
    lines: pd.DataFrame, master: pd.DataFrame, lineage: pd.DataFrame, universe: Collection[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(series, fallback)` of the stored FINRA lines: the (ticker, date) short-volume series of the `universe` companies
    over the served dates (a supported absence where no line of the company was filed), and the lineage-fallback stamps
    of the companies with no FTD line (the tenure mask's evidence).

    `master` are the step's `universe_cusip` lines, `lineage` its `universe_cik_symbol` rows. A served date is a date of a
    stored line (every FINRA file carries the universe's common lines).
    """
    rows = lines.reset_index(drop=True).assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
    keyed, tape = _short_master(master), _lineage_tape(lineage)
    days = np.sort(rows["date"].unique().astype("datetime64[ns]"))

    # stamp, sum, then the checked dates: supported or filed, minus unresolved-line and invalid-ratio dates
    stamped = _stamp_short(rows, keyed, tape, universe)
    grain = _short_sums(stamped, universe)
    dates = pd.concat([_supported_short(universe, days, keyed, tape), grain[["ticker", "date"]]], ignore_index=True).drop_duplicates()
    dates = dates.loc[dates["ticker"].isin(set(universe))]
    blocked = _blocked_short(stamped, keyed, universe, days).assign(_blocked=True)
    dates = dates.merge(blocked, on=["ticker", "date"], how="left")
    dates = dates.loc[dates.pop("_blocked").isna()]

    # the series: each checked date's sum, else a supported absence (0 short, total unknown)
    series = dates.merge(grain, on=["ticker", "date"], how="left")
    series["short_volume"] = series["short_volume"].fillna(0.0)
    series = series.astype({"short_volume": "float64", "total_volume": "float64"}).sort_values(["ticker", "date"], ignore_index=True)
    is_fallback = stamped["security_id"].fillna("").str.startswith("S") & ~stamped["ticker"].isin(set(keyed["ticker"].dropna()))
    fallback = stamped.loc[is_fallback & stamped["lineage_role"].isin(CANONICAL_ROLES), list(FALLBACK_COLUMNS)]
    return series[list(SHORT_SERIES_COLUMNS)], fallback.drop_duplicates(ignore_index=True)
