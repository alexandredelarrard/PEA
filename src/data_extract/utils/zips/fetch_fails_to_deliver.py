"""
fetch_fails_to_deliver.py (src/data_extract/utils/zips/fetch_fails_to_deliver.py)
------------------------------------------------------------------------------------
SEC Fails-to-Deliver semi-monthly ZIPs. `download_fails_to_deliver` (`zips-download --only ftd`) caches every listed
ZIP; `fetch_fails_to_deliver` (`zips-ingest --only ftd`) stores the lines of each new cached period as filed in
`sec_fails_to_deliver_security`, one row per (CUSIP, settlement date), source columns only. A line is kept when its
squashed symbol is a universe lineage symbol (not noise), its CUSIP-6 is a prefix of a universe company's Sharadar or
curated CUSIP, or its CUSIP-6 is on a symbol-matched line; the scope never reads `universe_cusip`. The identity step
reads these lines as CUSIP evidence; the institutionals cube maps them to companies (`src/utils/security_series.py`).
Values are cumulative net unsettled balances, not new fails. `full` re-reads every cached ZIP.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants.constants import FTD_ZIP_NAME_TEMPLATE
from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir, cached_periods, ensure_zip, read_zip_text
from src.data_extract.utils.common.identity_store import load_stored_lineage
from src.data_extract.utils.common.resume import ARCHIVE_NEW_KEY_DAYS, archive_periods, period_end, recently_changed
from src.data_extract.utils.common.security_lines import SHARADAR_CUSIP_COLUMNS, SecurityManual, ftd_scope, load_security_manual, sharadar_cusips
from src.data_store.schema import Resume, Tables
from src.data_store.store import DataStore
from src.utils.security_series import trade_dates
from src.utils.string import normalise_ticker, run_date, squash

logger = logging.getLogger(__name__)

#: Column order of `sec_fails_to_deliver_security`: the source line as filed and its ZIP period.
SECURITY_COLUMNS = ("date", "trade_date", "cusip", "source_symbol", "description", "price", "fails_quantity", "fails_value", "period")

# The ZIP's period tag, not the settlement day, controls availability; files <= 2017-06a live on the FOIA path.
SEC_FTD_URL_TEMPLATE = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_URL_TEMPLATE = "https://www.sec.gov/files/data/frequently-requested-foia-document-fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_OTHER_URL_TEMPLATE = "https://www.sec.gov/files/data/other/fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_LAST_PERIOD = "201706a"  # last period on the legacy path
SEC_FTD_ZIP_PREFIX = "cnsfails"  # a cached archive is named `{prefix}{period}.zip`


def ftd_periods(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Semi-monthly file tags ('YYYYMMa'/'YYYYMMb') from the data-set era to now."""
    today = run_date(today)
    source_start = pd.Timestamp(cast(str, cast(Resume, Tables.sec_fails_to_deliver_security.resume).source_start))
    out: list[str] = []
    for y in range(max(today.year - years_history, source_start.year), today.year + 1):
        for m in range(1, 13):
            if (y, m) < (source_start.year, source_start.month):
                continue
            if (y, m) > (today.year, today.month):
                break
            out += [period for half in ("a", "b") if period_end(period := f"{y}{m:02d}{half}") >= source_start.date()]
    return out


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested)                                                       #
# --------------------------------------------------------------------------- #
def _ftd_column(df: pd.DataFrame, cols: dict[str, str], name: str) -> pd.Series:
    """The stripped text of the file column headed `name` (any case), or an all-NA column when the file lacks it."""
    return df[cols[name]].astype("string").str.strip() if name in cols else pd.Series(pd.NA, index=df.index, dtype="string")


def _strip_ftd_trailer(lines: list[str]) -> list[str]:
    """The file's lines without the SEC `Trailer ...` lines; a stated record count must equal the data lines."""
    body = [row for row in lines if not row.startswith("Trailer ")]
    trailers = [row for row in lines if row.startswith("Trailer ")]
    counts = [match.group(1) for row in trailers if (match := re.fullmatch(r"Trailer record count\s+(\d+)\s*", row))]
    if counts and int(counts[-1]) != len(body) - 1:
        raise ValueError(f"FTD trailer record count {counts[-1]} != {len(body) - 1} data lines")
    return body


def _parse_ftd_lines(raw: str) -> pd.DataFrame:
    """Every source line of one FTD file, raw: symbol as filed, description, PRICE ('.' -> NULL), trade date.

    The SEC trailer lines are dropped after checking their record count. Exact duplicate lines collapse
    to one; nothing is summed.
    """
    columns = list(SECURITY_COLUMNS[:8])
    lines = _strip_ftd_trailer(raw.strip().splitlines())
    header = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"
    if not lines or lines[0].strip().lstrip("﻿").upper() != header:
        raise ValueError("FTD invalid header or source line width")
    body = "\n".join(lines[1:])
    if body.count("|") == 5 * (len(lines) - 1):
        # every line has exactly six fields: the C reader, every field kept as filed text
        df = pd.read_csv(io.StringIO(body), sep="|", header=None, names=header.split("|"), dtype=str, na_filter=False, quoting=csv.QUOTE_NONE)
    else:
        # A description can itself hold a '|' (cnsfails202104b): the first four fields and the price stay fixed.
        rows = [row.split("|") for row in lines[1:]]
        if any(len(fields) < 6 for fields in rows):
            raise ValueError("FTD invalid header or source line width")
        df = pd.DataFrame([[*fields[:4], "|".join(fields[4:-1]), fields[-1]] for fields in rows], columns=header.split("|"), dtype=str)
    cols = {c.strip().upper(): c for c in df.columns}
    symbol = _ftd_column(df, cols, "SYMBOL")
    price = pd.to_numeric(_ftd_column(df, cols, "PRICE").where(lambda s: s != "."), errors="coerce")
    qty = pd.to_numeric(_ftd_column(df, cols, "QUANTITY (FAILS)"), errors="coerce")
    valid_price = _ftd_column(df, cols, "PRICE").eq(".") | (np.isfinite(price) & price.ge(0))
    valid_quantity = pd.Series(np.isfinite(qty) & qty.ge(0) & valid_price, index=df.index)
    if not valid_quantity.fillna(False).all():
        raise ValueError("FTD invalid finite nonnegative quantity or price")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(_ftd_column(df, cols, "SETTLEMENT DATE"), format="%Y%m%d", errors="coerce"),
            "cusip": _ftd_column(df, cols, "CUSIP").str.upper(),
            "source_symbol": symbol.map({value: normalise_ticker(value) for value in symbol.unique()}),
            "description": _ftd_column(df, cols, "DESCRIPTION"),
            "price": price,
            "fails_quantity": qty,
            "fails_value": qty * price,
        }
    )
    if out["date"].isna().any() or out["cusip"].eq("").any():
        raise ValueError("FTD invalid settlement date or CUSIP")
    if np.isinf(out["fails_value"].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("FTD source notional overflows finite storage")
    out = out.drop_duplicates(ignore_index=True)
    settled = out["date"].drop_duplicates()
    out["trade_date"] = out["date"].map(pd.Series(trade_dates(settled).to_numpy(), index=settled))
    return out[columns]


def unique_ftd_keys(lines: pd.DataFrame, label: str) -> pd.DataFrame:
    """Collapse exact source repeats; refuse conflicting payloads or ZIP periods at the raw PK."""
    kept = lines.drop_duplicates([column for column in SECURITY_COLUMNS if column in lines], ignore_index=True)
    if kept.duplicated(["cusip", "date"]).any():
        raise ValueError(f"{label}: conflicting FTD source payload or period for (cusip, settlement date)")
    return kept


def scope_lines(raw: pd.DataFrame, symbols: frozenset[str], prefixes: Collection[str]) -> tuple[pd.DataFrame, frozenset[str]]:
    """The lines of one period in scope (a lineage symbol, a CUSIP-6 in `prefixes`, or the CUSIP-6 of a symbol-matched line
    of this period) and the CUSIP-6 of its symbol-matched lines."""
    keys = raw["source_symbol"].map({value: squash(value) for value in raw["source_symbol"].dropna().unique()})
    matched = keys.isin(symbols)
    seen = frozenset(raw.loc[matched, "cusip"].str[:6])
    kept = matched | raw["cusip"].str[:6].isin(set(prefixes) | seen)
    return raw.loc[kept, list(SECURITY_COLUMNS)], seen


# --------------------------------------------------------------------------- #
# IO: cache/download                                                            #
# --------------------------------------------------------------------------- #


def _period_urls(period: str) -> tuple[str, ...]:
    """Download URLs for a period: the date-appropriate path first (legacy <= 2017-06a), the other
    as fallback, then the `other/` root. Fixed-width 'YYYYMMx' tags sort chronologically, so a string
    compare is safe."""
    modern = SEC_FTD_URL_TEMPLATE.format(period=period)
    legacy = SEC_FTD_LEGACY_URL_TEMPLATE.format(period=period)
    other = SEC_FTD_OTHER_URL_TEMPLATE.format(period=period)
    return (legacy, modern, other) if period <= SEC_FTD_LEGACY_LAST_PERIOD else (modern, legacy, other)


def cached_ftd_periods(cache: Path) -> set[str]:
    """Period tags present in the local SEC FTD ZIP cache."""
    return cached_periods(cache, prefix=SEC_FTD_ZIP_PREFIX)


def read_ftd_period(context: Context, cache: Path, period: str) -> pd.DataFrame | None:
    """One cached period's raw lines; None when unavailable.

    A file with invalid content is logged and kept in the cache; only a corrupt archive is deleted.
    """
    path = cache / FTD_ZIP_NAME_TEMPLATE.format(period=period)
    raw = read_zip_text(path, log=logger)
    try:
        if raw is None:
            raise ValueError("missing or unreadable text member")
        parsed = _parse_ftd_lines(raw).assign(period=period)
        end = pd.Timestamp(period_end(period))
        # SEC b units may include day15: retain their actual ZIP tag and publication clock.
        start = end.replace(day=1 if period.endswith("a") else 15)
        if not parsed["date"].between(start, end).all():
            raise ValueError(f"settlement date outside declared source period {period}")
    except ValueError as exc:
        logger.warning("FTD %s: invalid original (%s); skipped, cached file kept; good stored rows retained", period, exc)
        return None
    parsed.attrs["period"] = period
    return parsed


def ensure_ftd_zip(context: Context, cache: Path, period: str) -> Path | None:
    """The cached ZIP of `period`, downloaded once when absent; None when no location serves it."""
    return ensure_zip(
        context, cache / FTD_ZIP_NAME_TEMPLATE.format(period=period), _period_urls(period), label=f"FTD {period}", timeout=180, log=logger
    )


def download_fails_to_deliver(context: Context, years_history: int = 31, as_of: pd.Timestamp | None = None) -> int:
    """Cache every listed FTD ZIP, as a scope growth re-reads the whole cache (a cached ZIP is never downloaded again);
    returns the ZIPs available. Writes no table."""
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    periods = sorted(cached_ftd_periods(cache) | set(ftd_periods(years_history + 1, run_date(as_of))))
    return sum(ensure_ftd_zip(context, cache, period) is not None for period in tqdm(periods, desc="FTD download"))


# --------------------------------------------------------------------------- #
# IO: store the in-scope lines as filed                                         #
# --------------------------------------------------------------------------- #


@dataclass
class _Reads:
    """One run's reads: the cache, the CUSIP-6 in scope before the run, each period's CUSIP-6 set and read order, and each
    CUSIP-6 a symbol-matched line put in scope during the run (with the read that found it)."""

    cache: Path
    known: frozenset[str]
    held: dict[str, frozenset[str]] = field(default_factory=dict)
    order: dict[str, int] = field(default_factory=dict)
    found: dict[str, int] = field(default_factory=dict)


def _seen_prefixes(pairs: pd.DataFrame | None, symbols: frozenset[str]) -> frozenset[str]:
    """The CUSIP-6 of the stored lines a lineage symbol matched (the stored part of the symbol-matched scope)."""
    if pairs is None or pairs.empty:
        return frozenset()
    return frozenset(pairs.loc[pairs["source_symbol"].fillna("").map(squash).isin(symbols), "cusip"].astype(str).str[:6])


def _rowless_keys(
    lineage: pd.DataFrame,
    listed: dict[str, frozenset[str]],
    manual: SecurityManual,
    pairs: pd.DataFrame | None,
    companies: Collection[str],
    as_of: pd.Timestamp,
) -> tuple[frozenset[str], frozenset[str]]:
    """The scope keys whose source changed within `ARCHIVE_NEW_KEY_DAYS` and that have no stored line: the lineage symbols of
    recently changed rows, and the Sharadar / curated CUSIP-6 of `companies` (recently changed or joined)."""
    rows = lineage.loc[lineage["role"].eq("symbol") & lineage["status"].ne("noise")].reset_index(drop=True)
    recent = set(recently_changed({str(i): stamp for i, stamp in rows["scope_changed_at"].items()}, as_of, ARCHIVE_NEW_KEY_DAYS))
    symbols = {key for i, key in rows["symbol"].map(squash).items() if str(i) in recent and key}
    curated = [
        *manual.boundaries.loc[manual.boundaries["ticker"].isin(companies), "cusip"],
        *manual.ratios.loc[manual.ratios["ticker"].isin(companies), "cusip"],
    ]
    prefixes = {str(cusip)[:6] for cusip in [*curated, *(c for t in companies for c in listed.get(t, ()))] if isinstance(cusip, str) and cusip}
    stored = pairs if pairs is not None else pd.DataFrame(columns=["cusip", "source_symbol"])
    have_symbols, have_prefixes = set(stored["source_symbol"].fillna("").map(squash)), set(stored["cusip"].astype(str).str[:6])
    return frozenset(symbols - have_symbols), frozenset(prefixes - have_prefixes)


def _save_lines(store: DataStore, period: str, lines: pd.DataFrame, replace: bool) -> int:
    """A period-scoped replace of `period`'s stored lines (`replace`), else a keyed upsert of `lines`."""
    table = Tables.sec_fails_to_deliver_security
    lines = unique_ftd_keys(lines, f"FTD {period}")
    if not replace:
        return store.save(table, lines) if not lines.empty else 0
    with store.transaction() as bound:
        bound.delete(table, {"period": period})
        return bound.save(table, lines) if not lines.empty else 0


def _read_periods(context: Context, reads: _Reads, periods: list[str], symbols: frozenset[str], prefixes: frozenset[str], replace: bool) -> int:
    """Read each period once and save its lines of `symbols`, `prefixes` and the CUSIP-6 found so far; a CUSIP-6 a
    symbol-matched line adds is in scope for the periods read after it. Returns the lines written."""
    written = 0
    for period in tqdm(periods, desc="SEC fails-to-deliver" if replace else "SEC fails-to-deliver (scope growth)"):
        raw = read_ftd_period(context, reads.cache, period)
        if raw is None:
            continue
        reads.order[period] = len(reads.order)
        reads.held[period] = frozenset(raw["cusip"].str[:6].unique())
        kept, seen = scope_lines(raw, symbols, prefixes | set(reads.found))
        reads.found.update({prefix: reads.order[period] for prefix in seen - reads.known - reads.found.keys()})
        written += _save_lines(context.store, period, kept, replace)
    return written


def _catch_up(context: Context, reads: _Reads) -> int:
    """Each period read before a CUSIP-6 entered the scope, and holding it, is read again for that CUSIP-6 (keyed upsert)."""
    late = {period: {p for p, at in reads.found.items() if order < at and p in reads.held[period]} for period, order in reads.order.items()}
    written = 0
    for period, need in tqdm(sorted((period, need) for period, need in late.items() if need), desc="SEC fails-to-deliver (catch-up)"):
        raw = read_ftd_period(context, reads.cache, period)
        if raw is not None:
            written += _save_lines(context.store, period, raw.loc[raw["cusip"].str[:6].isin(need), list(SECURITY_COLUMNS)], False)
    return written


def fetch_fails_to_deliver(context: Context, full: bool = False, as_of: pd.Timestamp | None = None) -> int:
    """Store the in-scope lines of each new cached FTD period as filed (`full`: every cached period); returns lines written.

    Scope growth: a key whose source changed within `ARCHIVE_NEW_KEY_DAYS` and that has no stored line, and a CUSIP-6 a
    symbol-matched line adds during the run, re-read the cached periods for those keys only (a keyed upsert).
    """
    # load: the lineage, the Sharadar CUSIPs, the curation, the stored (CUSIP, symbol) keys and the planned periods
    table = Tables.sec_fails_to_deliver_security
    lineage = load_stored_lineage(context)
    sharadar = context.store.load(Tables.sharadar_tickers, columns=list(SHARADAR_CUSIP_COLUMNS), optional=True)
    manual = load_security_manual(str(context.config_dir))
    pairs = context.store.summarize(table, ["cusip", "source_symbol"], {"n": ("count", None)}) if context.store.exists(table) else None
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    listed_periods, stored = archive_periods(context, [table], sorted(cached_ftd_periods(cache)))
    if lineage is None or lineage.empty:
        logger.warning("FTD: no universe_cik_symbol yet (run `identity` first); nothing stored")
        return 0

    # the scope: lineage symbols, Sharadar / curated CUSIP-6 and the CUSIP-6 of stored symbol-matched lines
    listed = sharadar_cusips(context.universe.frame, sharadar)
    scope = ftd_scope(lineage, listed, manual)
    reads = _Reads(cache, scope.prefixes | _seen_prefixes(pairs, scope.symbols))
    day = run_date(as_of)
    companies = set(recently_changed(lineage.groupby("canonical_ticker")["scope_changed_at"].max().to_dict(), day)) | context.universe.new_since(
        ARCHIVE_NEW_KEY_DAYS, day
    )
    symbols, prefixes = _rowless_keys(lineage, listed, manual, pairs, companies, day)

    # save: each new period, then the scope growth over the other cached periods, then the periods read before a growth
    pending = listed_periods if full else [period for period in listed_periods if period not in stored]
    written = _read_periods(context, reads, pending, scope.symbols, reads.known, True)
    others = [period for period in listed_periods if period not in reads.order]
    grown = _read_periods(context, reads, others, symbols, prefixes, False) if others and (symbols or prefixes or reads.found) else 0
    grown += _catch_up(context, reads)
    logger.info(
        "FTD: %d period(s) stored, %d line(s); scope growth: %d key(s) re-read (%d symbol(s), %d CUSIP-6), %d line(s)",
        len(pending),
        written,
        len(symbols) + len(prefixes) + len(reads.found),
        len(symbols),
        len(prefixes) + len(reads.found),
        grown,
    )
    return written + grown
