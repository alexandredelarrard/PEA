"""
fetch_fails_to_deliver.py (src/data_extract/utils/institutionals/fetch_fails_to_deliver.py)
------------------------------------------------------------------------------------
SEC Fails-to-Deliver semi-monthly ZIPs -> `sec_fails_to_deliver` (ticker, date), its own table so
its publication lag never moves `short_interest`'s frontier. Values are the cumulative net
unsettled balance on each settlement date, not new fails. Resume skips periods already processed
under the current symbol policy; each (symbol, settlement date) resolves through the dated `entity_lineage`
symbol intervals (`ticker_for_symbol`); `full` replaces the table. `download_fails_to_deliver` (the
`ftd-download` stage) caches the ZIPs and stores the in-scope source lines raw, per CUSIP, in
`sec_fails_to_deliver_security` for the security master.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Collection
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants.constants import FTD_ZIP_NAME_TEMPLATE
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir,
    ensure_zip,
    mark_processed,
    pending_periods,
    read_zip_text,
)
from src.data_extract.utils.common.identity import (
    Identity,
    load_identity,
    log_symbol_resolutions,
    symbol_rows_to_tickers,
)
from src.data_extract.utils.common.incremental import stored_values
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.security_master import (
    cusip_votes,
    lineage_scope_symbols,
    load_security_manual,
    squash,
    trade_dates,
)
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

_OUT_COLS = ["ticker", "date", "fails_quantity", "fails_value", "period"]
#: Column order of `sec_fails_to_deliver_security`; the last four are stamped from the security master.
SECURITY_COLUMNS = (
    "date",
    "trade_date",
    "cusip",
    "source_symbol",
    "description",
    "price",
    "fails_quantity",
    "fails_value",
    "period",
    "security_id",
    "ticker",
    "lineage_role",
    "security_class",
)
_LINEAGE_SCOPE_COLUMNS = ["entity_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources"]
#: Scope members of the raw ingest that are CUSIP-6 prefixes, not symbols.
_PREFIX_TAG = "cusip6:"
_POLICY_MARKER = "__point_in_time_symbol_identity_v2__"

# The ZIP's period tag, not the settlement day, controls availability; files <= 2017-06a live on the FOIA path.
SEC_FTD_URL_TEMPLATE = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_URL_TEMPLATE = "https://www.sec.gov/files/data/frequently-requested-foia-document-fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_LAST_PERIOD = "201706a"  # last period on the legacy path
SEC_FTD_FIRST_YEAR = 2009  # earliest FTD file (2009-07)


def _periods(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Semi-monthly file tags ('YYYYMMa'/'YYYYMMb') from the data-set era to now."""
    today = (today or pd.Timestamp.today()).normalize()
    out: list[str] = []
    for y in range(today.year - years_history, today.year + 1):
        if y < SEC_FTD_FIRST_YEAR:
            continue
        for m in range(1, 13):
            if (y, m) > (today.year, today.month):
                break
            out += [f"{y}{m:02d}a", f"{y}{m:02d}b"]
    return out


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested)                                                       #
# --------------------------------------------------------------------------- #
def _parse_ftd(raw: str) -> pd.DataFrame:
    """Parse one FTD file while retaining its historical symbol and raw CUSIP."""
    if not raw or "|" not in raw:
        return pd.DataFrame(columns=["date", "source_symbol", "cusip", "fails_quantity", "fails_value"])
    df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, engine="python", on_bad_lines="skip")
    cols = {c.strip().upper(): c for c in df.columns}

    def col(name: str) -> pd.Series:
        return df[cols[name]] if name in cols else pd.Series(pd.NA, index=df.index)

    price = pd.to_numeric(col("PRICE").astype("string").str.strip().where(lambda s: s != "."), errors="coerce")
    qty = pd.to_numeric(col("QUANTITY (FAILS)"), errors="coerce")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(col("SETTLEMENT DATE"), format="%Y%m%d", errors="coerce"),
            "source_symbol": col("SYMBOL")
            .astype("string")
            .str.upper()
            .str.replace(".", "-", regex=False)
            .str.replace("/", "-", regex=False)
            .str.strip(),
            "cusip": col("CUSIP").astype("string").str.strip(),
            "fails_quantity": qty,
            "fails_value": qty * price,
        }
    ).dropna(subset=["date", "source_symbol"])
    out = out[out["source_symbol"] != ""]
    if out.empty:
        return out
    return out.groupby(["date", "source_symbol", "cusip"], as_index=False, dropna=False)[["fails_quantity", "fails_value"]].sum(min_count=1)


def _parse_ftd_lines(raw: str) -> pd.DataFrame:
    """Every source line of one FTD file, raw: symbol as filed, description, PRICE ('.' -> NULL), trade date.

    Exact duplicate lines collapse to one; nothing is summed.
    """
    columns = list(SECURITY_COLUMNS[:8])
    if not raw or "|" not in raw:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, engine="python", on_bad_lines="skip", quoting=3)
    cols = {c.strip().upper(): c for c in df.columns}

    def col(name: str) -> pd.Series:
        return df[cols[name]].astype("string").str.strip() if name in cols else pd.Series(pd.NA, index=df.index, dtype="string")

    price = pd.to_numeric(col("PRICE").where(lambda s: s != "."), errors="coerce")
    qty = pd.to_numeric(col("QUANTITY (FAILS)"), errors="coerce")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(col("SETTLEMENT DATE"), format="%Y%m%d", errors="coerce"),
            "cusip": col("CUSIP").str.upper(),
            "source_symbol": col("SYMBOL").str.upper(),
            "description": col("DESCRIPTION"),
            "price": price,
            "fails_quantity": qty,
            "fails_value": qty * price,
        }
    ).dropna(subset=["date", "cusip"])
    out = out[out["cusip"].ne("")].drop_duplicates(ignore_index=True)
    out["trade_date"] = trade_dates(out["date"])
    return out[columns]


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #


def _period_urls(period: str) -> tuple[str, ...]:
    """Download URLs for a period: the date-appropriate path first (legacy <= 2017-06a), the other
    as fallback. Fixed-width 'YYYYMMx' tags sort chronologically, so a string compare is safe."""
    modern = SEC_FTD_URL_TEMPLATE.format(period=period)
    legacy = SEC_FTD_LEGACY_URL_TEMPLATE.format(period=period)
    return (legacy, modern) if period <= SEC_FTD_LEGACY_LAST_PERIOD else (modern, legacy)


def _cached_periods(cache: Path) -> set[str]:
    """Period tags present in the local SEC FTD ZIP cache."""
    return {path.stem.removeprefix("cnsfails") for path in cache.glob("cnsfails*.zip")}


def _canonicalise_ftd(
    context: Context,
    frame: pd.DataFrame,
    identity: Identity,
    universe: frozenset[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve each symbol on its settlement date and aggregate onto the destination table grain."""
    accepted, unresolved = symbol_rows_to_tickers(identity, frame, universe)
    log_symbol_resolutions(context, "FTD", accepted, unresolved, universe=universe)
    if accepted.empty:
        return pd.DataFrame(columns=_OUT_COLS), unresolved
    grouped = accepted.groupby(["ticker", "date"], as_index=False).agg(
        fails_quantity=("fails_quantity", "sum"), fails_value=("fails_value", lambda values: values.sum(min_count=1)), period=("period", "first")
    )
    return grouped[_OUT_COLS], unresolved


def resolve_ticker_fails(context: Context, tickers: Collection[str], identity: Identity) -> tuple[pd.DataFrame, set[str]]:
    """`(rows, periods read)`: `tickers`' FTD rows re-resolved from every cached zip on the destination grain.

    No download, marker file or manifest entry.
    """
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    universe = frozenset(str(ticker).strip().upper() for ticker in tickers)
    candidates = identity.universe_symbols(universe)
    frames: list[pd.DataFrame] = []
    periods: set[str] = set()
    for period in sorted(_cached_periods(cache)):
        raw = read_zip_text(cache / FTD_ZIP_NAME_TEMPLATE.format(period=period), log=logger)
        if raw is None:
            continue
        periods.add(period)
        df = _parse_ftd(raw)
        df = df[df["source_symbol"].isin(candidates)].copy()
        if not df.empty:
            frames.append(df.assign(period=period))
    if not frames:
        return pd.DataFrame(columns=_OUT_COLS), periods
    accepted, _ = _canonicalise_ftd(context, pd.concat(frames, ignore_index=True), identity, universe)
    return accepted, periods


def _validate_full_frame(
    frame: pd.DataFrame,
    universe: frozenset[str],
    parsed_periods: set[str],
    cached_periods: set[str],
) -> None:
    """Raise before replacement when cache coverage, keys or universe scope are unsafe."""
    missing = sorted(cached_periods - parsed_periods)
    if missing:
        raise ValueError(f"FTD full rebuild did not parse cached period(s): {missing}")
    if frame.empty:
        raise ValueError("FTD full rebuild staged no accepted rows")
    if frame[["ticker", "date"]].isna().any().any():
        raise ValueError("FTD full rebuild staged a null destination key")
    duplicate = frame.duplicated(["ticker", "date"], keep=False)
    if duplicate.any():
        raise ValueError(f"FTD full rebuild staged {int(duplicate.sum())} duplicate key row(s)")
    outside = sorted(set(frame["ticker"]) - set(universe))
    if outside:
        raise ValueError(f"FTD full rebuild staged non-universe ticker(s): {outside}")


def fetch_fails_to_deliver(
    context: Context,
    tickers: list[str],
    years_history: int = 15,
    full: bool = False,
    identity: Identity | None = None,
) -> int:
    """Resolve SEC FTD history through the lineage and incrementally save or fully replace it."""

    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    resolver = identity or load_identity(context)
    universe = frozenset(str(ticker).strip().upper() for ticker in tickers)
    candidates = resolver.universe_symbols(universe)
    policy_scope = set(candidates) | {_POLICY_MARKER}
    # A full rebuild must reproduce every stored period.
    stored_periods = stored_values(context, Tables.sec_fails_to_deliver, "period") if full else frozenset()

    saved = 0
    initial_cached = _cached_periods(cache)
    periods = sorted(initial_cached | set(_periods(years_history + 1)))
    pending = pending_periods(context, cache, Tables.sec_fails_to_deliver, periods, policy_scope, reparse=full)
    raw_frames: list[pd.DataFrame] = []
    parsed_periods: set[str] = set()
    for period in tqdm(pending, desc="SEC fails-to-deliver"):
        path = ensure_zip(
            context,
            cache / FTD_ZIP_NAME_TEMPLATE.format(period=period),
            _period_urls(period),
            label=f"FTD {period}",
            timeout=180,
            log=logger,
        )
        if path is None:
            if full and period in stored_periods:
                raise FileNotFoundError(f"FTD full rebuild cannot reproduce stored period {period}")
            continue
        raw = read_zip_text(path, log=logger)
        if raw is None:
            if full:
                raise ValueError(f"FTD full rebuild cannot read cached period {period}")
            continue
        df = _parse_ftd(raw)
        parsed_periods.add(period)
        df = df[df["source_symbol"].isin(candidates)].copy()
        if df.empty:
            continue
        df["period"] = period
        raw_frames.append(df)

    raw_complete = (
        pd.concat(raw_frames, ignore_index=True)
        if raw_frames
        else pd.DataFrame(columns=["date", "source_symbol", "cusip", "fails_quantity", "fails_value", "period"])
    )
    accepted, unresolved = _canonicalise_ftd(context, raw_complete, resolver, universe)

    if full:
        final_cached = _cached_periods(cache)
        _validate_full_frame(accepted, universe, parsed_periods, final_cached)
        saved = context.store.replace(Tables.sec_fails_to_deliver, accepted)
    elif not accepted.empty:
        saved = context.store.save(Tables.sec_fails_to_deliver, accepted)

    unresolved_count = len(unresolved)
    mark_processed(cache, Tables.sec_fails_to_deliver, policy_scope)
    logger.info(f"sec_fails_to_deliver completed ({len(periods)} files scanned) +{saved}")
    logger.info(f"FTD: {unresolved_count} unresolved raw row(s) excluded")
    record_run(context, Tables.sec_fails_to_deliver, len(tickers), saved, is_full_rescan=full)
    return saved


def _ingest_scope(context: Context) -> tuple[pd.DataFrame | None, frozenset[str], frozenset[str]]:
    """`(lineage, squashed lineage symbols, CUSIP-6 prefixes)`; prefixes come from the stored master and the manual config."""
    lineage = context.store.load(Tables.entity_lineage, columns=_LINEAGE_SCOPE_COLUMNS, optional=True)
    if lineage is None:
        return None, frozenset(), frozenset()
    master = context.store.load(Tables.security_master, columns=["cusip"], optional=True)
    manual = load_security_manual(getattr(context, "config_dir", None))
    cusips = set() if master is None else set(master["cusip"].dropna().astype(str))
    cusips |= set(manual.boundaries["cusip"].dropna()) | set(manual.ratios["cusip"].dropna())
    return lineage, lineage_scope_symbols(lineage), frozenset(c[:6] for c in cusips if c)


def _in_scope(lines: pd.DataFrame, symbols: frozenset[str], prefixes: frozenset[str]) -> pd.DataFrame:
    keys = lines["source_symbol"].map(squash)
    return lines[keys.isin(symbols) | lines["cusip"].str[:6].isin(prefixes)]


def _read_period(context: Context, cache: Path, period: str, *, download: bool) -> pd.DataFrame | None:
    """One period's raw lines from the cache (downloading it once when `download`); None when unavailable."""
    path: Path | None = cache / FTD_ZIP_NAME_TEMPLATE.format(period=period)
    if download:
        path = ensure_zip(context, path, _period_urls(period), label=f"FTD {period}", timeout=180, log=logger)
    raw = None if path is None else read_zip_text(path, log=logger)
    return None if raw is None else _parse_ftd_lines(raw).assign(period=period)


def download_fails_to_deliver(context: Context, years_history: int = 15, full: bool = False) -> int:
    """Cache the FTD ZIPs and store their in-scope source lines raw in `sec_fails_to_deliver_security`.

    Scope: lineage symbols plus the CUSIP-6 of the master's securities; a CUSIP-6 newly voted for by a
    lineage symbol interval re-reads every cached period for it. Stamp columns stay NULL. `full` replaces
    the table from every cached period. Returns the rows written.
    """
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    table = Tables.sec_fails_to_deliver_security
    lineage, symbols, prefixes = _ingest_scope(context)
    periods = sorted(_cached_periods(cache) | set(_periods(years_history + 1)))
    if lineage is None:
        cached = [
            p
            for p in periods
            if ensure_zip(context, cache / FTD_ZIP_NAME_TEMPLATE.format(period=p), _period_urls(p), label=f"FTD {p}", timeout=180, log=logger)
        ]
        logger.warning("FTD download: no entity_lineage yet; %d zip(s) cached, lines are ingested after identity-tables has run", len(cached))
        return 0
    scope = set(symbols) | {_PREFIX_TAG + p for p in prefixes}
    pending = pending_periods(context, cache, table, periods, scope, reparse=full)
    frames = []
    for period in tqdm(pending, desc="FTD download"):
        frame = _read_period(context, cache, period, download=True)
        if frame is not None:
            frames.append(_in_scope(frame, symbols, prefixes))
    kept = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(SECURITY_COLUMNS[:9]))
    voted = cusip_votes(kept, lineage) if not kept.empty else pd.DataFrame(columns=["cusip"])
    added = frozenset(str(c)[:6] for c in voted["cusip"]) - prefixes
    if added:
        logger.info("FTD download: %d CUSIP-6 prefix(es) newly voted for (%s); re-reading every cached period", len(added), ", ".join(sorted(added)))
        for period in sorted(_cached_periods(cache)):
            frame = _read_period(context, cache, period, download=False)
            if frame is not None:
                kept = pd.concat([kept, _in_scope(frame, frozenset(), added)], ignore_index=True)
        scope |= {_PREFIX_TAG + p for p in added}
    kept = kept.drop_duplicates(["cusip", "date", "source_symbol", "fails_quantity"], ignore_index=True)
    duplicate = kept.duplicated(["cusip", "date"], keep="first")
    if duplicate.any():
        logger.warning("FTD download: %d line(s) repeat a (cusip, settlement date) key with other values; the first is kept", int(duplicate.sum()))
        kept = kept[~duplicate]
    rows = kept.assign(security_id=None, ticker=None, lineage_role=None, security_class=None)[list(SECURITY_COLUMNS)]
    if full:
        written = context.store.replace(table, rows)
    else:
        written = context.store.save(table, rows) if not rows.empty else 0
    mark_processed(cache, table, scope)
    record_run(context, table, 0, written, is_full_rescan=full)
    logger.info(
        "FTD download: %d period(s) read, %d in-scope line(s) stored (%d symbol(s), %d CUSIP-6 prefix(es) in scope)",
        len(pending),
        written,
        len(symbols),
        len(prefixes | added),
    )
    return written
