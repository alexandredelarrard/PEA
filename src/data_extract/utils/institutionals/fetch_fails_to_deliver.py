"""
fetch_fails_to_deliver.py (src/data_extract/utils/institutionals/fetch_fails_to_deliver.py)
------------------------------------------------------------------------------------
SEC Fails-to-Deliver semi-monthly ZIPs -> `sec_fails_to_deliver` (ticker, date), its own table so
its publication lag never moves `short_interest`'s frontier. Values are the cumulative net
unsettled balance on each settlement date, not new fails. Each run parses the periods with no stored
row (`resume.archive_worklist`), plus every cached period for a new ticker with no row yet; historical
symbols resolve point-in-time; an unscoped `full` run replaces the table.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants.constants import FTD_ZIP_NAME_TEMPLATE
from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir, cached_periods, ensure_zip, read_zip_text
from src.data_extract.utils.common.identity import (
    Identity,
    load_identity,
    log_symbol_resolutions,
    resolve_symbol_rows,
)
from src.data_extract.utils.common.incremental import stored_values
from src.data_extract.utils.common.resume import archive_worklist
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

_OUT_COLS = ["ticker", "date", "fails_quantity", "fails_value", "period"]
_RAW_COLS = ["date", "source_symbol", "cusip", "fails_quantity", "fails_value", "period"]

# The ZIP's period tag, not the settlement day, controls availability; files <= 2017-06a live on the FOIA path.
SEC_FTD_URL_TEMPLATE = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_URL_TEMPLATE = "https://www.sec.gov/files/data/frequently-requested-foia-document-fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_LAST_PERIOD = "201706a"  # last period on the legacy path
SEC_FTD_FIRST_YEAR = 2009  # earliest FTD file (2009-07)
SEC_FTD_ZIP_PREFIX = "cnsfails"  # a cached archive is named `{prefix}{period}.zip`


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


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #


def _period_urls(period: str) -> tuple[str, ...]:
    """Download URLs for a period: the date-appropriate path first (legacy <= 2017-06a), the other
    as fallback. Fixed-width 'YYYYMMx' tags sort chronologically, so a string compare is safe."""
    modern = SEC_FTD_URL_TEMPLATE.format(period=period)
    legacy = SEC_FTD_LEGACY_URL_TEMPLATE.format(period=period)
    return (legacy, modern) if period <= SEC_FTD_LEGACY_LAST_PERIOD else (modern, legacy)


def _canonicalise_ftd(
    context: Context,
    frame: pd.DataFrame,
    identity: Identity,
    universe: frozenset[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve historical symbols and aggregate onto the destination table grain."""
    accepted, unresolved = resolve_symbol_rows(identity, frame, universe)
    log_symbol_resolutions(context, "FTD", accepted, unresolved, universe=universe)
    if accepted.empty:
        return pd.DataFrame(columns=_OUT_COLS), unresolved
    grouped = accepted.groupby(["ticker", "date"], as_index=False).agg(
        fails_quantity=("fails_quantity", "sum"), fails_value=("fails_value", lambda values: values.sum(min_count=1)), period=("period", "first")
    )
    return grouped[_OUT_COLS], unresolved


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


def _read_period(
    context: Context, cache: Path, period: str, symbols: frozenset[str], *, rebuild: bool = False, stored: bool = False
) -> pd.DataFrame | None:
    """One period's raw rows for `symbols`, tagged with the period; None when the archive is not
    served or unreadable. A full rebuild raises instead on an unreadable archive or an unserved stored period."""
    path = ensure_zip(
        context, cache / FTD_ZIP_NAME_TEMPLATE.format(period=period), _period_urls(period), label=f"FTD {period}", timeout=180, log=logger
    )
    if path is None:
        if rebuild and stored:
            raise FileNotFoundError(f"FTD full rebuild cannot reproduce stored period {period}")
        return None
    raw = read_zip_text(path, log=logger)
    if raw is None:
        if rebuild:
            raise ValueError(f"FTD full rebuild cannot read cached period {period}")
        return None
    df = _parse_ftd(raw)
    return df[df["source_symbol"].isin(symbols)].assign(period=period)


def _concat_raw(frames: list[pd.DataFrame]) -> pd.DataFrame:
    kept = [df for df in frames if not df.empty]
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=_RAW_COLS)


def _rebuild(context: Context, cache: Path, periods: list[str], resolver: Identity, universe: frozenset[str]) -> int:
    """Re-parse every listed period for the whole universe and replace the table, after validation."""
    stored_periods = stored_values(context, Tables.sec_fails_to_deliver, "period")
    candidates = resolver.candidate_symbols(universe)
    frames: list[pd.DataFrame] = []
    parsed: set[str] = set()
    for period in tqdm(periods, desc="SEC fails-to-deliver (full)"):
        df = _read_period(context, cache, period, candidates, rebuild=True, stored=period in stored_periods)
        if df is None:
            continue
        parsed.add(period)
        frames.append(df)
    df_accepted, df_unresolved = _canonicalise_ftd(context, _concat_raw(frames), resolver, universe)
    _validate_full_frame(df_accepted, universe, parsed, cached_periods(cache, prefix=SEC_FTD_ZIP_PREFIX))
    logger.info(f"FTD: {len(df_unresolved)} unresolved raw row(s) excluded")
    return context.store.replace(Tables.sec_fails_to_deliver, df_accepted)


def fetch_fails_to_deliver(
    context: Context,
    tickers: list[str],
    years_history: int,
    full: bool = False,
    identity: Identity | None = None,
    as_of: pd.Timestamp | None = None,
) -> int:
    """Parse the FTD periods missing from the table (and every cached period for a new ticker), resolve
    symbols point-in-time and upsert; an unscoped `full` run re-parses everything and replaces the table."""
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    resolver = identity or load_identity(context)
    run_date = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    cached = cached_periods(cache, prefix=SEC_FTD_ZIP_PREFIX)
    published = sorted(cached | set(_periods(years_history + 1, run_date)))
    work = archive_worklist(context, (Tables.sec_fails_to_deliver,), published, cached, tickers, run_date, full=full)
    if full and not work.scoped:
        saved = _rebuild(context, cache, sorted(set(work.listed) | cached), resolver, frozenset(work.keys))
        logger.info(f"sec_fails_to_deliver replaced ({len(work.listed)} files listed) {saved} rows")
        return saved

    # Pending periods keep every run key's rows, rescanned ones only the rescan keys'.
    groups = {"pending": (frozenset(work.keys), work.pending), "rescan": (frozenset(work.rescan_keys), work.rescan)}
    accepted_frames: list[pd.DataFrame] = []
    unresolved_count = 0
    for name, (keys, periods) in groups.items():
        if not keys or not periods:
            continue
        candidates = resolver.candidate_symbols(keys)
        frames = [
            df
            for period in tqdm(periods, desc=f"SEC fails-to-deliver ({name})")
            if (df := _read_period(context, cache, period, candidates)) is not None
        ]
        df_accepted, df_unresolved = _canonicalise_ftd(context, _concat_raw(frames), resolver, keys)
        accepted_frames.append(df_accepted)
        unresolved_count += len(df_unresolved)
    df_accepted = (
        pd.concat(accepted_frames, ignore_index=True).drop_duplicates(["ticker", "date"]) if accepted_frames else pd.DataFrame(columns=_OUT_COLS)
    )
    saved = context.store.save(Tables.sec_fails_to_deliver, df_accepted) if not df_accepted.empty else 0
    logger.info(f"sec_fails_to_deliver completed ({len(work.pending)} pending + {len(work.rescan)} rescanned period(s)) +{saved}")
    logger.info(f"FTD: {unresolved_count} unresolved raw row(s) excluded")
    return saved
