"""
fetch_fails_to_deliver.py (src/data_extract/utils/institutionals/fetch_fails_to_deliver.py)
------------------------------------------------------------------------------------
SEC Fails-to-Deliver (FTD): semi-monthly settlement-fail files, a signal for
settlement stress / short-squeeze risk. Kept in its own table, separate from
`short_interest`, so its ~2-month publication lag doesn't corrupt that table's
global-max-date incremental sync (see schema.py).

It is a Cumulative Balance, NOT Daily New Fails:
The number listed on date t represents
the total net unsettled balance as of that night.
$$\text{FTD}_t = \text{FTD}_{t-1} + \text{New Fails}_t - \text{Resolved Fails}_t$$
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir,
    ensure_zip,
    ingested_periods,
    read_zip_text,
)
from src.data_extract.utils.common.identity import (
    Identity,
    load_identity,
    log_symbol_resolutions,
    resolve_symbol_rows,
)
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import load_processed_universe, save_processed_universe
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

_OUT_COLS = ["ticker", "date", "fails_quantity", "fails_value", "period"]
_POLICY_MARKER = "__point_in_time_symbol_identity_v1__"

# {period} = "YYYYMMa" for settlement dates 1-15, "YYYYMMb" for 16-end. The SAME
# cnsfails{period}.zip files (identical pipe format) live under TWO paths:
#   * current path       -> 2017-06b onward
#   * FOIA "legacy" path  -> 2009-07a .. 2017-06a  (pre-2017-06 history)
SEC_FTD_URL_TEMPLATE = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_URL_TEMPLATE = "https://www.sec.gov/files/data/" "frequently-requested-foia-document-fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_LAST_PERIOD = "201706a"  # last period on the legacy path (>= 201706b uses the current path)
SEC_FTD_FIRST_YEAR = 2009  # earliest FTD file overall (2009-07, legacy path) -> full 15y coverage


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
            "source_symbol": col("SYMBOL").astype("string").str.upper().str.replace(".", "-", regex=False).str.strip(),
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
    """Download URL(s) for a semi-monthly period, path chosen by date: the FOIA
    'legacy' path for <= 2017-06a, the current path for >= 2017-06b. The other path
    is tried as a fallback (boundary / occasional re-issued files live on both).
    Fixed-width 'YYYYMMx' tags sort chronologically, so a string compare is safe."""
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


def fetch_fails_to_deliver(
    context: Context,
    tickers: list[str],
    years_history: int = 15,
    full: bool = False,
    identity: Identity | None = None,
) -> int:
    """Resolve SEC FTD history point-in-time and incrementally save or fully replace it."""

    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    resolver = identity or load_identity(context)
    universe = frozenset(str(ticker).strip().upper() for ticker in tickers)
    candidates = resolver.candidate_symbols(universe)
    policy_scope = set(candidates) | {_POLICY_MARKER}
    processed_scope = load_processed_universe(cache, Tables.sec_fails_to_deliver)
    changed_scope = policy_scope - processed_scope
    done = set() if full else ingested_periods(context, Tables.sec_fails_to_deliver)
    if changed_scope and not full:
        logger.info("FTD: identity scope changed by %d symbol(s) -> re-parsing cache", len(changed_scope))

    saved = 0
    initial_cached = _cached_periods(cache)
    periods = sorted(initial_cached | set(_periods(years_history + 1)))
    raw_frames: list[pd.DataFrame] = []
    parsed_periods: set[str] = set()
    stored_periods = (
        set(context.store.distinct(Tables.sec_fails_to_deliver, "period")) if full and context.store.exists(Tables.sec_fails_to_deliver) else set()
    )
    for period in tqdm(periods, desc="SEC fails-to-deliver"):
        if period in done and not changed_scope:
            continue
        path = ensure_zip(context, cache / f"cnsfails{period}.zip", _period_urls(period), label=f"FTD {period}", timeout=180, log=logger)
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
    save_processed_universe(cache, Tables.sec_fails_to_deliver, policy_scope)
    logger.info(f"sec_fails_to_deliver completed ({len(periods)} files scanned) +{saved}")
    logger.info(f"FTD: {unresolved_count} unresolved raw row(s) excluded")
    record_run(context, Tables.sec_fails_to_deliver, len(tickers), saved, is_full_rescan=full)
    return saved
