"""
fetch_short_interest.py (src/data_extract/utils/institutionals/fetch_short_interest.py)
---------------------------------------------------------------------------------
FINRA RegSHO consolidated daily short-sale VOLUME (`CNMSshvol` files; not reported short interest).
Each day-file line is stored raw in `sec_short_volume_security`, keyed on the symbol as filed and stamped
from `security_master`: the FINRA symbol is read before upper-casing (`BACpB` -> `BACPRB`, `BRK/A` -> `BRKA`,
the FTD spelling) and resolved on its trade date with `Identity.security_on`; a symbol never seen in FTD maps
only through a lineage window CIK inside its window (P21). The ticker-grain `short_interest` is rebuilt from
those rows: per (ticker, date) the sum over the canonical and secondary-class lines of volume x conversion
ratio. The day files read come from `resume.series_windows` over `short_interest` (forward overlap, new
keys in full until their days reach the later of the source start and their first `prices` date) plus the
calendar sessions inside the stored span on which no key has a row; `repair` adds each key's own interior
gaps once. An unscoped `full` re-fetches every served date and
keeps no legacy row; a scoped one (`-t`) re-reads every served date for its tickers and upserts them.
Missing the Lit exchange volumes.
"""

from __future__ import annotations

import io
import logging
import re
import time
from collections.abc import Collection, Mapping, Sequence

import pandas as pd
import requests
from src.utils.batching import batched_tuples
from tqdm import tqdm

from src.constants.constants import BROWSER_HEADERS, CANONICAL_CURRENT, CANONICAL_PREDECESSOR, DATE_FORMAT_COMPACT
from src.context import Context
from src.data_extract.utils.common.identity import Identity, SecurityHit, load_identity
from src.data_extract.utils.common.resume import document_floor, series_windows, session_dates, trading_calendar
from src.data_extract.utils.common.security_master import EXCLUDED, SOURCE_FTD
from src.data_extract.utils.common.sessions import last_completed_session
from src.data_extract.utils.common.symbol_tenure import normalise_market_symbol
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import load_fails_master, master_stamps
from src.data_extract.utils.institutionals.security_tape import (
    KEY_CHUNK,
    SUMMED_ROLES,
    TICKER_CHUNK,
    apply_grain,
    load_chunked,
    outside_scope,
    restamp_targets,
    stamp_changed,
    summed_lines,
    warn_lost_rows,
)
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker, squash

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
#: The CDN's first served date (a rolling window) and the one older file it still serves.
FIRST_SERVED_DAY = pd.Timestamp("2018-08-01")
EXTRA_SERVED_DAYS = (pd.Timestamp("2017-12-29"),)

RAW_COLUMNS = ("date", "source_symbol", "market", "short_volume", "short_exempt_volume", "total_volume")
#: Column order of `sec_short_volume_security`; the last four are stamped from the security master.
SECURITY_COLUMNS = (*RAW_COLUMNS, "security_id", "ticker", "lineage_role", "security_class")
#: Columns of the ticker-grain `sec_short_interest` (the consumers' schema).
TICKER_COLUMNS = ("date", "ticker", "short_volume", "total_volume")
_VOLUMES = ["short_volume", "short_exempt_volume", "total_volume"]

#: FINRA lower-case markers: their FTD spelling and the kind of line they mark.
_MARKER_SPELLING = {"p": "PR", "r": "RT", "w": "WI"}
_MARKER_KIND = {"p": "preferred", "r": "right", "w": "when_issued"}
#: Suffixes after a `/`: the kind of line they mark (a single class letter marks a common class).
_SUFFIX_KIND = (("WS", "warrant"), ("CL", "called"), ("U", "unit"))
_CLASS_SUFFIX = re.compile(r"^[A-TV-Z]$")

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure parse and symbology (unit-tested)                                        #
# --------------------------------------------------------------------------- #
def _parse_regsho(text: str) -> pd.DataFrame:
    """Every line of one RegSHO file, raw: the symbol as filed (case and separators kept), `Market`, the three volumes."""
    if not text or "|" not in text:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    if "Symbol" not in df.columns:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"].str.strip(), format="%Y%m%d", errors="coerce"),
            "source_symbol": df["Symbol"].str.strip(),
            "market": df["Market"].str.strip() if "Market" in df.columns else "",
            **{
                column: pd.to_numeric(df[name], errors="coerce")
                for column, name in zip(_VOLUMES, ("ShortVolume", "ShortExemptVolume", "TotalVolume"), strict=True)
            },
        }
    )
    out = out[out["date"].notna() & out["source_symbol"].ne("")]
    out = out.groupby(["date", "source_symbol"], as_index=False, sort=False).agg(
        market=("market", "first"),
        short_volume=("short_volume", "sum"),
        short_exempt_volume=("short_exempt_volume", "sum"),
        total_volume=("total_volume", "sum"),
    )
    return out[list(RAW_COLUMNS)].reset_index(drop=True)


def finra_key(symbol: str) -> str:
    """The FTD spelling of a FINRA symbol, the key `security_master` rows are found by (`BACpB` -> `BACPRB`, `BRK/A` -> `BRKA`)."""
    return squash("".join(_MARKER_SPELLING.get(char, char) for char in str(symbol)))


def finra_marker_kind(symbol: str) -> str | None:
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


def _class_letter(symbol: str) -> str | None:
    parts = str(symbol).split("/")
    return parts[1] if len(parts) == 2 and _CLASS_SUFFIX.match(parts[1]) else None


def _market_symbol(symbol: str) -> str:
    return normalise_market_symbol(str(symbol).upper().replace("/", "-"))


# --------------------------------------------------------------------------- #
# Stamp through the security master, aggregate to the ticker grain (pure)        #
# --------------------------------------------------------------------------- #
def master_keys(identity: Identity, universe: Collection[str]) -> frozenset[str]:
    """FTD keys of the master securities of `universe` companies."""
    requested = frozenset(normalise_ticker(t) for t in universe)
    return frozenset(
        key
        for (source, key), intervals in identity.securities_by_symbol.items()
        if source == SOURCE_FTD and any(hit.canonical_company in requested for _, _, hit in intervals)
    )


def _fallback(identity: Identity, symbol: str, day: pd.Timestamp) -> SecurityHit | None:
    """P21: the lineage tape interval of a window CIK inside its window, as an `S<cik>:<symbol>` security."""
    interval = identity.tape_interval(_market_symbol(symbol), day)
    company = None if interval is None else identity.ticker_by_entity.get(interval.entity)
    if interval is None or company is None:
        return None
    role = CANONICAL_CURRENT if identity.roster_cik.get(company) == interval.cik else CANONICAL_PREDECESSOR
    letter = _class_letter(symbol)
    return SecurityHit(f"S{interval.cik}:{_market_symbol(symbol)}", company, role, f"class_{letter}" if letter else "common", 1.0)


def _hit_column(hits: Sequence[SecurityHit | None], field: str, index: pd.Index) -> pd.Series:
    """One `SecurityHit` field per row (None where the row has no hit)."""
    return pd.Series([None if h is None else getattr(h, field) for h in hits], index=index, dtype=object)


def stamp_short_volume(lines: pd.DataFrame, identity: Identity, universe: Collection[str]) -> pd.DataFrame:
    """The rows of `lines` kept in `sec_short_volume_security`, with the four stamp columns and `conversion_ratio`.

    A symbol whose FTD key is a master security of a universe company is kept, stamped with the security covering
    its date (NULL stamps when none does, or when two symbols of that date share the key: a conflict); a symbol
    never seen in FTD is kept only through the P21 fallback; any other line is dropped. A line marked non-common by
    its own FINRA symbology is never summed.
    """
    requested = frozenset(normalise_ticker(t) for t in universe)
    known = master_keys(identity, requested)
    rows = lines.reset_index(drop=True).copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    keys = rows["source_symbol"].astype(str).map(finra_key)
    conflict = rows.groupby([rows["date"], keys])["source_symbol"].transform("nunique").gt(1)
    hits: list[SecurityHit | None] = []
    kept: list[bool] = []
    for symbol, day, key, clash in zip(rows["source_symbol"].astype(str), rows["date"], keys, conflict, strict=True):
        if key in known:
            hit = None if clash else identity.security_on(symbol=key, source=SOURCE_FTD, day=day)
            hits.append(hit if hit is not None and hit.canonical_company in requested else None)
            kept.append(True)
            continue
        hit = None if (clash or finra_marker_kind(symbol) or symbol != symbol.upper()) else _fallback(identity, symbol, day)
        hit = hit if hit is not None and hit.canonical_company in requested else None
        hits.append(hit)
        kept.append(hit is not None)
    out = rows.assign(
        security_id=_hit_column(hits, "security_id", rows.index),
        ticker=_hit_column(hits, "canonical_company", rows.index),
        lineage_role=_hit_column(hits, "lineage_role", rows.index),
        security_class=_hit_column(hits, "security_class", rows.index),
        conversion_ratio=_hit_column(hits, "conversion_ratio", rows.index).astype("float64"),
    )
    marked = out["source_symbol"].astype(str).map(finra_marker_kind)
    unsafe = marked.notna() & out["lineage_role"].isin(SUMMED_ROLES)
    out.loc[unsafe, "lineage_role"] = EXCLUDED
    out.loc[unsafe, "security_class"] = marked[unsafe]
    return out[pd.Series(kept, index=out.index)].reset_index(drop=True)


def ticker_rows(stamped: pd.DataFrame, tickers: Collection[str] | None = None) -> pd.DataFrame:
    """Ticker-grain rows: per (ticker, date) over the canonical and secondary-class lines, the volumes x conversion ratio."""
    summed = summed_lines(stamped, tickers)
    if summed.empty:
        return pd.DataFrame(columns=list(TICKER_COLUMNS))
    ratio = pd.to_numeric(summed["conversion_ratio"], errors="coerce").fillna(1.0)
    work = pd.DataFrame(
        {
            "date": pd.to_datetime(summed["date"]).dt.normalize(),
            "ticker": summed["ticker"].astype(str),
            "short_volume": pd.to_numeric(summed["short_volume"], errors="coerce") * ratio,
            "total_volume": pd.to_numeric(summed["total_volume"], errors="coerce") * ratio,
        }
    )
    return work.groupby(["ticker", "date"], as_index=False, sort=True)[["short_volume", "total_volume"]].sum()[list(TICKER_COLUMNS)]


def change_stamps(master: pd.DataFrame | None, identity: Identity) -> dict[str, pd.Timestamp]:
    """`{company: latest change}` over its master rows' `scope_changed_at` and its lineage symbol rows (the P21 fallback)."""
    stamps = {} if master is None or master.empty else master_stamps(master)
    for entity, stamp in identity.symbols_changed_at_by_entity.items():
        company = identity.ticker_by_entity.get(entity)
        if company is not None and pd.notna(stamp):
            stamps[company] = max(stamps.get(company, stamp), stamp)
    return stamps


# --------------------------------------------------------------------------- #
# IO                                                                             #
# --------------------------------------------------------------------------- #
def _fetch_day(
    day: pd.Timestamp,
    session: requests.Session | None = None,
) -> str | None:
    """Fetch one date, optionally reusing a run-scoped HTTP connection pool."""
    url = _URL.format(yyyymmdd=day.strftime(DATE_FORMAT_COMPACT))
    r = session.get(url, headers=BROWSER_HEADERS, timeout=30) if session is not None else requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    return r.text if r.status_code == 200 else None


def _missing_days(context: Context, calendar: pd.DatetimeIndex, floor: pd.Timestamp) -> pd.DatetimeIndex:
    """Calendar sessions inside the stored span (from `floor`) on which no key has a row: day files never stored."""
    stored = pd.DatetimeIndex(pd.to_datetime(context.store.distinct(Tables.short_interest, "date"))).normalize()
    if stored.empty:
        return pd.DatetimeIndex([])
    span = calendar[(calendar >= max(stored.min(), floor)) & (calendar <= stored.max())]
    return span.difference(stored)


def _plan_days(
    context: Context, tickers: list[str], years_history: int, full: bool, as_of: pd.Timestamp | None, *, repair: bool = False
) -> pd.DatetimeIndex:
    """The day files to read: every key's windows plus the never-stored days, as trading sessions (business
    days past the calendar). `repair` adds each key's own interior gaps (a one-time pass, not nightly).
    A new key's first `prices` date is its listing floor: a joiner listed after the FINRA start is read in full once."""
    run_date = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    until = last_completed_session(as_of)
    calendar = trading_calendar(context)
    work = series_windows(
        context,
        Tables.short_interest,
        tickers,
        run_date,
        until=until,
        years_history=years_history,
        full=full,
        calendar=calendar if repair else None,
        listing=Tables.prices,
    )
    days = _missing_days(context, calendar, document_floor(Tables.short_interest, run_date, years_history))
    for since, end, _keys in work.groups():
        days = days.union(session_dates(calendar, since, end))
    return days


def _stored_days(context: Context) -> set[pd.Timestamp]:
    """Every day stored in the raw table or in the ticker-grain table (which alone holds them before the first `full`)."""
    days: set[pd.Timestamp] = set()
    for table in (Tables.sec_short_volume_security, Tables.short_interest):
        if context.store.exists(table):
            days |= {pd.Timestamp(day).normalize() for day in context.store.distinct(table, "date")}
    return days


def _full_days(context: Context, today: pd.Timestamp) -> tuple[list[pd.Timestamp], set[pd.Timestamp]]:
    """`(days to fetch, stored days)`: every business day from `FIRST_SERVED_DAY`, the extra served files, and every
    day stored in either table."""
    stored = _stored_days(context)
    days = set(pd.bdate_range(FIRST_SERVED_DAY, today)) | set(EXTRA_SERVED_DAYS) | stored
    return sorted(day for day in days if day <= today), stored


def _company_rows(context: Context, identity: Identity, companies: Sequence[str]) -> pd.DataFrame:
    """Stored rows stamped with one of `companies`, or whose symbol's key is one of their master securities."""
    table = Tables.sec_short_volume_security
    keys = master_keys(identity, companies)
    symbols = sorted(s for s in map(str, context.store.distinct(table, "source_symbol")) if finra_key(s) in keys)
    kept = load_chunked(context, table, SECURITY_COLUMNS, "ticker", companies, TICKER_CHUNK)
    kept += load_chunked(context, table, SECURITY_COLUMNS, "source_symbol", symbols, KEY_CHUNK)
    if not kept:
        return pd.DataFrame(columns=list(SECURITY_COLUMNS))
    rows = pd.concat(kept, ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    return rows.drop_duplicates(["source_symbol", "date"], ignore_index=True)


def _delete_rows(context: Context, rows: pd.DataFrame) -> None:
    for symbol, group in rows.groupby("source_symbol", sort=True):
        for chunk in batched_tuples(sorted(group["date"]), KEY_CHUNK):
            context.store.delete(Tables.sec_short_volume_security, where={"source_symbol": str(symbol), "date": chunk})


def restamp_short_volume(
    context: Context,
    companies: Collection[str] | None,
    tickers: Collection[str],
    *,
    identity: Identity | None = None,
    stamps: Mapping[str, pd.Timestamp] | None = None,
    dry_run: bool = False,
    as_of: pd.Timestamp | None = None,
) -> list[dict]:
    """Re-stamp the stored rows of `companies` and rebuild their ticker rows, writing only what changed.

    `companies` None: `security_tape.restamp_targets` (master or lineage symbol rows changed recently on `as_of`,
    default today, or all vanished). Rows that no longer resolve to a kept security are deleted. A scoped run (`tickers`
    short of the universe) never deletes or re-stamps a row stamped, before or after, with a universe company
    outside it. Returns one record per ticker whose ticker rows vanished. A security newly in scope needs
    `short-interest --full` (rows are not cached).
    """
    table = Tables.sec_short_volume_security
    if not context.store.exists(table):
        return []
    resolver = identity or load_identity(context)
    if companies is None:
        stamps = change_stamps(load_fails_master(context), resolver) if stamps is None else stamps
        companies = restamp_targets(context, table, stamps, as_of)
    if not companies:
        return []
    universe = frozenset(normalise_ticker(t) for t in tickers)
    others = outside_scope(context, universe)
    names = sorted(set(companies) - others)
    if not names:
        return []
    stored = _company_rows(context, resolver, names)
    if stored.empty:
        return []
    fresh = stamp_short_volume(stored[list(RAW_COLUMNS)], resolver, universe | others)
    merged = stored.merge(fresh, on=["source_symbol", "date"], how="left", suffixes=("_old", ""), indicator=True)
    held = merged["ticker_old"].isin(others) | merged["ticker"].isin(others)
    gone = merged[merged["_merge"].eq("left_only") & ~held]
    both = merged[merged["_merge"].eq("both") & ~held]
    changed = stamp_changed(both, both, old_suffix="_old")
    if not dry_run:
        if not gone.empty:
            _delete_rows(context, gone)
        if changed.any():
            context.store.save(table, both.loc[changed, list(SECURITY_COLUMNS)])
    logger.info(
        "RegSHO: %d row(s) of %d company(ies) re-stamped, %d left scope%s", int(changed.sum()), len(names), len(gone), " (dry run)" if dry_run else ""
    )
    rebuilt = [name for name in names if name in universe]
    if not rebuilt:
        return []
    grain = ticker_rows(fresh, rebuilt)
    return apply_grain(context, rebuilt, grain, dry_run=dry_run, table=Tables.short_interest, columns=TICKER_COLUMNS)


def _log_stamps(stamped: pd.DataFrame) -> None:
    roles = stamped["lineage_role"].fillna("unresolved").value_counts().sort_index().to_dict()
    fallback = stamped["security_id"].fillna("").str.startswith("S")
    unresolved = stamped.loc[stamped["lineage_role"].isna(), "source_symbol"]
    logger.info("RegSHO: %d row(s) kept, by role %s; %d through the P21 lineage fallback", len(stamped), roles, int(fallback.sum()))
    if not unresolved.empty:
        counts = unresolved.value_counts().head(20)
        logger.warning(
            "RegSHO: %d row(s) of known securities resolve on no date (kept, never summed): %s",
            len(unresolved),
            ", ".join(f"{s} {n}" for s, n in counts.items()),
        )


def fetch_short_interest(
    context: Context,
    tickers: list[str],
    years_history: int = 15,
    pause: float = 0.05,
    full: bool = False,
    identity: Identity | None = None,
    as_of: pd.Timestamp | None = None,
    repair: bool = False,
) -> None:
    """Fetch the RegSHO day files, store their in-scope lines stamped, and rebuild `short_interest` for those days.

    Incremental: the `_plan_days` work list (`repair` adds each key's interior gaps), then a re-stamp of the
    companies whose master or lineage symbol rows changed recently. `full`: every served day (and every stored
    one); unscoped, any stored day not served or any fetch error aborts before the two tables are replaced;
    scoped (`-t`), the tickers' rows are upserted and no other key is touched.
    """
    universe = frozenset(normalise_ticker(ticker) for ticker in tickers)
    others = outside_scope(context, universe)
    scoped = bool(others)
    if full:
        days, stored_days = _full_days(context, pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize())
    else:
        days, stored_days = list(_plan_days(context, sorted(universe), years_history, False, as_of, repair=repair)), set()
    resolver = identity or load_identity(context)
    known, fallback = master_keys(resolver, universe), resolver.universe_symbols(universe)
    logger.info(f"Fetching {len(days)} RegSHO day-file(s) for {len(universe)} tickers")

    frames: list[pd.DataFrame] = []
    successful: list[pd.Timestamp] = []
    missing: list[pd.Timestamp] = []
    errors: list[str] = []
    session = requests.Session()
    for day in tqdm(days, "short_interest OffExchange - fetch RegSHO"):
        try:
            text = _fetch_day(day, session)
        except Exception as e:  # incremental: one bad day must not abort the run
            logger.error(f"RegSHO {day.date()} failed: {e}")
            errors.append(f"{day.date()} ({e})")
            continue
        if not text:
            missing.append(day)
            continue
        successful.append(day)
        lines = _parse_regsho(text)
        in_scope = lines["source_symbol"].map(finra_key).isin(known) | lines["source_symbol"].map(_market_symbol).isin(fallback)
        if in_scope.any():
            frames.append(lines[in_scope])
        time.sleep(pause)
    session.close()

    if full and not scoped:
        lost = sorted(day.date().isoformat() for day in set(missing) & stored_days)
        if errors or lost or not successful:
            raise RuntimeError(
                f"RegSHO full: {len(errors)} fetch error(s) {errors[:5]}, {len(lost)} stored date(s) not served {lost[:20]}, "
                f"{len(successful)} served; aborting before any write"
            )
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(RAW_COLUMNS))
    stamped = stamp_short_volume(raw, resolver, universe | others) if not raw.empty else pd.DataFrame(columns=[*SECURITY_COLUMNS, "conversion_ratio"])
    stamped = stamped[~stamped["ticker"].isin(others)]
    grain = ticker_rows(stamped, universe)
    if not stamped.empty:
        _log_stamps(stamped)

    if full and not scoped:
        if grain.empty or stamped.duplicated(["source_symbol", "date"]).any():
            raise ValueError("RegSHO full staged no ticker row, or a duplicate (source_symbol, date) key; aborting before any write")
        context.store.replace(Tables.sec_short_volume_security, stamped[list(SECURITY_COLUMNS)])
        written = context.store.replace(Tables.short_interest, grain)
        logger.info(f"RegSHO full: {len(successful)} served day(s), {len(stamped)} raw row(s), {written} ticker row(s); no legacy row kept")
        return

    written = 0
    if not stamped.empty:
        context.store.save(Tables.sec_short_volume_security, stamped[list(SECURITY_COLUMNS)])
    if successful and context.store.exists(Tables.short_interest):
        for chunk in batched_tuples(sorted(universe), TICKER_CHUNK):
            context.store.delete(Tables.short_interest, where={"ticker": chunk, "date": successful})
    if not grain.empty:
        written = context.store.save(Tables.short_interest, grain)
    logger.info(f"Saved {written} short-volume ticker row(s) over {len(successful)} day(s) to '{Tables.short_interest}'")
    warn_lost_rows(logger, "RegSHO", restamp_short_volume(context, None, universe, identity=resolver, as_of=as_of))
