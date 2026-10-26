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
from typing import cast

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.constants.constants import BROWSER_HEADERS, CANONICAL_CURRENT, CANONICAL_PREDECESSOR, DATE_FORMAT_COMPACT
from src.context import Context
from src.data_extract.utils.common.identity import Identity, SecurityHit, load_identity
from src.data_extract.utils.common.identity_store import SOURCE_LOCK, assert_source_identity, load_source_identity
from src.data_extract.utils.common.resume import document_floor, series_windows, session_dates, trading_calendar
from src.data_extract.utils.common.security_master import EXCLUDED, SOURCE_FTD
from src.data_extract.utils.common.sessions import last_completed_session
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import load_fails_master, master_stamps
from src.data_extract.utils.institutionals.security_tape import (
    SUMMED_ROLES,
    assert_tape_owners,
    assert_tape_source,
    commit_tape_source,
    load_tape_cohorts,
    outside_scope,
    restamp_targets,
    scope_tape_rows,
    stamp_changed,
    summed_lines,
    tape_key_matches,
    validate_tape_frames,
    warn_lost_rows,
)
from src.data_store.schema import Resume, Table, Tables
from src.data_store.store import DataStore, Where
from src.utils.filer_tables import filing_window
from src.utils.string import squash, yahoo_symbol

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"

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
    """Validate the entire served file before retaining its raw symbols and quantities."""
    lines = text.strip().splitlines()
    header = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
    legacy = "Date|Symbol|ShortVolume|TotalVolume|Market"
    supplied = lines[0].strip().lstrip("\ufeff") if lines else ""
    if len(lines) < 2 or supplied not in (header, legacy) or not lines[-1].strip().isdigit():
        raise ValueError("RegSHO missing header or record-count trailer")
    records = lines[1:-1]
    if int(lines[-1]) != len(records) or any(len(row.split("|")) != len(supplied.split("|")) for row in records):
        raise ValueError("RegSHO record-count or line-width mismatch")
    df = pd.read_csv(io.StringIO("\n".join([supplied, *records])), sep="|", dtype=str, keep_default_na=False)
    if "ShortExemptVolume" not in df:
        df["ShortExemptVolume"] = pd.NA
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
    validated = _VOLUMES if supplied == header else ["short_volume", "total_volume"]
    quantities = out[validated].to_numpy(dtype=float)
    if out["date"].isna().any() or out["source_symbol"].eq("").any() or not np.isfinite(quantities).all() or (quantities < 0).any():
        raise ValueError("RegSHO invalid date, symbol or finite nonnegative quantity")
    if out["short_volume"].gt(out["total_volume"]).any():
        raise ValueError("RegSHO short volume exceeds reported total volume")
    if supplied == header and out["short_exempt_volume"].gt(out["short_volume"]).any():
        raise ValueError("RegSHO exempt volume exceeds its short-volume total")
    out = out.groupby(["date", "source_symbol"], as_index=False, sort=False).agg(
        market=("market", "first"),
        short_volume=("short_volume", "sum"),
        short_exempt_volume=("short_exempt_volume", "sum"),
        total_volume=("total_volume", "sum"),
    )
    if supplied == legacy:
        out["short_exempt_volume"] = np.nan
    if not np.isfinite(out[validated].to_numpy(dtype=float)).all():
        raise ValueError("RegSHO grouped source quantities overflowed")
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


# --------------------------------------------------------------------------- #
# Stamp through the security master, aggregate to the ticker grain (pure)        #
# --------------------------------------------------------------------------- #
def master_keys(identity: Identity, universe: Collection[str]) -> frozenset[str]:
    """FTD keys of the master securities of `universe` companies."""
    return frozenset(
        key
        for (source, key), intervals in identity.securities_by_symbol.items()
        if source == SOURCE_FTD and any(hit.canonical_company in universe for _, _, hit in intervals)
    )


def _fallback(identity: Identity, symbol: str, day: pd.Timestamp) -> SecurityHit | None:
    """P21: the lineage tape interval of a window CIK inside its window, as an `S<cik>:<symbol>` security."""
    interval = identity.tape_interval(symbol, day)
    company = None if interval is None else identity.ticker_by_entity.get(interval.entity)
    if interval is None or company is None:
        return None
    role = CANONICAL_CURRENT if identity.roster_cik.get(company) == interval.cik else CANONICAL_PREDECESSOR
    letter = _class_letter(symbol)
    return SecurityHit(f"S{interval.cik}:{yahoo_symbol(symbol)}", company, role, f"class_{letter}" if letter else "common", 1.0)


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
    requested = frozenset(universe)
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
    ratio = pd.to_numeric(summed["conversion_ratio"], errors="coerce")
    if not (np.isfinite(ratio) & ratio.gt(0)).all():
        raise ValueError("RegSHO needs finite positive class conversion ratios")
    work = pd.DataFrame(
        {
            "date": pd.to_datetime(summed["date"]).dt.normalize(),
            "ticker": summed["ticker"].astype(str),
            "short_volume": pd.to_numeric(summed["short_volume"], errors="coerce") * ratio,
            "total_volume": pd.to_numeric(summed["total_volume"], errors="coerce") * ratio,
        }
    )
    result = work.groupby(["ticker", "date"], as_index=False, sort=True)[["short_volume", "total_volume"]].sum()[list(TICKER_COLUMNS)]
    if not np.isfinite(result[["short_volume", "total_volume"]].to_numpy(dtype=float)).all():
        raise ValueError("RegSHO class-converted aggregate quantities overflowed")
    return result


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


def _missing_days(calendar: pd.DatetimeIndex, floor: pd.Timestamp, known: set[pd.Timestamp], pending: set[pd.Timestamp]) -> pd.DatetimeIndex:
    """Never-stored global days plus scoped NULL quantities whose retained keys still require source."""
    stored = pd.DatetimeIndex(sorted(known))
    if stored.empty:
        return pd.DatetimeIndex([])
    span = calendar[(calendar >= max(stored.min(), floor)) & (calendar <= stored.max())]
    return span.difference(stored).union(pd.DatetimeIndex(sorted(pending)))


def _plan_days(
    context: Context, tickers: list[str], years_history: int, full: bool, as_of: pd.Timestamp | None, *, repair: bool = False
) -> pd.DatetimeIndex:
    """The day files to read: every key's windows plus the never-stored days, as trading sessions (business
    days past the calendar). `repair` adds each key's own interior gaps (a one-time pass, not nightly).
    A new key's first `prices` date is its listing floor: a joiner listed after the FINRA start is read in full once."""
    run_date = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    floor = document_floor(Tables.short_interest, run_date, years_history)
    until = last_completed_session(as_of)
    calendar = trading_calendar(context)
    known, pending = short_source_days(context.store, tickers)
    df_stats = context.store.key_stats(Tables.short_interest, "ticker", where={"ticker": tickers})
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
        stored=(df_stats, max(known) if known else None),
    )
    days = _missing_days(calendar, floor, known, pending)
    for since, end, _keys in work.groups():
        days = days.union(session_dates(calendar, since, end))
    return days.intersection(session_dates(calendar, floor, until))


def _stored_days(context: Context) -> set[pd.Timestamp]:
    """Every day stored in the raw table or in the ticker-grain table (which alone holds them before the first `full`)."""
    days: set[pd.Timestamp] = set()
    for table in (Tables.sec_short_volume_security, Tables.short_interest):
        if context.store.exists(table):
            days |= {pd.Timestamp(day).normalize() for day in context.store.distinct(table, "date")}
    return days


def _full_days(context: Context, today: pd.Timestamp) -> tuple[list[pd.Timestamp], set[pd.Timestamp]]:
    """`(days to fetch, stored days)`: every business day from the table's source start, and every
    day stored in either table."""
    stored = _stored_days(context)
    days = set(pd.bdate_range(cast(Resume, Tables.short_interest.resume).source_start, today)) | stored
    return sorted(day for day in days if day <= today), stored


def _company_rows(context: Context, identity: Identity, companies: Sequence[str]) -> pd.DataFrame:
    """Stored rows stamped with one of `companies`, or whose symbol's key is one of their master securities."""
    table = Tables.sec_short_volume_security
    keys = master_keys(identity, companies)
    symbols = sorted(s for s in map(str, context.store.distinct(table, "source_symbol")) if finra_key(s) in keys)
    rows = context.store.load_cohort(
        table, "source_symbol", [{"ticker": list(companies)}, {"source_symbol": symbols}], columns=context.store.columns(table)
    )
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    return rows


def _retain_short_source_proof(grain: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Carry only unchanged existing joint source observations under the guarded source/identity cohorts."""
    old = previous.reindex(columns=[*TICKER_COLUMNS, "source_checked_at"]).copy()
    old["date"] = pd.to_datetime(old["date"])
    incoming = grain.copy()
    incoming["date"] = pd.to_datetime(incoming["date"])
    merged = incoming.merge(old, on=["ticker", "date"], how="left", suffixes=("", "_old"))
    same = pd.Series(True, index=merged.index)
    for column in ("short_volume", "total_volume"):
        same &= merged[column].eq(merged[f"{column}_old"]) | (merged[column].isna() & merged[f"{column}_old"].isna())
    merged["source_checked_at"] = pd.to_datetime(merged["source_checked_at"]).where(same)
    result = merged[[*TICKER_COLUMNS, "source_checked_at"]]
    quiet = old.loc[old["source_checked_at"].notna() & old["short_volume"].eq(0) & old["total_volume"].isna()]
    quiet = quiet.loc[~pd.MultiIndex.from_frame(quiet[["ticker", "date"]]).isin(pd.MultiIndex.from_frame(result[["ticker", "date"]]))]
    return pd.concat([result, quiet], ignore_index=True)


def _short_grain_losses(previous: pd.DataFrame, grain: pd.DataFrame) -> list[dict]:
    old = previous.copy()
    old["date"] = pd.to_datetime(old["date"])
    missing = old.loc[~pd.MultiIndex.from_frame(old[["ticker", "date"]]).isin(pd.MultiIndex.from_frame(grain[["ticker", "date"]]))]
    records = []
    for ticker, group in missing.groupby("ticker", sort=True):
        first, last = filing_window(group["date"])
        records.append(
            {
                "table": Tables.short_interest.name,
                "ticker": str(ticker),
                "cik": "",
                "first_filed": first,
                "last_filed": last,
                "keys": len(group),
                "rows": len(group),
            }
        )
    return records


def _same_short_cohorts(stored: pd.DataFrame, payload: pd.DataFrame, previous: pd.DataFrame, grain: pd.DataFrame) -> bool:
    for left, right, keys in ((stored, payload, list(Tables.sec_short_volume_security.pk)), (previous, grain, list(Tables.short_interest.pk))):
        if (set(left.columns) ^ set(right.columns)) - {"source_checked_at"}:
            return False
        columns = sorted(set(left.columns) | set(right.columns))
        left, right = left.reindex(columns=columns), right.reindex(columns=columns)
        left["date"], right["date"] = pd.to_datetime(left["date"]), pd.to_datetime(right["date"])
        left, right = left.sort_values(keys), right.sort_values(keys)
        left, right = left.astype(object).where(left.notna(), None), right.astype(object).where(right.notna(), None)
        if left.to_dict("records") != right.to_dict("records"):
            return False
    return True


def restamp_short_volume(
    context: Context,
    companies: Collection[str] | None,
    tickers: Collection[str],
    *,
    identity: Identity | None = None,
    stamps: Mapping[str, pd.Timestamp] | None = None,
    dry_run: bool = False,
    as_of: pd.Timestamp | None = None,
    others: frozenset[str] | None = None,
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
    universe = frozenset(tickers)
    others = outside_scope(context, universe) if others is None else others
    names = sorted(set(companies) - others)
    if not names:
        return []
    stored = _company_rows(context, resolver, names)
    rebuilt = [name for name in names if name in universe]
    previous_columns = context.store.columns(Tables.short_interest)
    previous = context.store.load(Tables.short_interest, columns=previous_columns, where={"ticker": rebuilt}, optional=True) if rebuilt else None
    if stored.empty:
        return []
    fresh = stamp_short_volume(stored[list(RAW_COLUMNS)], resolver, universe | others)
    merged = stored.merge(fresh, on=["source_symbol", "date"], how="left", suffixes=("_old", ""), indicator=True)
    held = merged["ticker_old"].isin(others) | merged["ticker"].isin(others)
    gone = merged[merged["_merge"].eq("left_only") & ~held]
    both = merged[merged["_merge"].eq("both") & ~held]
    changed = stamp_changed(both, both, old_suffix="_old")
    logger.info(
        "RegSHO: %d row(s) of %d company(ies) re-stamped, %d left scope%s", int(changed.sum()), len(names), len(gone), " (dry run)" if dry_run else ""
    )
    if not rebuilt:
        return []
    old_grain = previous if previous is not None else pd.DataFrame(columns=list(TICKER_COLUMNS))
    grain = _retain_short_source_proof(ticker_rows(fresh.loc[~fresh["ticker"].isin(others)], rebuilt), old_grain)
    records = _short_grain_losses(old_grain, grain)
    retained = merged.loc[held, [f"{column}_old" if column not in ("source_symbol", "date") else column for column in SECURITY_COLUMNS]]
    retained.columns = list(SECURITY_COLUMNS)
    payload = pd.concat([fresh.loc[~fresh["ticker"].isin(others), list(SECURITY_COLUMNS)], retained], ignore_index=True)
    if not dry_run and not _same_short_cohorts(stored, payload, old_grain, grain):
        expected_grain = old_grain.reindex(columns=list(dict.fromkeys([*previous_columns, *grain.columns])))
        replace_short_source(
            context.store,
            {table: payload, Tables.short_interest: grain},
            {table: {"source_symbol": sorted(set(stored["source_symbol"]))}, Tables.short_interest: {"ticker": rebuilt}},
            resolver.source_cohort,
            {table: stored, Tables.short_interest: expected_grain},
        )
    return records


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


def _validated_regsho_day(day: pd.Timestamp, session: requests.Session) -> pd.DataFrame | None:
    text = _fetch_day(day, session)
    if not text:
        return None
    lines = _parse_regsho(text)
    if not lines["date"].eq(day.normalize()).all():
        raise ValueError("RegSHO source date does not match requested day")
    return lines


def _read_regsho_days(
    days: Sequence[pd.Timestamp], identity: Identity, universe: Collection[str], pause: float
) -> tuple[pd.DataFrame, list[pd.Timestamp], list[pd.Timestamp], list[str]]:
    """Stage fully validated source units; failed units never enter the replacement cohort."""
    known, fallback = master_keys(identity, universe), identity.universe_symbols(universe)
    frames: list[pd.DataFrame] = []
    successful: list[pd.Timestamp] = []
    missing: list[pd.Timestamp] = []
    errors: list[str] = []
    with requests.Session() as session:
        for day in tqdm(days, "short_interest OffExchange - fetch RegSHO"):
            try:
                lines = _validated_regsho_day(day, session)
            except Exception as exc:
                logger.error("RegSHO %s failed: %s", day.date(), exc)
                errors.append(f"{day.date()} ({exc})")
                continue
            if lines is None:
                missing.append(day)
                continue
            successful.append(day)
            in_scope = lines["source_symbol"].map(finra_key).isin(known) | lines["source_symbol"].map(yahoo_symbol).isin(fallback)
            frames.append(lines.loc[in_scope])
            time.sleep(pause)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(RAW_COLUMNS))
    return raw, successful, missing, errors


def _incomplete_regsho_keys(stamped: pd.DataFrame, identity: Identity) -> set[tuple[str, pd.Timestamp]]:
    """Any relevant unresolved raw security prevents a complete ordinary-company aggregate."""
    incomplete = set()
    for row in stamped.loc[stamped["lineage_role"].isna()].itertuples(index=False):
        if finra_marker_kind(str(row.source_symbol)) is not None:
            continue
        intervals = identity.securities_by_symbol.get((SOURCE_FTD, finra_key(str(row.source_symbol))), ())
        for _, _, hit in intervals:
            if hit.canonical_company is not None and hit.lineage_role in SUMMED_ROLES:
                incomplete.add((hit.canonical_company, cast(pd.Timestamp, row.date)))
    return incomplete


def _invalid_regsho_basis(identity: Identity, days: Sequence[pd.Timestamp], universe: Collection[str]) -> set[tuple[str, pd.Timestamp]]:
    """An unknown class conversion cannot establish a checked company absence."""
    intervals = [item for (source, _), values in identity.securities_by_symbol.items() if source == SOURCE_FTD for item in values]
    invalid = set()
    for start, end, hit in intervals:
        if (
            hit.canonical_company in universe
            and hit.lineage_role in SUMMED_ROLES
            and (not np.isfinite(hit.conversion_ratio) or hit.conversion_ratio <= 0)
        ):
            invalid.update((hit.canonical_company, day) for day in days if day >= start and (end is None or day < end))
    return invalid


def _checked_regsho_grain(stamped: pd.DataFrame, days: Sequence[pd.Timestamp], identity: Identity, universe: Collection[str]) -> pd.DataFrame:
    """A checked mapped absence has zero short flow and an unavailable total-volume denominator."""
    grain = ticker_rows(stamped, universe)
    candidates = pd.MultiIndex.from_product([sorted(universe), days], names=["ticker", "date"]).to_frame(index=False)
    symbols = identity.universe_symbols(universe)
    mappings = [
        (symbol, day, identity.security_on(symbol=finra_key(symbol), source=SOURCE_FTD, day=day) or _fallback(identity, symbol, day))
        for symbol in symbols
        for day in days
    ]
    supported = {(hit.canonical_company, day) for _, day, hit in mappings if hit is not None and hit.lineage_role in SUMMED_ROLES}
    supported.update(grain[["ticker", "date"]].itertuples(index=False, name=None))
    supported -= _incomplete_regsho_keys(stamped, identity)
    supported -= _invalid_regsho_basis(identity, days, universe)
    candidates = candidates.loc[[(ticker, day) in supported for ticker, day in candidates.itertuples(index=False, name=None)]]
    grain = grain.loc[[(ticker, day) in supported for ticker, day in grain[["ticker", "date"]].itertuples(index=False, name=None)]]
    if not candidates.empty:
        candidates = candidates.merge(grain, on=["ticker", "date"], how="left")
        candidates["short_volume"] = candidates["short_volume"].fillna(0.0)
        grain = candidates
    return grain.astype({"short_volume": "float64", "total_volume": "float64"}).assign(source_checked_at=pd.Timestamp.now(tz="UTC").tz_localize(None))


def fetch_short_interest(
    context: Context,
    tickers: list[str],
    years_history: int = 31,
    pause: float = 0.05,
    full: bool = False,
    identity: Identity | None = None,
    as_of: pd.Timestamp | None = None,
    repair: bool = False,
    *,
    all_tickers: Collection[str] | None = None,
) -> None:
    """Fetch the RegSHO day files, store their in-scope lines stamped, and rebuild `short_interest` for those days.

    Incremental: the `_plan_days` work list (`repair` adds each key's interior gaps), then a re-stamp of the
    companies whose master or lineage symbol rows changed recently. `full`: every served day (and every stored
    one); unscoped, any stored day not served or any fetch error aborts before the two tables are replaced;
    scoped (`-t`), the tickers' rows are upserted and no other key is touched.
    """
    universe = frozenset(tickers)
    others = outside_scope(context, universe, all_tickers)
    scoped = bool(others)
    if full:
        days, stored_days = _full_days(context, pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize())
    else:
        days, stored_days = list(_plan_days(context, sorted(universe), years_history, False, as_of, repair=repair)), set()
    resolver = identity or load_identity(context)
    logger.info(f"Fetching {len(days)} RegSHO day-file(s) for {len(universe)} tickers")
    raw, successful, missing, errors = _read_regsho_days(days, resolver, universe, pause)

    if full and not scoped:
        lost = sorted(day.date().isoformat() for day in set(missing) & stored_days)
        if errors or lost or not successful:
            raise RuntimeError(
                f"RegSHO full: {len(errors)} fetch error(s) {errors[:5]}, {len(lost)} stored date(s) not served {lost[:20]}, "
                f"{len(successful)} served; aborting before any write"
            )
    stamped = stamp_short_volume(raw, resolver, universe | others) if not raw.empty else pd.DataFrame(columns=[*SECURITY_COLUMNS, "conversion_ratio"])
    stamped = stamped.astype(dict.fromkeys(_VOLUMES, "float64"))
    stamped = stamped[~stamped["ticker"].isin(others)]
    grain = _checked_regsho_grain(stamped, successful, resolver, universe)
    if not stamped.empty:
        _log_stamps(stamped)

    if full and not scoped:
        if grain.empty or stamped.duplicated(["source_symbol", "date"]).any():
            raise ValueError("RegSHO full staged no ticker row, or a duplicate (source_symbol, date) key; aborting before any write")
        counts = replace_short_source(
            context.store,
            {Tables.sec_short_volume_security: stamped[list(SECURITY_COLUMNS)], Tables.short_interest: grain},
            {Tables.sec_short_volume_security: None, Tables.short_interest: None},
            resolver.source_cohort,
        )
        written = counts[Tables.short_interest]
        logger.info(f"RegSHO full: {len(successful)} served day(s), {len(stamped)} raw row(s), {written} ticker row(s); no legacy row kept")
        return

    written = 0
    if successful:
        counts = replace_short_source(
            context.store,
            {Tables.sec_short_volume_security: stamped[list(SECURITY_COLUMNS)], Tables.short_interest: grain},
            {
                Tables.sec_short_volume_security: {"ticker": sorted(universe), "date": successful},
                Tables.short_interest: {"ticker": sorted(universe), "date": successful},
            },
            resolver.source_cohort,
        )
        written = counts[Tables.short_interest]

    logger.info(f"Saved {written} short-volume ticker row(s) over {len(successful)} day(s) to '{Tables.short_interest}'")
    warn_lost_rows(logger, "RegSHO", restamp_short_volume(context, None, universe, identity=resolver, as_of=as_of, others=others))


def short_source_days(store: DataStore, tickers: Sequence[str]) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    """One compact date summary retains global presence and scoped unknown-volume retries."""
    unknown = {"short_volume": None} if "short_volume" in store.columns(Tables.short_interest) else {}
    df_days = store.summarize(
        Tables.short_interest,
        "date",
        {"retry": ("count", None)},
        count_where={"retry": {"ticker": list(tickers), **unknown}},
        where={"date": store.NOT_NULL},
    )
    dates = pd.to_datetime(df_days["date"]).dt.normalize()
    return set(dates), set(dates.loc[df_days["retry"].gt(0)])


def replace_short_source(
    store: DataStore,
    frames: dict[Table, pd.DataFrame],
    scopes: dict[Table, dict[str, object] | None],
    expected_identity: dict[Table, pd.DataFrame] | None = None,
    expected_source: dict[Table, pd.DataFrame] | None = None,
) -> dict[Table, int]:
    """Atomically publish a FINRA raw/derived pair after fresh identity and retained-owner checks."""
    raw, grain = Tables.sec_short_volume_security, Tables.short_interest
    if set(frames) != {raw, grain}:
        raise ValueError("FINRA writes require their raw and ticker-grain pair")
    validate_tape_frames(frames, scopes)
    with store.transaction(SOURCE_LOCK) as bound:
        identity = load_source_identity(bound) if expected_identity is not None else None
        queries: dict[Table, Where | None] = {}
        for table, scope in scopes.items():
            queries[table] = [scope, frames[table][list(table.pk)]] if scope is not None else None
        current = load_tape_cohorts(bound, queries)

        if expected_identity is not None and identity is not None:
            assert_source_identity(bound, identity, expected_identity)
        assert_tape_owners(tape_key_matches(raw, frames[raw], current[raw]), (scopes[raw] or {}).get("ticker"))
        old = {table: scope_tape_rows(table, current[table], scopes[table]) for table in frames}
        assert_tape_source(bound, old, expected_source)
        return commit_tape_source(bound, frames, scopes)
