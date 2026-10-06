"""
fetch_fails_to_deliver.py (src/data_extract/utils/institutionals/fetch_fails_to_deliver.py)
------------------------------------------------------------------------------------
SEC Fails-to-Deliver semi-monthly ZIPs. `download_fails_to_deliver` (the `ftd-download` stage) caches
the ZIPs and stores the in-scope source lines raw, per CUSIP, in `sec_fails_to_deliver_security`.
`fetch_fails_to_deliver` stamps those lines from `security_master` (security, canonical company,
lineage role, class) and rebuilds the ticker-grain `sec_fails_to_deliver`, its own table so its
publication lag never moves `short_interest`'s frontier: per (ticker, settlement date) the sum over
the canonical and secondary-class lines of quantity x conversion ratio, and of quantity x the file's
PRICE (NULL when a summed line has no price). Values are cumulative net unsettled balances, not new
fails. The download reads the periods no raw line carries (`resume.archive_periods`), plus every cached
period for the symbols and CUSIP-6 prefixes whose lineage or master rows changed recently; `full`
rebuilds both tables from every cached ZIP.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Collection, Sequence
from itertools import batched
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants.constants import FTD_ZIP_NAME_TEMPLATE
from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir, cached_periods, ensure_zip, read_zip_text
from src.data_extract.utils.common.resume import archive_periods, recently_changed
from src.data_extract.utils.common.security_master import (
    SOURCE_FTD,
    cusip_votes,
    lineage_scope_symbols,
    load_security_manual,
    squash,
    trade_dates,
)
from src.data_extract.utils.institutionals.security_tape import (
    KEY_CHUNK,
    TICKER_CHUNK,
    apply_grain,
    load_chunked,
    nullable,
    outside_scope,
    restamp_targets,
    stamp_changed,
    summed_lines,
    warn_lost_rows,
)
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker

logger = logging.getLogger(__name__)

#: Columns of the ticker-grain `sec_fails_to_deliver` (the consumers' schema).
TICKER_COLUMNS = ("ticker", "date", "fails_quantity", "fails_value", "period")
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
#: The `security_master` columns read for stamping.
_MASTER_COLUMNS = [
    "security_id",
    "canonical_company",
    "cusip",
    "source_symbol",
    "security_class",
    "conversion_ratio",
    "lineage_role",
    "valid_from",
    "valid_to",
    "scope_changed_at",
]
_LINEAGE_SCOPE_COLUMNS = ["entity_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources"]
#: The open end of a master interval.
_FAR = pd.Timestamp("2262-01-01")

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
def _ftd_column(df: pd.DataFrame, cols: dict[str, str], name: str) -> pd.Series:
    """The stripped text of the file column headed `name` (any case), or an all-NA column when the file lacks it."""
    return df[cols[name]].astype("string").str.strip() if name in cols else pd.Series(pd.NA, index=df.index, dtype="string")


def _parse_ftd_lines(raw: str) -> pd.DataFrame:
    """Every source line of one FTD file, raw: symbol as filed, description, PRICE ('.' -> NULL), trade date.

    Exact duplicate lines collapse to one; nothing is summed.
    """
    columns = list(SECURITY_COLUMNS[:8])
    if not raw or "|" not in raw:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, engine="python", on_bad_lines="skip", quoting=3)
    cols = {c.strip().upper(): c for c in df.columns}
    price = pd.to_numeric(_ftd_column(df, cols, "PRICE").where(lambda s: s != "."), errors="coerce")
    qty = pd.to_numeric(_ftd_column(df, cols, "QUANTITY (FAILS)"), errors="coerce")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(_ftd_column(df, cols, "SETTLEMENT DATE"), format="%Y%m%d", errors="coerce"),
            "cusip": _ftd_column(df, cols, "CUSIP").str.upper(),
            "source_symbol": _ftd_column(df, cols, "SYMBOL").str.upper(),
            "description": _ftd_column(df, cols, "DESCRIPTION"),
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
    return cached_periods(cache, prefix=SEC_FTD_ZIP_PREFIX)


# --------------------------------------------------------------------------- #
# Stamp from the security master, aggregate to the ticker grain (pure)           #
# --------------------------------------------------------------------------- #
_HIT_COLUMNS = ["canonical_company", "lineage_role", "security_class", "conversion_ratio"]


def prepare_master(master: pd.DataFrame) -> pd.DataFrame:
    """The FTD rows of `security_master`, keyed for stamping: upper-case CUSIP and symbol, open ends at a far date."""
    rows = master[master["source"].eq(SOURCE_FTD)] if "source" in master.columns else master
    out = rows[_MASTER_COLUMNS].copy()
    out["cusip"] = out["cusip"].astype(str).str.strip().str.upper()
    out["source_symbol"] = out["source_symbol"].fillna("").astype(str).str.strip().str.upper()
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"]).fillna(_FAR)
    out["conversion_ratio"] = pd.to_numeric(out["conversion_ratio"]).fillna(1.0).astype("float64")
    company = out["canonical_company"].astype(object)
    out["canonical_company"] = company.where(company.notna() & company.astype(str).ne(""), None)
    out["scope_changed_at"] = pd.to_datetime(out["scope_changed_at"], errors="coerce")
    return out.reset_index(drop=True)


def master_stamps(master: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """`{canonical company: latest scope_changed_at}` over its master rows."""
    stamps = master.dropna(subset=["canonical_company"]).groupby("canonical_company")["scope_changed_at"].max()
    return {str(company): pd.Timestamp(stamp) for company, stamp in stamps.items() if pd.notna(stamp)}


def _covering(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["trade_date"] >= frame["valid_from"]) & (frame["trade_date"] < frame["valid_to"])]


def stamp_lines(lines: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """`lines` with `security_id`, `ticker`, `lineage_role`, `security_class` and `conversion_ratio`.

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
            "source_symbol": out["source_symbol"].fillna("").astype(str).str.strip().str.upper(),
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
    return out


def ticker_rows(stamped: pd.DataFrame, tickers: Collection[str] | None = None) -> pd.DataFrame:
    """Ticker-grain rows: per (ticker, settlement date) over the canonical and secondary-class lines, the sum of
    quantity x conversion ratio and of quantity x PRICE; the dollars are NULL when a summed line has no price."""
    summed = summed_lines(stamped, tickers)
    if summed.empty:
        return pd.DataFrame(columns=list(TICKER_COLUMNS))
    quantity = pd.to_numeric(summed["fails_quantity"], errors="coerce")
    value = pd.to_numeric(summed["fails_value"], errors="coerce")
    work = pd.DataFrame(
        {
            "ticker": summed["ticker"].astype(str),
            "date": pd.to_datetime(summed["date"]).dt.normalize(),
            "fails_quantity": quantity * summed["conversion_ratio"].astype("float64"),
            "fails_value": value,
            "unpriced": value.isna() & quantity.notna(),
            "period": summed["period"].astype(str),
        }
    )
    grouped = work.groupby(["ticker", "date"], as_index=False, sort=True).agg(
        fails_quantity=("fails_quantity", "sum"), fails_value=("fails_value", "sum"), unpriced=("unpriced", "any"), period=("period", "min")
    )
    grouped["fails_value"] = grouped["fails_value"].where(~grouped["unpriced"])
    return grouped[list(TICKER_COLUMNS)]


def _unique_keys(lines: pd.DataFrame, label: str) -> pd.DataFrame:
    """One line per (cusip, settlement date): exact repeats collapse; a repeat with other values keeps the first (WARNING)."""
    kept = lines.drop_duplicates(["cusip", "date", "source_symbol", "fails_quantity"], ignore_index=True)
    duplicate = kept.duplicated(["cusip", "date"], keep="first")
    if duplicate.any():
        logger.warning("%s: %d line(s) repeat a (cusip, settlement date) key with other values; the first is kept", label, int(duplicate.sum()))
        kept = kept[~duplicate].reset_index(drop=True)
    return kept


# --------------------------------------------------------------------------- #
# IO: stamp the stored lines, rebuild the ticker grain                          #
# --------------------------------------------------------------------------- #


def load_fails_master(context: Context) -> pd.DataFrame | None:
    """The stored `security_master` prepared for stamping; None before the first `identity-tables` build."""
    master = context.store.load(Tables.security_master, columns=[*_MASTER_COLUMNS, "source"], optional=True)
    return None if master is None else prepare_master(master)


def _log_unplaced(stamped: pd.DataFrame) -> None:
    unplaced = stamped["security_id"].notna() & stamped["lineage_role"].isna()
    if unplaced.any():
        cusips = sorted(set(stamped.loc[unplaced, "cusip"]))
        logger.warning(
            "FTD: %d line(s) of %d master CUSIP(s) fall in no master interval; stored, never summed: %s",
            int(unplaced.sum()),
            len(cusips),
            ", ".join(cusips[:20]),
        )


def _purge_out_of_scope(context: Context, stamped: pd.DataFrame, *, by_key: bool = False) -> int:
    """Delete the stored lines of CUSIPs the master lacks (no universe security, or a co-registrant's): every line
    of such a CUSIP, or under `by_key` only the (CUSIP, settlement date) keys of `stamped`."""
    gone = stamped[stamped["security_id"].isna()]
    if gone.empty:
        return 0
    cusips = sorted(set(gone["cusip"].astype(str)))
    if by_key:
        for cusip, group in gone.groupby(gone["cusip"].astype(str), sort=True):
            for chunk in batched(sorted(pd.to_datetime(group["date"])), KEY_CHUNK, strict=False):
                context.store.delete(Tables.sec_fails_to_deliver_security, where={"cusip": str(cusip), "date": chunk})
    else:
        for chunk in batched(cusips, KEY_CHUNK, strict=False):
            context.store.delete(Tables.sec_fails_to_deliver_security, where={"cusip": chunk})
    logger.warning("FTD: %d stored line(s) over %d CUSIP(s) have no security_master row; deleted", len(gone), len(cusips))
    return len(gone)


def _held_keys(context: Context, cusips: Sequence[str], others: frozenset[str]) -> pd.DataFrame:
    """The stored (cusip, date) keys of `cusips` stamped with a company of `others`."""
    kept = load_chunked(context, Tables.sec_fails_to_deliver_security, ["cusip", "date", "ticker"], "cusip", cusips, KEY_CHUNK) if others else []
    rows = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["cusip", "date", "ticker"])
    rows = rows[rows["ticker"].isin(others)]
    return pd.DataFrame({"cusip": rows["cusip"].astype(str), "date": pd.to_datetime(rows["date"])})


def _rebuild_from_cache(context: Context, master: pd.DataFrame, universe: frozenset[str], others: frozenset[str] = frozenset()) -> tuple[int, int]:
    """Every cached ZIP's lines of the master's CUSIPs, stamped; both tables replaced. `(raw rows, ticker rows)`.

    With `others` (the universe companies outside a `-t` scope) only the CUSIPs of `universe`'s companies are read
    and their lines and ticker rows upserted; a line stamped, stored or fresh, with a company of `others` is left."""
    upsert = bool(others)
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    cached = sorted(_cached_periods(cache))
    missing = sorted(set(map(str, context.store.distinct(Tables.sec_fails_to_deliver_security, "period"))) - set(cached))
    if missing:
        raise FileNotFoundError(f"FTD full rebuild cannot reproduce stored period(s) {missing}: not in the cache")
    cusips = frozenset(master.loc[master["canonical_company"].isin(universe), "cusip"] if upsert else master["cusip"])
    frames: list[pd.DataFrame] = []
    other = 0
    for period in tqdm(cached, desc="SEC fails-to-deliver (rebuild)"):
        lines = _read_period(context, cache, period, download=False)
        if lines is None:
            raise ValueError(f"FTD full rebuild cannot read cached period {period}")
        scoped = lines["cusip"].isin(cusips)
        other += int((~scoped).sum())
        frames.append(stamp_lines(lines[scoped], master))
    stamped = _unique_keys(pd.concat(frames, ignore_index=True), "FTD rebuild") if frames else pd.DataFrame(columns=list(SECURITY_COLUMNS))
    grain = ticker_rows(stamped, universe)
    if grain.empty and not upsert:
        raise ValueError("FTD full rebuild staged no ticker rows")
    _log_unplaced(stamped)
    if upsert:
        held = _held_keys(context, sorted(cusips), others).assign(_held=True)
        keys = pd.DataFrame({"cusip": stamped["cusip"].astype(str), "date": pd.to_datetime(stamped["date"])})
        mine = keys.merge(held, on=["cusip", "date"], how="left")["_held"].isna().to_numpy() & ~stamped["ticker"].isin(others).to_numpy()
        raw = context.store.save(Tables.sec_fails_to_deliver_security, stamped.loc[mine, list(SECURITY_COLUMNS)])
        apply_grain(
            context, sorted(universe), grain, dry_run=False, table=Tables.sec_fails_to_deliver, columns=TICKER_COLUMNS, text_columns=("period",)
        )
        logger.info("FTD scoped rebuild: %d line(s) and %d ticker row(s) upserted for %s", raw, len(grain), ", ".join(sorted(universe)))
        return raw, len(grain)
    raw = context.store.replace(Tables.sec_fails_to_deliver_security, stamped[list(SECURITY_COLUMNS)])
    saved = context.store.replace(Tables.sec_fails_to_deliver, grain)
    roles = stamped["lineage_role"].fillna("none").value_counts().sort_index().to_dict()
    logger.info(
        "FTD rebuild: %d period(s); %d line(s) stored, by role %s; %d line(s) of other CUSIPs not stored; %d ticker row(s)",
        len(cached),
        raw,
        roles,
        other,
        saved,
    )
    return raw, saved


def _stamp_new(context: Context, master: pd.DataFrame, universe: frozenset[str], others: frozenset[str] = frozenset()) -> int:
    """Stamp the NULL-stamped lines (`ftd-download`'s new rows) and rebuild the ticker rows of their periods.

    The stamps are saved after the rebuild, so a failed run leaves the lines pending. With `others` (a `-t` run)
    only the lines of `universe`'s companies are stamped; every other line stays pending for an unscoped run."""
    table = Tables.sec_fails_to_deliver_security
    new = context.store.load(table, columns=list(SECURITY_COLUMNS[:9]), where={"security_id": None}, optional=True)
    if new is None:
        return 0
    stamped = stamp_lines(new, master)
    unowned = stamped[stamped["security_id"].isna()]
    stamped = stamped[stamped["security_id"].notna()]
    if others:
        scope_cusips = set(master.loc[master["canonical_company"].isin(universe), "cusip"])
        stamped = stamped[stamped["ticker"].isin(universe) | (stamped["ticker"].isna() & stamped["cusip"].isin(scope_cusips))]
    if stamped.empty:
        if not others:
            _purge_out_of_scope(context, unowned)
        return 0
    _log_unplaced(stamped)
    periods = sorted(set(stamped["period"].astype(str)))
    lines = context.store.load(table, columns=list(SECURITY_COLUMNS), where={"period": periods}, optional=True)
    grain = ticker_rows(stamp_lines(lines, master), universe) if lines is not None else pd.DataFrame(columns=list(TICKER_COLUMNS))
    for chunk in batched(sorted(universe), TICKER_CHUNK, strict=False):
        context.store.delete(Tables.sec_fails_to_deliver, where={"ticker": chunk, "period": periods})
    saved = context.store.save(Tables.sec_fails_to_deliver, grain) if not grain.empty else 0
    context.store.save(table, stamped[list(SECURITY_COLUMNS)])
    if not others:
        _purge_out_of_scope(context, unowned)
    logger.info("FTD: %d new line(s) stamped over %d period(s); %d ticker row(s) rebuilt", len(stamped), len(periods), saved)
    return saved


def _company_lines(context: Context, master: pd.DataFrame, companies: Sequence[str]) -> pd.DataFrame:
    """Stored lines stamped with one of `companies` or on one of their master CUSIPs."""
    table = Tables.sec_fails_to_deliver_security
    columns = list(SECURITY_COLUMNS)
    cusips = sorted(set(master.loc[master["canonical_company"].isin(set(companies)), "cusip"]))
    kept = load_chunked(context, table, columns, "ticker", companies, TICKER_CHUNK) + load_chunked(
        context, table, columns, "cusip", cusips, KEY_CHUNK
    )
    if not kept:
        return pd.DataFrame(columns=columns)
    lines = pd.concat(kept, ignore_index=True)
    lines["date"] = pd.to_datetime(lines["date"])
    return lines.drop_duplicates(["cusip", "date"], ignore_index=True)


def restamp_fails(
    context: Context,
    companies: Collection[str] | None,
    tickers: Collection[str],
    *,
    master: pd.DataFrame | None = None,
    dry_run: bool = False,
    as_of: pd.Timestamp | None = None,
) -> list[dict]:
    """Re-stamp the stored lines of `companies` from the master and rebuild their ticker rows, writing only what changed.

    `companies` None: `security_tape.restamp_targets` (master rows changed recently on `as_of`, default today, or all
    vanished). Lines of CUSIPs the master no longer holds are deleted.
    Returns one record per ticker whose ticker rows vanished. A scoped run (`tickers` short of the universe) never
    deletes or re-stamps a line stamped, before or after, with a universe company outside it; stamps are saved after
    the ticker rows are rebuilt.
    """
    master = load_fails_master(context) if master is None else master
    if master is None or master.empty:
        return []
    if companies is None:
        companies = restamp_targets(context, Tables.sec_fails_to_deliver_security, master_stamps(master), as_of)
    if not companies:
        return []
    universe = frozenset(normalise_ticker(ticker) for ticker in tickers)
    others = outside_scope(context, universe)
    names = sorted(set(companies) - others)
    if not names:
        return []
    stored = _company_lines(context, master, names)
    if stored.empty:
        return []
    fresh = stamp_lines(stored, master)
    mine = ~(stored["ticker"].isin(others) | fresh["ticker"].isin(others))
    changed = stamp_changed(stored, fresh) & fresh["security_id"].notna() & mine
    rebuilt = [name for name in names if name in universe]
    grain = ticker_rows(fresh, rebuilt)
    records = (
        apply_grain(context, rebuilt, grain, dry_run=dry_run, table=Tables.sec_fails_to_deliver, columns=TICKER_COLUMNS, text_columns=("period",))
        if rebuilt
        else []
    )
    if not dry_run:
        if changed.any():
            context.store.save(Tables.sec_fails_to_deliver_security, fresh.loc[changed, list(SECURITY_COLUMNS)])
        _purge_out_of_scope(context, fresh[mine], by_key=bool(others))
    logger.info("FTD: %d line(s) of %d company(ies) re-stamped%s", int(changed.sum()), len(names), " (dry run)" if dry_run else "")
    return records


def fetch_fails_to_deliver(context: Context, tickers: Sequence[str], full: bool = False, as_of: pd.Timestamp | None = None) -> int:
    """Stamp the stored raw FTD lines from `security_master` and rebuild `sec_fails_to_deliver`; returns ticker rows written.

    Incremental: the NULL-stamped lines (new from `ftd-download`) and the lines of companies whose master rows changed
    recently; no ZIP is read. `full` rebuilds both tables from every cached ZIP; a scoped (`-t`) `full` upserts only
    its tickers' lines and rows.
    """
    universe = frozenset(normalise_ticker(ticker) for ticker in tickers)
    master = load_fails_master(context)
    if master is None or master.empty:
        logger.warning("FTD: no security_master yet (run identity-tables first); nothing stamped")
        return 0
    others = outside_scope(context, universe)
    if full:
        _, saved = _rebuild_from_cache(context, master, universe, others)
    else:
        saved = _stamp_new(context, master, universe, others)
        warn_lost_rows(logger, "FTD", restamp_fails(context, None, universe, master=master, as_of=as_of))
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


def _recent_keys(rows: pd.DataFrame, key: str, as_of: pd.Timestamp) -> list[str]:
    """The values of `key` whose latest `scope_changed_at` is inside `resume.recently_changed`'s window on `as_of`."""
    stamps = pd.to_datetime(rows["scope_changed_at"], errors="coerce").groupby(rows[key].astype(str)).max()
    return recently_changed({str(k): v for k, v in stamps.items()}, as_of)


def _changed_scope(context: Context, lineage: pd.DataFrame, as_of: pd.Timestamp) -> tuple[frozenset[str], frozenset[str]]:
    """`(symbols, CUSIP-6 prefixes)` of the lineage entities and master securities whose rows changed recently."""
    if "scope_changed_at" not in context.store.columns(Tables.entity_lineage):
        return frozenset(), frozenset()
    stamped = context.store.load(Tables.entity_lineage, columns=["entity_id", "scope_changed_at"])
    assert stamped is not None
    entities = set(_recent_keys(stamped, "entity_id", as_of))
    symbols = lineage_scope_symbols(lineage[lineage["entity_id"].astype(str).isin(entities)]) if entities else frozenset()
    if "scope_changed_at" not in context.store.columns(Tables.security_master):
        return symbols, frozenset()
    master = context.store.load(Tables.security_master, columns=["cusip", "scope_changed_at"], optional=True)
    if master is None or master.empty:
        return symbols, frozenset()
    return symbols, frozenset(cusip[:6] for cusip in _recent_keys(master, "cusip", as_of) if cusip)


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


def download_fails_to_deliver(context: Context, years_history: int = 15, full: bool = False, as_of: pd.Timestamp | None = None) -> int:
    """Cache the FTD ZIPs and store their in-scope source lines raw in `sec_fails_to_deliver_security`.

    Scope: lineage symbols plus the CUSIP-6 of the master's securities. The periods read are those no stored line
    carries; every cached period is re-read for the symbols and prefixes whose lineage or master rows changed
    recently and for a CUSIP-6 newly voted for by a lineage symbol interval. Stamp columns stay NULL. `full`
    replaces the table from every listed period. Returns the rows written.
    """
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    table = Tables.sec_fails_to_deliver_security
    run_date = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    lineage, symbols, prefixes = _ingest_scope(context)
    periods = sorted(_cached_periods(cache) | set(_periods(years_history + 1, run_date)))
    if lineage is None:
        cached = [
            p
            for p in periods
            if ensure_zip(context, cache / FTD_ZIP_NAME_TEMPLATE.format(period=p), _period_urls(p), label=f"FTD {p}", timeout=180, log=logger)
        ]
        logger.warning("FTD download: no entity_lineage yet; %d zip(s) cached, lines are ingested after identity-tables has run", len(cached))
        return 0
    listed, stored = archive_periods(context, (table,), periods)
    pending = listed if full else [p for p in listed if p not in stored]
    frames = []
    for period in tqdm(pending, desc="FTD download"):
        frame = _read_period(context, cache, period, download=True)
        if frame is not None:
            frames.append(_in_scope(frame, symbols, prefixes))
    kept = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(SECURITY_COLUMNS[:9]))
    voted = cusip_votes(kept, lineage) if not kept.empty else pd.DataFrame(columns=["cusip"])
    added = frozenset(str(c)[:6] for c in voted["cusip"]) - prefixes
    changed_symbols, changed_prefixes = (frozenset(), frozenset()) if full else _changed_scope(context, lineage, run_date)
    rescan_prefixes = added | changed_prefixes
    if rescan_prefixes or changed_symbols:
        logger.info(
            "FTD download: re-reading every cached period for %d changed symbol(s) and %d CUSIP-6 prefix(es) (%s)",
            len(changed_symbols),
            len(rescan_prefixes),
            ", ".join(sorted(rescan_prefixes)),
        )
        for period in sorted(_cached_periods(cache)):
            frame = _read_period(context, cache, period, download=False)
            if frame is not None:
                kept = pd.concat([kept, _in_scope(frame, changed_symbols, rescan_prefixes)], ignore_index=True)
    kept = _unique_keys(kept, "FTD download")
    rows = kept.assign(security_id=None, ticker=None, lineage_role=None, security_class=None)[list(SECURITY_COLUMNS)]
    if full:
        written = context.store.replace(table, rows)
    else:
        written = context.store.save(table, rows) if not rows.empty else 0
    logger.info(
        "FTD download: %d period(s) read, %d in-scope line(s) stored (%d symbol(s), %d CUSIP-6 prefix(es) in scope)",
        len(pending),
        written,
        len(symbols),
        len(prefixes | added),
    )
    return written
