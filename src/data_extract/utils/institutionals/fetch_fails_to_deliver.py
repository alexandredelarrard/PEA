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
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants.constants import CANONICAL_ROLES, FTD_ZIP_NAME_TEMPLATE, SHARADAR_ACTION_SPLIT
from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir, cached_periods, ensure_zip, period_end, read_zip_text
from src.data_extract.utils.common.identity import Identity, load_source_identity
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
    load_chunked,
    nullable,
    outside_scope,
    restamp_targets,
    stamp_changed,
    summed_lines,
)
from src.data_store.schema import Resume, Table, Tables
from src.utils.price_adjustments import ShareBasis, load_bugfix, resolve_share_events
from src.utils.price_basis import prepare_quote_anchors
from src.utils.string import normalise_ticker, run_date, yahoo_symbol

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
    "market_symbol",
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
SEC_FTD_ZIP_PREFIX = "cnsfails"  # a cached archive is named `{prefix}{period}.zip`


def _periods(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Semi-monthly file tags ('YYYYMMa'/'YYYYMMb') from the data-set era to now."""
    today = (today or pd.Timestamp.today()).normalize()
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


def _parse_ftd_lines(raw: str) -> pd.DataFrame:
    """Every source line of one FTD file, raw: symbol as filed, description, PRICE ('.' -> NULL), trade date.

    Exact duplicate lines collapse to one; nothing is summed.
    """
    columns = list(SECURITY_COLUMNS[:8])
    lines = raw.strip().splitlines()
    header = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"
    if not lines or lines[0].strip().lstrip("\ufeff").upper() != header or any(len(row.split("|")) != 6 for row in lines[1:]):
        raise ValueError("FTD invalid header or source line width")
    df = pd.read_csv(io.StringIO("\n".join([header, *lines[1:]])), sep="|", dtype=str, keep_default_na=False, quoting=3)
    cols = {c.strip().upper(): c for c in df.columns}
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
            "source_symbol": _ftd_column(df, cols, "SYMBOL").map(normalise_ticker),
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
_HIT_COLUMNS = ["canonical_company", "lineage_role", "security_class", "conversion_ratio", "market_symbol"]


def prepare_master(master: pd.DataFrame) -> pd.DataFrame:
    """The FTD rows of `security_master`, keyed for stamping: upper-case CUSIP and symbol, open ends at a far date."""
    rows = master[master["source"].eq(SOURCE_FTD)] if "source" in master.columns else master
    out = rows.reindex(columns=_MASTER_COLUMNS).copy()
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


def ticker_rows(stamped: pd.DataFrame, tickers: Collection[str] | None = None, basis: ShareBasis | None = None) -> pd.DataFrame:
    """Ticker-grain rows: per (ticker, settlement date) over the canonical and secondary-class lines, the sum of
    quantity x conversion ratio and of quantity x PRICE; the dollars are NULL when a summed line has no price."""
    summed = summed_lines(stamped, tickers)
    if summed.empty:
        return _empty_fails_grain()[list(TICKER_COLUMNS)]
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
    return grouped[list(TICKER_COLUMNS)]


def _settlement_quantity(summed: pd.DataFrame, basis: ShareBasis | None) -> pd.Series:
    """Raw outstanding quantities are settlement units; identity/class ratios use the inferred date."""
    quantity = pd.to_numeric(summed["fails_quantity"], errors="coerce")
    ratio = pd.to_numeric(summed["conversion_ratio"], errors="coerce")
    if not (np.isfinite(quantity) & quantity.ge(0) & np.isfinite(ratio) & ratio.gt(0)).all():
        raise ValueError("FTD quantities and class ratios require finite valid source units")
    result = quantity * ratio
    if basis is None:
        return result
    symbols = summed["market_symbol"].map(yahoo_symbol)
    canonical = summed["ticker"].map(yahoo_symbol)
    trade = pd.to_datetime(summed["trade_date"])
    settlement = pd.to_datetime(summed["date"])
    class_quantity = basis.rebase(quantity, symbols, settlement, trade)
    converted = basis.rebase(class_quantity * ratio, canonical, trade, settlement)
    same_class = symbols.eq(canonical) | summed["lineage_role"].isin(CANONICAL_ROLES)
    return converted.where(~same_class, result)


def _checked_ftd_grain(
    source: pd.DataFrame, master: pd.DataFrame, universe: frozenset[str], basis: ShareBasis | None, held: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Infer absence only on a fully parsed period's actual served dates and supported company scope."""
    stamped = stamp_lines(source, master)
    grain = ticker_rows(stamped, universe, basis)
    dates = source.attrs.get("served_dates", source[["date", "trade_date", "period"]].drop_duplicates())
    candidates = dates.merge(master, how="cross")
    candidates = candidates.loc[
        candidates["canonical_company"].isin(universe)
        & candidates["lineage_role"].isin(("canonical_current", "canonical_predecessor", "secondary_class"))
    ]
    candidates = _covering(candidates)
    invalid_ratio = ~np.isfinite(candidates["conversion_ratio"]) | candidates["conversion_ratio"].le(0)
    invalid = set(zip(candidates.loc[invalid_ratio, "canonical_company"], candidates.loc[invalid_ratio, "date"], strict=True))
    candidates = candidates.rename(columns={"canonical_company": "ticker"})[["ticker", "date", "period"]].drop_duplicates()
    unresolved = stamped.loc[stamped["lineage_role"].isna() | stamped["ticker"].isna()]
    blocked = unresolved[["cusip", "date"]].merge(master[["cusip", "canonical_company"]].drop_duplicates(), on="cusip")
    symbols = unresolved[["source_symbol", "date"]].merge(master[["source_symbol", "canonical_company"]].drop_duplicates(), on="source_symbol")
    blocked = pd.concat([blocked, symbols], ignore_index=True)
    blocked = set(zip(blocked["canonical_company"], pd.to_datetime(blocked["date"]), strict=True))
    blocked |= invalid
    if held is not None and not held.empty:
        retained = stamped.merge(held[["cusip", "date"]], on=["cusip", "date"])
        blocked.update(zip(retained["ticker"], pd.to_datetime(retained["date"]), strict=True))
    candidates = candidates.loc[
        [(ticker, pd.Timestamp(day)) not in blocked for ticker, day in zip(candidates["ticker"], candidates["date"], strict=True)]
    ]
    result = candidates.merge(grain, on=["ticker", "date", "period"], how="left", indicator=True)
    absent = result.pop("_merge").eq("left_only")
    result.loc[absent, ["fails_quantity", "fails_value"]] = 0.0
    result[["fails_quantity", "fails_value"]] = result[["fails_quantity", "fails_value"]].astype("float64")
    result["source_checked_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return result


def _empty_fails_grain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype=object),
            "date": pd.Series(dtype="datetime64[ns]"),
            "fails_quantity": pd.Series(dtype="float64"),
            "fails_value": pd.Series(dtype="float64"),
            "period": pd.Series(dtype=object),
            "source_checked_at": pd.Series(dtype="datetime64[ns]"),
        }
    )


def _scope_fails_source(source: pd.DataFrame, master: pd.DataFrame, universe: frozenset[str]) -> pd.DataFrame:
    """Retain all relevant actual lines plus compact, actual served-calendar evidence."""
    selected = master.loc[master["canonical_company"].isin(universe)]
    relevant = source["cusip"].isin(selected["cusip"]) | source["source_symbol"].map(squash).isin(selected["source_symbol"].map(squash))
    scoped = source.loc[relevant].copy()
    scoped.attrs["served_dates"] = source.attrs.get("served_dates", source[["date", "trade_date", "period"]].drop_duplicates())
    return scoped


def _unique_keys(lines: pd.DataFrame, label: str) -> pd.DataFrame:
    """Collapse exact source repeats; refuse conflicting payloads or ZIP periods at the raw PK."""
    source_columns = [column for column in SECURITY_COLUMNS[:8] + ("period",) if column in lines]
    kept = lines.drop_duplicates(source_columns, ignore_index=True)
    if kept.duplicated(["cusip", "date"]).any():
        raise ValueError(f"{label}: conflicting FTD source payload or period for (cusip, settlement date)")
    return kept


# --------------------------------------------------------------------------- #
# IO: stamp the stored lines, rebuild the ticker grain                          #
# --------------------------------------------------------------------------- #


def load_fails_master(context: Context) -> pd.DataFrame | None:
    """The stored `security_master` prepared for stamping; None before the first `identity-tables` build."""
    master = context.store.load(Tables.security_master, columns=[*_MASTER_COLUMNS, "source"], optional=True)
    return None if master is None else prepare_master(master)


def load_fails_basis(context: Context, master: pd.DataFrame) -> ShareBasis:
    """Resolve the shared genuine-event contract for the scoped source classes and canonical companies."""
    names = sorted(set(master["market_symbol"].dropna().astype(str)) | set(master["canonical_company"].dropna().astype(str)))
    bugfix = load_bugfix(getattr(context, "config_dir", "configs"))
    policies = {ticker: policy for ticker, policy in bugfix.get("share_events", {}).items() if ticker in names}
    anchors = [(ticker, anchor) for ticker, policy in policies.items() for anchor in policy["anchors"] if "quoted_price" in anchor]
    actions = context.store.load(
        Tables.sharadar_actions,
        columns=["ticker", "date", "action", "value"],
        where={"ticker": names, "action": [SHARADAR_ACTION_SPLIT]},
        optional=True,
    )
    splits = context.store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": names}, optional=True)
    vendor = (
        context.store.load(
            Tables.sharadar_fundamentals,
            columns=["ticker", "date", "sharesbas", "price"],
            where={"ticker": sorted(policies), "dimension": "ARQ"},
            optional=True,
        )
        if policies
        else None
    )
    quotes = (
        context.store.load(
            Tables.prices,
            columns=["ticker", "date", "close_split", "volume"],
            where={"ticker": sorted({ticker for ticker, _ in anchors})},
            optional=True,
        )
        if anchors
        else None
    )
    if splits is None and context.store.exists(Tables.prices_splits):
        splits = pd.DataFrame(columns=["ticker", "date", "ratio"])
    if actions is None and context.store.exists(Tables.sharadar_actions):
        actions = pd.DataFrame(columns=["ticker", "date", "action", "value"])
    checked_quotes = prepare_quote_anchors(quotes, vendor, anchors, bugfix) if anchors else None
    return resolve_share_events(actions, splits, bugfix, shares=vendor, quotes=checked_quotes, tickers=names)


def _prepare_fails_period(
    context: Context, source: pd.DataFrame, master: pd.DataFrame, universe: frozenset[str], identity: dict[Table, pd.DataFrame], basis: ShareBasis
) -> tuple[dict[Table, pd.DataFrame], dict[Table, dict[str, object] | None]]:
    """Prepare one actual source period's exact raw/derived pair and guarded scopes."""
    source = _scope_fails_source(_unique_keys(source, "FTD source period"), master, universe)
    periods = sorted(set(source["period"].astype(str))) or [str(source.attrs["period"])]
    roster = identity[Tables.sp500_tickers]
    others = sorted(set(roster["ticker"].astype(str)) - universe) if "ticker" in roster else []
    held = (
        context.store.load(
            Tables.sec_fails_to_deliver_security, columns=["cusip", "date", "ticker"], where={"period": periods, "ticker": others}, optional=True
        )
        if others
        else None
    )
    if held is not None:
        held["date"] = pd.to_datetime(held["date"])
    stamped = stamp_lines(source, master)
    scoped = stamped.loc[stamped["ticker"].isin(universe) | stamped["ticker"].isna()]
    if held is not None:
        scoped = scoped.loc[~pd.MultiIndex.from_frame(scoped[["cusip", "date"]]).isin(pd.MultiIndex.from_frame(held[["cusip", "date"]]))]
    grain = _checked_ftd_grain(source, master, universe, basis, held)
    return (
        {Tables.sec_fails_to_deliver_security: scoped[list(SECURITY_COLUMNS)], Tables.sec_fails_to_deliver: grain},
        {
            Tables.sec_fails_to_deliver_security: {"ticker": sorted(universe), "period": periods},
            Tables.sec_fails_to_deliver: {"ticker": sorted(universe), "period": periods},
        },
    )


def _write_fails_period(
    context: Context, frames: dict[Table, pd.DataFrame], scopes: dict[Table, dict[str, object] | None], identity: dict[Table, pd.DataFrame]
) -> tuple[int, int]:
    counts = context.store.replace_institutional_source(frames, scopes, identity)
    return counts[Tables.sec_fails_to_deliver_security], counts[Tables.sec_fails_to_deliver]


def _save_fails_period(
    context: Context, source: pd.DataFrame, master: pd.DataFrame, universe: frozenset[str], identity: dict[Table, pd.DataFrame], basis: ShareBasis
) -> tuple[int, int]:
    """Commit one actual source period's existing pair together."""
    frames, scopes = _prepare_fails_period(context, source, master, universe, identity, basis)
    return _write_fails_period(context, frames, scopes, identity)


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


def _rebuild_from_cache(
    context: Context, master: pd.DataFrame, universe: frozenset[str], others: frozenset[str], identity: dict[Table, pd.DataFrame], basis: ShareBasis
) -> tuple[int, int]:
    """Validate every required cached original before bounded atomic period replacements."""
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    cached = sorted(_cached_periods(cache))
    stored = context.store.distinct(Tables.sec_fails_to_deliver_security, "period")
    missing = sorted(set(map(str, stored)) - set(cached))
    if missing:
        raise FileNotFoundError(f"FTD full rebuild cannot reproduce stored period(s) {missing}: not in the cache")
    frames: list[pd.DataFrame] = []
    for period in tqdm(cached, desc="SEC fails-to-deliver (rebuild)"):
        source = _read_period(context, cache, period, download=False)
        if source is None:
            raise ValueError(f"FTD full rebuild cannot read cached period {period}")
        frames.append(_scope_fails_source(source, master, universe))
    if frames:
        payloads = []
        for frame in frames:
            payload = frame.copy()
            payload.attrs = {}
            payloads.append(payload)
        combined = pd.concat(payloads, ignore_index=True)
        _unique_keys(combined, "FTD rebuild")
    raw, saved = 0, 0
    for source in frames:
        n_raw, n_saved = _save_fails_period(context, source, master, universe, identity, basis)
        raw += n_raw
        saved += n_saved
    logger.info("FTD rebuild: %d validated periods; %d raw and %d ticker rows; %d outside companies retained", len(frames), raw, saved, len(others))
    return raw, saved


def _stamp_new(
    context: Context, master: pd.DataFrame, universe: frozenset[str], others: frozenset[str], identity: dict[Table, pd.DataFrame], basis: ShareBasis
) -> int:
    """Reconcile pending units and recover required originals missing from the cache."""
    stored, pending = context.store.fails_source_periods()
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    cached = _cached_periods(cache)
    pending |= stored.symmetric_difference(cached)
    saved = 0
    for period in sorted(map(str, pending)):
        source = _read_period(context, cache, period, download=True)
        if source is None:
            logger.warning("FTD: pending %s has no readable original; good rows retained, original remains required", period)
            continue
        scoped = _scope_fails_source(source, master, universe)
        _, written = _save_fails_period(context, scoped, master, universe, identity, basis)
        saved += written
    logger.info("FTD: %d pending periods reconciled; %d ticker rows; %d outside companies retained", len(pending), saved, len(others))
    return saved


def _lost_fails_records(gone: list[pd.DataFrame]) -> list[dict]:
    """Report only actual keys absent from successfully reconciled complete periods."""
    records = []
    lost = pd.concat(gone, ignore_index=True) if gone else pd.DataFrame(columns=["ticker", "date"])
    for ticker, group in lost.groupby("ticker", sort=True):
        dates = pd.to_datetime(group["date"])
        records.append(
            {
                "table": Tables.sec_fails_to_deliver.name,
                "ticker": str(ticker),
                "cik": "",
                "first_filed": dates.min().date().isoformat(),
                "last_filed": dates.max().date().isoformat(),
                "keys": len(group),
                "rows": len(group),
            }
        )
    return records


def _recover_fails_companies(
    context: Context, master: pd.DataFrame, universe: frozenset[str], identity: dict[Table, pd.DataFrame], basis: ShareBasis, dry_run: bool = False
) -> tuple[int, list[dict]]:
    """A changed scope needs actual source units again, including periods where this issuer was absent."""
    stored = context.store.load(Tables.sec_fails_to_deliver, columns=["ticker", "date", "period"], where={"ticker": sorted(universe)}, optional=True)
    observed = stored if stored is not None else pd.DataFrame(columns=["ticker", "date", "period"])
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    saved, gone = 0, []
    for period in sorted(_cached_periods(cache) | set(observed["period"].astype(str))):
        source = _read_period(context, cache, period, download=not dry_run)
        if source is None:
            logger.warning("FTD: %s original unavailable; scope proof remains unknown", period)
            continue
        frames, scopes = _prepare_fails_period(context, source, master, universe, identity, basis)
        if not dry_run:
            _, written = _write_fails_period(context, frames, scopes, identity)
            saved += written
        old = observed.loc[observed["period"].astype(str).eq(period), ["ticker", "date"]].copy()
        old["date"] = pd.to_datetime(old["date"])
        keys = frames[Tables.sec_fails_to_deliver][["ticker", "date"]]
        gone.append(old.loc[~pd.MultiIndex.from_frame(old).isin(pd.MultiIndex.from_frame(keys))])
    return saved, _lost_fails_records(gone)


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
    identity: dict[Table, pd.DataFrame] | None = None,
    dry_run: bool = False,
    as_of: pd.Timestamp | None = None,
    others: frozenset[str] | None = None,
) -> list[dict]:
    """Reconcile the complete originals needed by changed companies with atomic scoped source pairs.

    `companies` None: `security_tape.restamp_targets` (master rows changed recently on `as_of`, default today, or all
    vanished). Lines of CUSIPs the master no longer holds are deleted.
    Returns one record per ticker whose ticker rows vanished. A scoped run (`tickers` short of the universe) never
    deletes or re-stamps a line stamped, before or after, with a universe company outside it. Dry runs read available
    cached originals and report their actual missing keys without downloads or database writes.
    """

    cohort = load_source_identity(context) if identity is None else identity
    master = prepare_master(cohort[Tables.security_master])
    if master.empty:
        return []
    if companies is None:
        companies = restamp_targets(context, Tables.sec_fails_to_deliver_security, master_stamps(master), as_of)
    if not companies:
        return []
    universe = frozenset(tickers)
    others = outside_scope(context, universe) if others is None else others
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
    records = []
    if rebuilt:
        _, records = _recover_fails_companies(context, master, frozenset(rebuilt), cohort, load_fails_basis(context, master), dry_run)
    logger.info("FTD: %d line(s) of %d company(ies) re-stamped%s", int(changed.sum()), len(names), " (dry run)" if dry_run else "")
    return records


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
    if path is None:
        return None
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
        logger.warning("FTD %s: invalid original (%s); deleting cache for normal retry; good stored rows retained", period, exc)
        path.unlink(missing_ok=True)
        return None
    parsed.attrs["period"] = period
    return parsed


def download_fails_to_deliver(context: Context, years_history: int = 31, full: bool = False, as_of: pd.Timestamp | None = None) -> int:
    """Cache the FTD ZIPs and store their in-scope source lines raw in `sec_fails_to_deliver_security`.

    Scope: lineage symbols plus the CUSIP-6 of the master's securities. The periods read are those no stored line
    carries; every cached period is re-read for the symbols and prefixes whose lineage or master rows changed
    recently and for a CUSIP-6 newly voted for by a lineage symbol interval. Stamp columns stay NULL. `full`
    replaces the table from every listed period. Returns the rows written.
    """
    cache = cache_dir(context, context.config.local.paths.fails_deliver)
    table = Tables.sec_fails_to_deliver_security
    start_date = run_date(as_of)
    lineage, symbols, prefixes = _ingest_scope(context)
    periods = sorted(_cached_periods(cache) | set(_periods(years_history + 1, start_date)))
    if lineage is None:
        cached = [
            p
            for p in periods
            if ensure_zip(context, cache / FTD_ZIP_NAME_TEMPLATE.format(period=p), _period_urls(p), label=f"FTD {p}", timeout=180, log=logger)
        ]
        logger.warning("FTD download: no entity_lineage yet; %d zip(s) cached, lines are ingested after identity-tables has run", len(cached))
        return 0
    listed, stored = archive_periods(context, (table,), periods)
    cached = _cached_periods(cache)
    pending = listed if full else [p for p in listed if p not in stored or p not in cached]
    frames = []
    reconciled: set[str] = set()
    for period in tqdm(pending, desc="FTD download"):
        frame = _read_period(context, cache, period, download=True)
        if frame is not None:
            frames.append(_in_scope(frame, symbols, prefixes))
            reconciled.add(period)
    kept = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(SECURITY_COLUMNS[:9]))
    voted = cusip_votes(kept, lineage) if not kept.empty else pd.DataFrame(columns=["cusip"])
    added = frozenset(str(c)[:6] for c in voted["cusip"]) - prefixes
    changed_symbols, changed_prefixes = (frozenset(), frozenset()) if full else _changed_scope(context, lineage, start_date)
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
                kept = pd.concat([kept, _in_scope(frame, symbols, prefixes | rescan_prefixes)], ignore_index=True)
                reconciled.add(period)
    kept = _unique_keys(kept, "FTD download")
    rows = kept.assign(security_id=None, ticker=None, lineage_role=None, security_class=None)[list(SECURITY_COLUMNS)]
    written = _save_pending_fails(context, rows, sorted(reconciled))
    logger.info(
        "FTD download: %d period(s) read, %d in-scope line(s) stored (%d symbol(s), %d CUSIP-6 prefix(es) in scope)",
        len(pending),
        written,
        len(symbols),
        len(prefixes | added),
    )
    return written


def _save_pending_fails(context: Context, rows: pd.DataFrame, periods: Sequence[str]) -> int:
    """New raw units and retained, explicitly unverified derived rows commit as one source pair."""
    previous = (
        context.store.load(
            Tables.sec_fails_to_deliver, columns=context.store.columns(Tables.sec_fails_to_deliver), where={"period": list(periods)}, optional=True
        )
        if periods
        else None
    )
    observed = previous if previous is not None else _empty_fails_grain()
    if "source_checked_at" not in observed:
        observed = observed.assign(source_checked_at=pd.Series(pd.NaT, index=observed.index, dtype="datetime64[ns]"))
    written = 0
    for period in periods:
        payload = rows.loc[rows["period"].eq(period)]
        expected = observed.loc[observed["period"].eq(period)]
        retained = expected.assign(source_checked_at=pd.Series(pd.NaT, index=expected.index, dtype="datetime64[ns]"))
        quiet = expected["source_checked_at"].notna() & expected["fails_quantity"].eq(0)
        retained.loc[quiet, ["fails_quantity", "fails_value"]] = float("nan")
        counts = context.store.replace_institutional_source(
            {Tables.sec_fails_to_deliver_security: payload, Tables.sec_fails_to_deliver: retained},
            {Tables.sec_fails_to_deliver_security: {"period": period}, Tables.sec_fails_to_deliver: {"period": period}},
            expected_source={Tables.sec_fails_to_deliver: expected},
        )
        written += counts[Tables.sec_fails_to_deliver_security]
    return written


def fetch_fails_to_deliver(
    context: Context,
    tickers: Sequence[str],
    full: bool = False,
    as_of: pd.Timestamp | None = None,
    *,
    all_tickers: Collection[str] | None = None,
    identity: Identity | None = None,
) -> int:
    """Stamp the stored raw FTD lines from `security_master` and rebuild `sec_fails_to_deliver`; returns ticker rows written.

    Incremental runs reread or recover the complete originals for pending units and changed issuer scopes. `full`
    rebuilds both tables from every cached ZIP; a scoped (`-t`) run replaces only its guarded issuer source pairs.
    """
    universe = frozenset(tickers)
    cohort = identity.source_cohort if identity is not None else load_source_identity(context)
    master = prepare_master(cohort[Tables.security_master]) if not cohort[Tables.security_master].empty else None
    if master is None or master.empty:
        logger.warning("FTD: no security_master yet (run identity-tables first); nothing stamped")
        return 0
    others = outside_scope(context, universe, all_tickers)
    basis = load_fails_basis(context, master)
    if full:
        _, saved = _rebuild_from_cache(context, master, universe, others, cohort, basis)
    else:
        saved = _stamp_new(context, master, universe, others, cohort, basis)
        changed = restamp_targets(context, Tables.sec_fails_to_deliver_security, master_stamps(master), as_of)
        if set(changed) & universe:
            recovered, _ = _recover_fails_companies(context, master, frozenset(set(changed) & universe), cohort, basis)
            saved += recovered
    return saved
