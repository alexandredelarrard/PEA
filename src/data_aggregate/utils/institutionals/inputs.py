"""Input loading for the institutional cube part."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS
from src.context import Context
from src.data_aggregate.utils.common.level_basis import apply_volume_scale
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import PriceFrames, load_price_frames
from src.data_aggregate.utils.institutionals.insider_quality import OTHER_OFFICER, clean_officer_title, officer_role, owner_name_key
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.string import pad_cik, yahoo_symbol

SHARES_OUT_COLUMNS = ("ticker", "as_of", "sharesOutstanding", "sharesOutstandingPit")
#: `symbol_tenure` sources behind the proven-tenure mask; cover-page `dei` rows stay out so features do not move.
TENURE_SOURCES = ("form345", "manual")
#: Columns a concurrently traded common class (`security_master` role `secondary_class`) needs for its volume.
CLASS_LINE_COLUMNS = ("canonical_company", "market_symbol", "conversion_ratio", "valid_from", "valid_to")
CLASS_BAR_COLUMNS = ("ticker", "date", "volume")


def load_full_price_frames(
    store: DataStore,
    context: Context,
    fields: Sequence[str],
) -> PriceFrames:
    """Load full-calendar price inputs from the already-persisted peer dependency."""
    return load_price_frames(
        store,
        peers=load_peers_or_raise(context),
        fields=fields,
        since=None,
    )


def load_source(
    store: DataStore,
    log: logging.Logger,
    table: Table,
    universe: Sequence[str] | None = None,
    where: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Load one source projected to its registry columns and scoped to the universe.

    The universe cut defines the cross-section rather than optimizing the read. Source
    tables can contain names outside the cube universe; excluding them at the read keeps
    every ``_xs`` leg on the same ticker denominator as the other cube parts. `where` adds row filters.
    """
    filters: dict[str, Any] = dict(where or {})
    if universe is not None and table.ticker_col:
        filters[table.ticker_col] = sorted(set(map(str, universe)))
        _report_off_universe(store, log, table, universe)

    frame = store.load(table, project=True, where=filters or None, optional=True)
    if frame is None:
        log.warning("%s is absent or empty -> its features are skipped.", table.name)
    else:
        log.info("Loaded %s: %s rows x %s cols", table.name, len(frame), len(frame.columns))
    return frame


def _report_off_universe(
    store: DataStore,
    log: logging.Logger,
    table: Table,
    universe: Sequence[str],
) -> None:
    """Log source tickers excluded by the universe-scoped read."""
    column = table.ticker_col
    if not column:
        return
    present = {str(ticker) for ticker in store.distinct(table, column)}
    off_universe = sorted(present - set(map(str, universe)))
    if not off_universe:
        return
    log.info(
        "%s: %s of its %s ticker(s) are outside the %s-name universe (%s) -- not read, so the `_xs` cross-section is the cube's",
        table.name,
        len(off_universe),
        len(present),
        len(universe),
        ", ".join(off_universe[:15]) + (", ..." if len(off_universe) > 15 else ""),
    )


def load_shares_out(store: DataStore, log: logging.Logger) -> pd.DataFrame | None:
    """Load both share-count bases required by institutional market-cap features."""
    frame = store.load(
        Tables.fundamentals_history,
        columns=list(SHARES_OUT_COLUMNS),
        optional=True,
    )
    if frame is None:
        log.warning("No fundamentals history -> the market-cap-scaled ownership features are skipped.")
        return None
    return frame


def load_symbol_lineage(
    store: DataStore,
    log: logging.Logger,
    universe: Sequence[str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load the issuer lineage (`form345`/`manual` tenure on the roster CIKs) used to validate already-canonical source rows."""
    tickers = sorted(set(map(str, universe)))
    ticker_ciks = store.load(
        Tables.sp500_tickers,
        columns=("ticker", "cik"),
        where={"ticker": tickers},
        optional=True,
    )
    if ticker_ciks is None or ticker_ciks.empty:
        return None, ticker_ciks

    ciks = sorted({pad_cik(value) for value in ticker_ciks["cik"] if pad_cik(value)})
    if not ciks:
        return None, ticker_ciks
    symbol_tenure = store.load(
        Tables.symbol_tenure,
        columns=("symbol", "issuer_cik", "valid_from", "valid_to"),
        where={"issuer_cik": ciks, "source": list(TENURE_SOURCES)},
        optional=True,
    )
    aliases = len(set(symbol_tenure["symbol"].astype(str))) if symbol_tenure is not None else 0
    log.info("Symbol lineage: %s current tickers backed by %s proven historical symbols", len(tickers), aliases)
    return symbol_tenure, ticker_ciks


def load_secondary_classes(
    store: DataStore,
    log: logging.Logger,
    universe: Sequence[str],
    bugfix: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The universe companies' `secondary_class` lines from `security_master` and those classes' `prices` volume bars.

    `prices` is read by exactly the classes' Yahoo symbols, which the universe filter of its other readers would drop.
    The price register's `volume_scale` entries (`bugfix`) repair a vendor volume unit defect on those bars.
    None when the master is absent or holds no class line, so the features keep the canonical tape.
    """
    lines = store.load(
        Tables.security_master,
        columns=list(CLASS_LINE_COLUMNS),
        where={"lineage_role": SECONDARY_CLASS, "canonical_company": sorted(set(map(str, universe)))},
        optional=True,
    )
    if lines is None or lines.empty:
        log.info("No secondary share-class lines in security_master -> ADV20 and RegSHO coverage use the canonical tape only.")
        return None
    symbols = sorted({yahoo_symbol(symbol) for symbol in lines["market_symbol"]})
    bars = store.load(Tables.prices, columns=list(CLASS_BAR_COLUMNS), where={"ticker": symbols}, optional=True)
    if bars is None:
        bars = pd.DataFrame(columns=list(CLASS_BAR_COLUMNS))
    if bugfix:
        bars = apply_volume_scale(bars, bugfix, log.info)
    log.info(
        "Secondary share classes: %s line(s), %s symbol(s) of %s companies, %s volume bar(s)",
        len(lines),
        len(symbols),
        lines["canonical_company"].nunique(),
        len(bars),
    )
    return lines, bars


def load_insider_transactions(store: DataStore, log: logging.Logger, universe: Sequence[str] | None = None) -> pd.DataFrame | None:
    """`insider_transactions` rows of the companies' own history: `lineage_role` canonical (predecessor or current).
    Acquired-constituent rows and rows without a stamp are not read."""
    table = Tables.insider_transactions
    present = store.columns(table)
    if not present:
        log.warning("%s is absent -> insider features are skipped.", table.name)
        return None
    columns = [column for column in dict.fromkeys((*table.read_columns, "issuer_cik")) if column in present]
    where: dict[str, Any] = {"lineage_role": list(CANONICAL_ROLES)}
    if universe is not None:
        where["ticker"] = sorted(set(map(str, universe)))
        _report_off_universe(store, log, table, universe)
    chunks = list(store.iter_load(table, columns=columns, where=where))
    if not chunks:
        return None
    frame = pd.concat(chunks, ignore_index=True)
    log.info("Loaded %s: %s rows x %s cols", table.name, len(frame), len(frame.columns))
    return frame


def load_insider_role_evidence(store: DataStore, insider: pd.DataFrame) -> pd.DataFrame:
    """Projected proxy title evidence for unique same-issuer/full-name owner identities only."""
    columns = ["ticker", "issuer_cik", "owner_cik", "role_as_of", "officer_title_clean", "role", "role_source"]
    if not {"issuer_cik", "owner_cik", "owner_name", "officer_title"}.issubset(insider):
        return pd.DataFrame(columns=columns)
    candidates = insider[insider["officer_title"].fillna("").astype(str).str.strip().eq("")].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    owners = insider[["ticker", "issuer_cik", "owner_cik", "owner_name", "filing_date"]].copy()
    owners["filing_date"] = pd.to_datetime(owners["filing_date"], errors="coerce").dt.normalize()
    owners = owners.sort_values("filing_date").drop_duplicates(["ticker", "issuer_cik", "owner_cik", "owner_name"])
    owners["name_key"] = owners["owner_name"].map(owner_name_key)
    for column in ("issuer_cik", "owner_cik"):
        owners[column] = owners[column].map(pad_cik).fillna("")
    identity = ["ticker", "issuer_cik", "name_key"]
    owners["identity_from"] = pd.to_datetime(owners["filing_date"], errors="coerce").dt.normalize()
    owners = owners[owners["name_key"].ne("") & owners["owner_cik"].ne("")].dropna(subset=["identity_from"])
    owners = owners.groupby([*identity, "owner_cik"], as_index=False)["identity_from"].min()
    ticks = sorted(candidates["ticker"].dropna().astype(str).unique())
    df_proxy = store.load(
        Tables.def14a_executive_comp, columns=["ticker", "cik", "as_of", "name", "title", "fiscal_year"], where={"ticker": ticks}, optional=True
    )
    if df_proxy is None or df_proxy.empty:
        return pd.DataFrame(columns=columns)
    df_proxy["issuer_cik"] = df_proxy["cik"].map(pad_cik).fillna("")
    df_proxy["name_key"] = df_proxy["name"].map(owner_name_key)
    df_proxy["role_as_of"] = pd.to_datetime(df_proxy["as_of"], errors="coerce").dt.normalize()
    df_proxy["officer_title_clean"] = df_proxy["title"].map(clean_officer_title)
    df_proxy["role"] = df_proxy["officer_title_clean"].map(officer_role)
    latest = df_proxy.groupby(["ticker", "as_of"])["fiscal_year"].transform("max")
    df_proxy = df_proxy[df_proxy["fiscal_year"].eq(latest)].dropna(subset=["role_as_of"])
    # Explicit non-management/former titles are history barriers; parent CEO fallback is not evidence.
    df_proxy = df_proxy[df_proxy["officer_title_clean"].ne("")]
    df_proxy = df_proxy.merge(owners, on=identity, how="inner")
    df_proxy["role_as_of"] = df_proxy[["role_as_of", "identity_from"]].max(axis=1)
    # Name uniqueness is evaluated when the mapping becomes public, never against future owners.
    identities = owners.set_index(identity)
    for _, group in df_proxy.groupby(identity, sort=False):
        first = identities.loc[tuple(group.iloc[0][identity])]["identity_from"]
        if isinstance(first, pd.Timestamp):
            continue
        ambiguous = group["role_as_of"].map(lambda day, first=first: first.le(day).sum() > 1)
        df_proxy.loc[group.index[ambiguous], "role"] = OTHER_OFFICER
    df_proxy["role_source"] = "explicit_proxy_title"
    conflicted = df_proxy.groupby(["ticker", "issuer_cik", "owner_cik", "role_as_of"])["role"].transform("nunique").gt(1)
    df_proxy.loc[conflicted, ["role", "officer_title_clean"]] = [OTHER_OFFICER, "conflicting roles"]
    return df_proxy[columns].drop_duplicates()


def load_filing_vintages(
    store: DataStore,
    log: logging.Logger,
    until: object,
    ciks: Sequence[str] | None = None,
    *,
    table: Table = Tables.sec13f_hr,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Read complete accession metadata and stream its whole books, never latest-only substitutes."""
    columns = (
        "accession_number",
        "cik",
        "period",
        "filing_date",
        "accepted_at",
        "amendment_type",
        "amendment_number",
        "report_type",
        "restricted",
        "parsed_row_count",
        "parse_status",
    )
    if not set(columns) <= set(store.columns(table)):
        log.warning("%s has no completed accession schema; replay original sources before building PIT features.", table.name)
        return None
    where: dict[str, Any] = {"parse_status": "complete", "cusip": table.empty_marker[1]}
    if ciks is not None:
        where["cik"] = sorted({pad_cik(cik) for cik in ciks})
    manifests = store.load(table, columns=columns, where=where, until=until, date_col="filing_date", markers=True, optional=True)
    if manifests is None:
        log.warning("No complete immutable 13F filings in scope; recover originals/amendments. Latest-only rows cannot supply PIT books.")
        return None
    numeric = ("shares", "value_usd", "call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value")
    row_columns = (
        "cik",
        "accession_number",
        "cusip",
        "period",
        "filing_date",
        "position_type",
        "class_evidence",
        *numeric,
        *("raw_" + col for col in numeric if "value" in col),
    )
    position_scope: dict[str, Any] = {"parse_status": "position"}
    if ciks is not None:
        position_scope["cik"] = where["cik"]
    chunks = list(store.iter_load(table, columns=row_columns, where=position_scope, until=until, date_col="filing_date"))
    holdings = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=row_columns)
    for column in ("period", "filing_date"):
        manifests[column] = pd.to_datetime(manifests[column])
        holdings[column] = pd.to_datetime(holdings[column])
    log.info(
        "Immutable 13F: %s complete accessions / %s whole-book rows / %s managers; unverified legacy rows are excluded.",
        len(manifests),
        len(holdings),
        manifests.cik.nunique(),
    )
    return manifests, holdings
