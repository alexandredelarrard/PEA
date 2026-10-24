"""Insider, source lineage and short-flow source inputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.constants.constants import CANONICAL_ROLES
from src.data_aggregate.utils.institutionals.insider_quality import OTHER_OFFICER, clean_officer_title, officer_role, owner_name_key
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)
TENURE_SOURCES = ("form345", "manual")
CLASS_LINE_COLUMNS = ("canonical_company", "market_symbol", "conversion_ratio", "valid_from", "valid_to")


def load_symbol_lineage(
    store: DataStore,
    universe: Sequence[str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load the issuer lineage (`form345`/`manual` tenure on the roster CIKs) used to validate already-canonical source rows."""
    tickers = sorted(set(universe))
    # The roster determines the bounded issuer scope for the tenure read.
    df_ticker_ciks = store.load(
        Tables.sp500_tickers,
        columns=("ticker", "cik"),
        where={"ticker": tickers},
        optional=True,
    )
    if df_ticker_ciks is None or df_ticker_ciks.empty:
        return None, df_ticker_ciks

    ciks = sorted({cik for cik in df_ticker_ciks["cik"].map(pad_cik) if cik})
    if not ciks:
        return None, df_ticker_ciks
    df_symbol_tenure = store.load(
        Tables.symbol_tenure,
        columns=("symbol", "issuer_cik", "valid_from", "valid_to"),
        where={"issuer_cik": ciks, "source": list(TENURE_SOURCES)},
        optional=True,
    )

    # Report the loaded lineage without another payload read.
    aliases = len(set(df_symbol_tenure["symbol"].astype(str))) if df_symbol_tenure is not None else 0
    logger.info("Symbol lineage: %s current tickers backed by %s proven historical symbols", len(tickers), aliases)
    return df_symbol_tenure, df_ticker_ciks


def load_insider_transactions(store: DataStore, universe: Sequence[str] | None = None) -> pd.DataFrame | None:
    """`insider_transactions` rows of the companies' own history: `lineage_role` canonical (predecessor or current).
    Acquired-constituent rows and rows without a stamp are not read."""
    table = Tables.insider_transactions
    present = store.columns(table)
    if not present:
        logger.warning("%s is absent -> insider features are skipped.", table.name)
        return None
    columns = [column for column in dict.fromkeys((*table.read_columns, "issuer_cik")) if column in present]
    where: dict[str, Any] = {"lineage_role": list(CANONICAL_ROLES)}
    if universe is not None:
        where["ticker"] = sorted(set(universe))
    chunks = list(store.iter_load(table, columns=columns, where=where))
    if not chunks:
        return None
    df_rows = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %s: %s rows x %s cols", table.name, len(df_rows), len(df_rows.columns))
    return df_rows


def load_insider_role_evidence(store: DataStore, insider: pd.DataFrame) -> pd.DataFrame:
    """Projected proxy title evidence for unique same-issuer/full-name owner identities only."""
    columns = ["ticker", "issuer_cik", "owner_cik", "role_as_of", "officer_title_clean", "role", "role_source"]
    if not {"issuer_cik", "owner_cik", "owner_name", "officer_title"}.issubset(insider):
        return pd.DataFrame(columns=columns)
    candidates = insider[insider["officer_title"].fillna("").astype(str).str.strip().eq("")].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    ticks = sorted(candidates["ticker"].dropna().astype(str).unique())
    df_proxy = store.load(
        Tables.def14a_executive_comp, columns=["ticker", "cik", "as_of", "name", "title", "fiscal_year"], where={"ticker": ticks}, optional=True
    )
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


def load_fails(store: DataStore, universe: Sequence[str]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Keep global ZIP/date coverage while retaining only requested economic rows."""
    coverage_chunks, ticker_chunks = [], []

    # Preserve market-wide observed dates before discarding other tickers' rows.
    for df_chunk in store.iter_load(Tables.sec_fails_to_deliver, project=True):
        coverage_chunks.append(df_chunk[["date", "period"]].drop_duplicates())
        ticker_chunks.append(df_chunk.loc[df_chunk["ticker"].isin(universe)])
    if not coverage_chunks:
        return None, None

    # Assemble the scoped economics and compact global coverage independently.
    return pd.concat(ticker_chunks, ignore_index=True), pd.concat(coverage_chunks, ignore_index=True).drop_duplicates()
