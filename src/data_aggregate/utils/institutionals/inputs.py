"""Insider, source lineage and short-flow source inputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from src.constants.constants import CANONICAL_ROLES
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.institutionals.insider_quality import OTHER_OFFICER, clean_officer_title, officer_role, owner_name_key
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.price_adjustments import ShareBasis
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)
CLASS_LINE_COLUMNS = ("canonical_company", "market_symbol", "conversion_ratio", "valid_from", "valid_to")


@dataclass(frozen=True)
class QuantityBasis:
    """Genuine ordinary-share event evidence and one full-context terminal basis date."""

    events: ShareBasis
    target_date: pd.Timestamp

    @classmethod
    def from_context(cls, events: ShareBasis, calendar: pd.DatetimeIndex) -> QuantityBasis:
        """Anchor to the full vendor tape calendar; later event metadata cannot advance its restatement basis."""
        if calendar.empty:
            raise ValueError("Quantity basis requires a nonempty full calculation calendar")
        return cls(events, pd.Timestamp(calendar.max()))


def short_quantity_history(df_history: pd.DataFrame | None, basis: QuantityBasis) -> pd.DataFrame | None:
    """Raw dated FINRA aggregates already class-converted upstream, on the target ordinary-share basis."""
    if df_history is None or df_history.empty:
        return df_history
    df_rows = df_history.copy()
    target = pd.Series(basis.target_date, index=df_rows.index)
    for column in ("short_volume", "total_volume"):
        if column in df_rows:
            df_rows[column] = basis.events.rebase(df_rows[column], df_rows["ticker"], df_rows["date"], target)
    return df_rows


def daily_share_quantities(df_history: pd.DataFrame | None, idx: pd.DatetimeIndex, basis: QuantityBasis) -> pd.DataFrame | None:
    """Project as-of ordinary PIT counts after rebasing to the common target; vendor valuation counts stay intact."""
    if df_history is None or df_history.empty or "sharesOutstandingPit" not in df_history:
        return None
    df_rows = df_history[["ticker", "as_of", "sharesOutstandingPit"]].copy()
    target = pd.Series(basis.target_date, index=df_rows.index)
    df_rows["sharesOutstandingPit"] = basis.events.rebase(df_rows["sharesOutstandingPit"], df_rows["ticker"], df_rows["as_of"], target)
    df_daily = fundamentals_to_daily(df_rows, "sharesOutstandingPit", idx)
    return df_daily if not df_daily.empty and df_daily.notna().any().any() else None


def tape_share_quantities(df_volume: pd.DataFrame | None, basis: QuantityBasis) -> pd.DataFrame | None:
    """Yahoo tape is already terminal-restated; only its explicitly evidenced quoted-security units need conversion."""
    if df_volume is None or df_volume.empty:
        return df_volume
    tickers = pd.Series(df_volume.columns, index=df_volume.columns)
    dates = pd.Series(basis.target_date, index=df_volume.columns)
    units = basis.events.factor(tickers, dates, "units")
    return df_volume.mul(units, axis="columns")


def load_finra_tenure(store: DataStore, universe: Sequence[str]) -> pd.DataFrame | None:
    """Observed fallback stamps already resolved by FINRA's dated owner/class mapper; missing dates stay unknown."""
    if not universe or not store.exists(Tables.sec_short_volume_security):
        return None
    columns = ("date", "ticker", "security_id", "lineage_role", "security_class")
    chunks = []
    for chunk in store.iter_load(
        Tables.sec_short_volume_security, columns=columns, where={"ticker": sorted(set(universe)), "lineage_role": list(CANONICAL_ROLES)}
    ):
        selected = chunk["security_id"].fillna("").str.fullmatch(r"S\d{10}:[A-Z0-9][A-Z0-9.-]*")
        chunks.append(chunk.loc[selected].drop_duplicates())
    return pd.concat(chunks, ignore_index=True).drop_duplicates() if chunks else None


@dataclass(frozen=True)
class InsiderSource:
    transactions: pd.DataFrame
    complete_through: pd.Series
    first_filed: pd.Series


def load_insider_transactions(store: DataStore, universe: Sequence[str] | None = None) -> InsiderSource | None:
    """Retain full issuer proof metadata, then select canonical economic rows and actual empty markers."""
    table = Tables.insider_transactions
    present = store.columns(table)
    if not present:
        logger.warning("%s is absent -> insider features are skipped.", table.name)
        return None
    columns = [column for column in dict.fromkeys((*table.read_columns, "issuer_cik", "lineage_role")) if column in present]
    where: dict[str, Any] = {}
    if universe is not None:
        where["ticker"] = sorted(set(universe))
    chunks = list(store.iter_load(table, columns=columns, where=where, markers=True))
    if not chunks:
        return None
    df_rows = pd.concat(chunks, ignore_index=True)
    if not {"ticker", "filing_date", "security_type"}.issubset(df_rows):
        logger.warning("%s lacks required source metadata -> insider features are skipped.", table.name)
        return None
    checked = pd.to_datetime(df_rows.get("inventory_checked_through", pd.Series(pd.NaT, index=df_rows.index)), errors="coerce")
    fetched = pd.to_datetime(df_rows.get("fetched_at", pd.Series(pd.NaT, index=df_rows.index)), errors="coerce")
    evidence = pd.DataFrame({"ticker": df_rows["ticker"], "checked": checked, "valid": checked.notna() & fetched.notna()})
    grouped = evidence.groupby("ticker", sort=False)
    joint = grouped["valid"].all() & grouped["checked"].nunique().eq(1)
    boundaries = grouped["checked"].min().where(joint)
    first = pd.to_datetime(df_rows["filing_date"], errors="coerce").groupby(df_rows["ticker"]).min()
    roles = df_rows.get("lineage_role", pd.Series(index=df_rows.index, dtype=str))
    selected = roles.isin(CANONICAL_ROLES) | df_rows["security_type"].eq("_empty")
    logger.info("Loaded %s: %s rows x %s cols", table.name, len(df_rows), len(df_rows.columns))
    return InsiderSource(df_rows.loc[selected], boundaries, first)


def load_insider_role_evidence(store: DataStore, insider: pd.DataFrame) -> pd.DataFrame:
    """Projected proxy title evidence for unique same-issuer/full-name owner identities only."""
    columns = ["ticker", "issuer_cik", "owner_cik", "role_as_of", "officer_title_clean", "role", "role_source"]
    if not {"issuer_cik", "owner_cik", "owner_name", "officer_title"}.issubset(insider):
        return pd.DataFrame(columns=columns)
    blank_title = insider["officer_title"].fillna("").astype(str).str.strip().eq("")
    if not blank_title.any():
        return pd.DataFrame(columns=columns)
    ticks = sorted(insider.loc[blank_title, "ticker"].dropna().astype(str).unique())

    # Load only proxy rows for tickers with a blank filed title.
    df_proxy = store.load(
        Tables.def14a_executive_comp, columns=["ticker", "cik", "as_of", "name", "title", "fiscal_year"], where={"ticker": ticks}, optional=True
    )
    if df_proxy is None or df_proxy.empty:
        return pd.DataFrame(columns=columns)

    # Match dated public titles only after an actual proxy payload is available.
    df_owners = _role_owner_identities(insider)
    df_roles = _match_proxy_roles(df_proxy, df_owners)
    return df_roles[columns].drop_duplicates()


def _role_owner_identities(df_insider: pd.DataFrame) -> pd.DataFrame:
    """Earliest public owner identity per issuer and strict full-name key."""
    identity = ["ticker", "issuer_cik", "name_key"]
    df_owners = df_insider[["ticker", "issuer_cik", "owner_cik", "owner_name", "filing_date"]].copy()

    # Canonicalize each owner before selecting its earliest public identity date.
    df_owners["filing_date"] = pd.to_datetime(df_owners["filing_date"], errors="coerce").dt.normalize()
    df_owners = df_owners.sort_values("filing_date").drop_duplicates(["ticker", "issuer_cik", "owner_cik", "owner_name"])
    df_owners["name_key"] = df_owners["owner_name"].map(owner_name_key)
    for column in ("issuer_cik", "owner_cik"):
        df_owners[column] = df_owners[column].map(pad_cik).fillna("")
    df_owners["identity_from"] = df_owners["filing_date"]
    df_owners = df_owners[df_owners["name_key"].ne("") & df_owners["owner_cik"].ne("")].dropna(subset=["identity_from"])
    return cast(pd.DataFrame, df_owners.groupby([*identity, "owner_cik"], as_index=False)["identity_from"].min())


def _match_proxy_roles(df_proxy: pd.DataFrame, df_owners: pd.DataFrame) -> pd.DataFrame:
    """Match same-issuer owners and preserve public ambiguity/conflict barriers."""
    identity = ["ticker", "issuer_cik", "name_key"]

    # Normalize explicit titles and retain the latest fiscal-year record per publication.
    df_proxy["issuer_cik"] = df_proxy["cik"].map(pad_cik).fillna("")
    df_proxy["name_key"] = df_proxy["name"].map(owner_name_key)
    df_proxy["role_as_of"] = pd.to_datetime(df_proxy["as_of"], errors="coerce").dt.normalize()
    df_proxy["officer_title_clean"] = df_proxy["title"].map(clean_officer_title)
    df_proxy["role"] = df_proxy["officer_title_clean"].map(officer_role)
    latest = df_proxy.groupby(["ticker", "as_of"])["fiscal_year"].transform("max")
    df_proxy = df_proxy[df_proxy["fiscal_year"].eq(latest)].dropna(subset=["role_as_of"])
    # Explicit non-management/former titles are history barriers; parent CEO fallback is not evidence.
    df_proxy = df_proxy[df_proxy["officer_title_clean"].ne("")]
    df_proxy = df_proxy.merge(df_owners, on=identity, how="inner")
    df_proxy["role_as_of"] = df_proxy[["role_as_of", "identity_from"]].max(axis=1)
    # Name uniqueness is evaluated when the mapping becomes public, never against future owners.
    identities = df_owners.set_index(identity)
    for _, group in df_proxy.groupby(identity, sort=False):
        first = identities.loc[tuple(group.iloc[0][identity])]["identity_from"]
        if isinstance(first, pd.Timestamp):
            continue
        ambiguous = group["role_as_of"].map(lambda day, first=first: first.le(day).sum() > 1)
        df_proxy.loc[group.index[ambiguous], "role"] = OTHER_OFFICER
    df_proxy["role_source"] = "explicit_proxy_title"
    conflicted = df_proxy.groupby(["ticker", "issuer_cik", "owner_cik", "role_as_of"])["role"].transform("nunique").gt(1)
    df_proxy.loc[conflicted, ["role", "officer_title_clean"]] = [OTHER_OFFICER, "conflicting roles"]
    return df_proxy


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
