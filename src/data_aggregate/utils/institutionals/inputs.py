"""Insider, source lineage and short-flow source inputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from src.constants.constants import CANONICAL_ROLES
from src.data_aggregate.utils.institutionals.insider_quality import (
    OTHER_OFFICER,
    clean_officer_title,
    insider_days,
    normalize_insider_dates,
    officer_role,
    owner_name_key,
)
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.identity_tables import read_identity
from src.utils.security_series import LINE_COLUMNS, SHORT_LINE_COLUMNS, fails_series, load_fails_basis, prepare_master, short_volume_series
from src.utils.string import pad_cik, yahoo_symbol

logger = logging.getLogger(__name__)

CLASS_LINE_COLUMNS = ("canonical_company", "market_symbol", "conversion_ratio", "valid_from", "valid_to")
#: The `universe_cusip` columns of the FTD lines the cube maps the security-grain sources with.
FTD_LINE_COLUMNS = (*CLASS_LINE_COLUMNS, "security_id", "source", "source_symbol", "security_class", "lineage_role", "cusip")
#: The `universe_cik_symbol` columns the short-volume lineage fallback reads.
LINEAGE_COLUMNS = ("company_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources")
#: The share-class letter of an insider `security_title` ("Class A Common Stock", "Series B").
_CLASS_LETTER = r"(?i)\b(?:CLASS|SERIES)\s+([A-Z])\b"


def load_ftd_lines(store: DataStore) -> pd.DataFrame | None:
    """The FTD `universe_cusip` lines (`FTD_LINE_COLUMNS`); None before the first identity build."""
    return read_identity(store, Tables.universe_cusip, columns=FTD_LINE_COLUMNS, where={"source": "ftd"}, optional=True)


def load_class_bars(store: DataStore, class_lines: pd.DataFrame | None) -> pd.DataFrame:
    """Projected raw secondary-class volume bars, read once."""
    symbols = sorted(set(class_lines["market_symbol"].map(yahoo_symbol))) if class_lines is not None else []
    df_bars = store.load(Tables.prices, columns=["ticker", "date", "volume"], where={"ticker": symbols}, optional=True) if symbols else None
    return df_bars if df_bars is not None else pd.DataFrame(columns=["ticker", "date", "volume"])


def load_short_volume(
    store: DataStore, universe: Sequence[str], lines: pd.DataFrame | None, lineage: pd.DataFrame | None
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The short-volume series of the `universe` companies and the lineage-fallback stamps the tenure mask reads, built
    from the as-filed FINRA lines, the step's `universe_cusip` lines and its `universe_cik_symbol` rows
    (`security_series.short_volume_series`)."""
    # one read of the stored lines
    stored = store.load(Tables.sec_short_volume_security, columns=list(SHORT_LINE_COLUMNS), optional=True)
    if stored is None or stored.empty or lines is None or lineage is None or not universe:
        logger.warning("%s or the identity tables are absent -> short-volume features are skipped.", Tables.sec_short_volume_security.name)
        return None, None

    # every universe company's series and its fallback stamps
    series, fallback = short_volume_series(stored, lines, lineage, universe)
    logger.info("Loaded %s: %s lines -> %s series rows", Tables.sec_short_volume_security.name, len(stored), len(series))
    return series, (fallback if not fallback.empty else None)


@dataclass(frozen=True)
class InsiderSource:
    """Canonical rows and empty-filing markers, the table-wide frontier (newest stored `filing_date`) and each
    ticker's first stored filing date."""

    transactions: pd.DataFrame
    complete_through: pd.Timestamp | None
    first_filed: pd.Series


def load_insider_transactions(store: DataStore, universe: Sequence[str] | None, lines: pd.DataFrame | None) -> InsiderSource | None:
    """Retain full issuer metadata, then select canonical economic rows and actual empty markers."""
    table = Tables.insider_transactions
    present = store.columns(table)
    if not present:
        logger.warning("%s is absent -> insider features are skipped.", table.name)
        return None

    # keep columns needed & read data in chunks; the frontier is the whole table's newest filing
    complete_through = store.max_date(table, "filing_date")
    columns = set(present).intersection(set((*table.read_columns, "issuer_cik", "lineage_role", "economic_date")))
    where: dict[str, Any] = {}
    if universe is not None:
        where["ticker"] = sorted(set(universe))
    chunks = list(store.iter_load(table, columns=list(columns), where=where, markers=True))
    if not chunks:
        return None

    df_rows = pd.concat(chunks, ignore_index=True)
    if not {"ticker", "filing_date", "security_type"}.issubset(df_rows):
        logger.warning("%s lacks required source metadata -> insider features are skipped.", table.name)
        return None

    # shares as filed -> canonical-class shares; clean dates of insider
    df_rows = normalize_insider_dates(canonical_class_units(df_rows, lines))
    first = df_rows["filing_date"].groupby(df_rows["ticker"]).min()
    selected = df_rows["lineage_role"].isin(CANONICAL_ROLES) | df_rows["security_type"].eq("_empty")

    logger.info("Loaded %s: %s rows x %s cols", table.name, len(df_rows), len(df_rows.columns))
    return InsiderSource(df_rows.loc[selected], complete_through, first)


def canonical_class_units(df_rows: pd.DataFrame, lines: pd.DataFrame | None) -> pd.DataFrame:
    """`df_rows` (as filed) with each non-derivative row of a class traded at a contractual ratio in canonical-class
    shares: `shares` and `shares_owned_after` x the `universe_cusip` line's `conversion_ratio` of the row's ticker and
    `security_title` class ("Class A ...") holding its `economic_date`, `price_per_share` / it; `value_usd` unchanged."""
    ratio_lines = None if lines is None else lines.loc[pd.to_numeric(lines["conversion_ratio"], errors="coerce").ne(1.0)]
    if ratio_lines is None or ratio_lines.empty or df_rows.empty:
        return df_rows

    # compute: each non-derivative class row's ratio from the line of its company and class covering its economic date
    letters = df_rows["security_title"].astype("string").str.extract(_CLASS_LETTER, expand=False).str.upper()
    day = pd.to_datetime(df_rows.reindex(columns=["economic_date"])["economic_date"], errors="coerce")
    rows = pd.DataFrame({"canonical_company": df_rows["ticker"], "security_class": ("class_" + letters).astype(object), "day": day})
    rows = rows.loc[df_rows["security_type"].eq("nonderiv") & letters.notna()].reset_index()
    hits = rows.merge(ratio_lines[["canonical_company", "security_class", "valid_from", "valid_to", "conversion_ratio"]])
    start, end = pd.to_datetime(hits["valid_from"]), pd.to_datetime(hits["valid_to"])
    hits = hits.loc[hits["day"].ge(start) & (hits["day"].lt(end) | end.isna())].drop_duplicates("index", keep="last")
    if hits.empty:
        return df_rows
    ratio = pd.Series(1.0, index=df_rows.index)
    ratio[hits["index"].to_numpy()] = pd.to_numeric(hits["conversion_ratio"]).to_numpy()

    # post-process: canonical-class quantities, the price per canonical share
    out = df_rows.copy()
    for column in ("shares", "shares_owned_after"):
        out[column] = pd.to_numeric(out[column], errors="coerce") * ratio
    out["price_per_share"] = pd.to_numeric(out["price_per_share"], errors="coerce") / ratio
    return out


def load_insider_role_evidence(store: DataStore, insider: pd.DataFrame, universe: Sequence[str] | None = None) -> pd.DataFrame:
    """Projected proxy title evidence for unique same-issuer/full-name owner identities only."""

    columns = list(Tables.def14a_executive_comp.read_columns)
    if not {"issuer_cik", "owner_cik", "owner_name", "officer_title"}.issubset(insider):
        return pd.DataFrame(columns=columns)

    # Load only proxy rows for tickers with a missing job claim.
    df_proxy = store.load(
        Tables.def14a_executive_comp, columns=["ticker", "cik", "as_of", "name", "title", "fiscal_year"], where={"ticker": universe}, optional=True
    )
    if df_proxy is None or df_proxy.empty:
        return pd.DataFrame(columns=columns)

    # Match dated public titles only after an actual proxy payload is available.
    df_proxy = df_proxy.loc[~df_proxy["cik"].isnull()].copy()
    df_owners = _role_owner_identities(insider.loc[insider["issuer_cik"].map(pad_cik).fillna("").ne("")])
    if df_proxy.empty or df_owners.empty:
        return pd.DataFrame(columns=columns)
    df_roles = _match_proxy_roles(df_proxy, df_owners)
    return df_roles[columns].drop_duplicates()


def _role_owner_identities(df_insider: pd.DataFrame) -> pd.DataFrame:
    """Earliest public owner identity per issuer and strict full-name key."""
    df_owners = df_insider[["ticker", "issuer_cik", "owner_cik", "owner_name", "filing_date"]].copy()

    # Canonicalize each owner before selecting its earliest public identity date.
    df_owners["filing_date"] = insider_days(df_owners["filing_date"])
    df_owners = df_owners.sort_values("filing_date").drop_duplicates(["ticker", "issuer_cik", "owner_cik", "owner_name"])
    df_owners["name_key"] = df_owners["owner_name"].map(owner_name_key)
    for column in ("issuer_cik", "owner_cik"):
        df_owners[column] = df_owners[column].map(pad_cik).fillna("")
    df_owners["identity_from"] = df_owners["filing_date"]
    df_owners = df_owners[df_owners["name_key"].ne("") & df_owners["owner_cik"].ne("")].dropna(subset=["identity_from"])
    return cast(pd.DataFrame, df_owners.groupby(["ticker", "issuer_cik", "name_key", "owner_cik"], as_index=False)["identity_from"].min())


def _match_proxy_roles(df_proxy: pd.DataFrame, df_owners: pd.DataFrame) -> pd.DataFrame:
    """Match same-issuer owners and preserve public ambiguity/conflict barriers."""
    identity = ["ticker", "issuer_cik", "name_key"]

    # Normalize explicit titles and retain the latest fiscal-year record per publication.
    df_proxy["issuer_cik"] = df_proxy["cik"].map(pad_cik).fillna("")
    df_proxy["name_key"] = df_proxy["name"].map(owner_name_key)
    df_proxy["role_as_of"] = insider_days(df_proxy["as_of"])
    df_proxy["officer_title_clean"] = df_proxy["title"].map(clean_officer_title)
    df_proxy = df_proxy[df_proxy["officer_title_clean"].ne("")]
    df_proxy["role"] = df_proxy["officer_title_clean"].map(officer_role)
    latest = df_proxy.groupby(["ticker", "as_of"])["fiscal_year"].transform("max")
    df_proxy = df_proxy[df_proxy["fiscal_year"].eq(latest)].dropna(subset=["role_as_of"])

    # Explicit non-management/former titles are history barriers; parent CEO fallback is not evidence.
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


def load_fails(
    store: DataStore, universe: Sequence[str], lines: pd.DataFrame | None, bugfix: dict
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The FTD series of the `universe` companies and the market-wide served calendar, built from the as-filed security
    lines and the step's `universe_cusip` lines (`security_series.fails_series`)."""
    # one read of the stored lines and of the share basis of the FTD lines' classes and companies
    stored = store.load(Tables.sec_fails_to_deliver_security, columns=list(LINE_COLUMNS), optional=True)
    master = prepare_master(lines) if lines is not None else None
    if stored is None or stored.empty or master is None or master.empty:
        return None, None
    basis = load_fails_basis(store, bugfix, master)

    # every company's series and the served dates; the economics of the universe only
    series, coverage = fails_series(stored, master, basis)
    return series.loc[series["ticker"].isin(set(universe))].reset_index(drop=True), coverage
