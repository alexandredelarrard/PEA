"""Resolve nominee roles from each proxy table's independently nearest prior eligible snapshot."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils.data_policy import source_read_window
from src.utils.filer_tables import def14a_evidence_mask
from src.utils.names import person_key
from src.utils.string import clean_text


def _role_source(context: Context, ticker: str) -> dict[str, pd.DataFrame]:
    """Read the three projected proxy sources once for this ticker, within their source windows."""
    # Load each table once before filtering the parent evidence.
    parent_window = source_read_window(context.config, Tables.def14a_llm, context.run.as_of)
    exec_window = source_read_window(context.config, Tables.def14a_executive_comp, context.run.as_of)
    director_window = source_read_window(context.config, Tables.def14a_director_comp, context.run.as_of)
    df_parent = (
        context.store.load(
            Tables.def14a_llm,
            columns=list(Tables.def14a_llm.read_columns),
            where={"ticker": ticker},
            since=parent_window.start,
            until=parent_window.end,
            optional=True,
        )
        if parent_window is not None
        else None
    )
    df_exec = (
        context.store.load(
            Tables.def14a_executive_comp,
            columns=["ticker", "as_of", "name", "title", "fiscal_year"],
            where={"ticker": ticker},
            since=exec_window.start,
            until=exec_window.end,
            optional=True,
        )
        if exec_window is not None
        else None
    )
    df_director = (
        context.store.load(
            Tables.def14a_director_comp,
            columns=["ticker", "as_of", "name"],
            where={"ticker": ticker},
            since=director_window.start,
            until=director_window.end,
            optional=True,
        )
        if director_window is not None
        else None
    )

    # A parent retry row does not establish an eligible proxy identity.
    parent = df_parent.loc[def14a_evidence_mask(df_parent)] if df_parent is not None else pd.DataFrame()
    return {
        "ceo": parent.reindex(columns=["ticker", "as_of", "ceo_name_proxy"]),
        "exec": df_exec if df_exec is not None else pd.DataFrame(),
        "director": df_director if df_director is not None else pd.DataFrame(),
    }


def _role_map(source: dict[str, pd.DataFrame], meeting_date: object) -> tuple[dict[str, str], dict[str, str]]:
    """Nearest-prior roles/titles, with CEO > executive > non-employee precedence."""
    # Resolve the latest eligible date independently for each table.
    prior: dict[str, pd.DataFrame] = {}
    for kind, df_rows in source.items():
        if df_rows.empty or "as_of" not in df_rows.columns or meeting_date is None:
            prior[kind] = df_rows.iloc[0:0]
            continue
        as_of = pd.to_datetime(df_rows["as_of"], errors="coerce")
        df_eligible = df_rows[as_of <= pd.Timestamp(cast(Any, meeting_date))]
        prior[kind] = df_eligible[pd.to_datetime(df_eligible["as_of"], errors="coerce") == as_of[df_eligible.index].max()]
    df_exec = prior["exec"]
    if not df_exec.empty and "fiscal_year" in df_exec.columns:
        years = pd.to_numeric(df_exec["fiscal_year"], errors="coerce")
        df_exec = df_exec[years == years.max()] if years.notna().any() else df_exec

    # Apply the same stored person key and precedence to the selected rosters.
    roles: dict[str, str] = {}
    titles: dict[str, str] = {}
    for _, row in prior["director"].iterrows():
        key = person_key(row.get("name"))
        if key:
            roles[key] = "non_employee"
    for _, row in df_exec.iterrows():
        key = person_key(row.get("name"))
        title = clean_text(row.get("title"))
        if key:
            roles[key] = "exec_officer"
        if key and title:
            titles[key] = title
    for _, row in prior["ceo"].iterrows():
        key = person_key(row.get("ceo_name_proxy"))
        if key:
            roles[key] = "ceo"
    return roles, titles
