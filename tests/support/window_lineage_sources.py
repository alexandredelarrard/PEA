"""Bounded own-ticker source selectors, retaining their complete production dependencies."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd

from src.context import Context
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.utils.common.inputs import load_source, source_read_scope
from src.data_aggregate.utils.common.pit import fiscal_values_to_daily
from src.data_aggregate.utils.fundamentals.employee_features import EMPLOYEE_COLUMNS, attach_headcount, headcount_proxy, part_time_weight
from src.data_aggregate.utils.governance.staleness import GOVERNANCE_EVENT_MAX_AGE_DAYS
from src.data_aggregate.utils.governance.vote_dissent_features import _say_on_pay_history, vote_dissent_fields
from src.data_aggregate.utils.institutionals.inputs import (
    FTD_LINE_COLUMNS,
    LINEAGE_COLUMNS,
    lag_insider_source,
    load_insider_role_evidence,
    load_insider_transactions,
)
from src.data_aggregate.utils.institutionals.insider_features import _first_filing, _rolling
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from src.data_aggregate.utils.institutionals.ownership_features import SUBJECT_COLUMNS, SubjectIdentity, _filing_counts, _filings
from src.data_store.schema import Table, Tables
from src.utils.publication import point_in_time
from src.validate.checks.window_lineage import _State
from tests.support.window_samples import cube_input_windows


def employees(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.fundamentals_employees
    windows = cube_input_windows(context, Tables.cube_part_fundamentals, sessions)
    raw = load_source(context, table, columns=[*EMPLOYEE_COLUMNS, "accession_number"], window=windows[table])
    history = StepCubeFundamentals(context, context.config)._load_fundamentals(tuple(context.universe.tickers), sessions, windows)
    if raw is None:
        raise ValueError("Pinned employees have no complete source rows")
    prepared = point_in_time(raw, context.config, table, sessions)
    proxy = headcount_proxy(prepared, part_time_weight(context.config.data_aggregate))
    merge = pd.merge_asof

    def observe(*args: Any, **kwargs: Any) -> pd.DataFrame:
        # Carry only the selected right-row identity through the existing pandas
        # selector. Ordering, keys, tolerance, direction and economic values stay exact.
        left, right = args
        return merge(left, right.assign(_selected_employee_asof=right["_on"]), **kwargs)

    with patch("src.data_aggregate.utils.fundamentals.employee_features.pd.merge_asof", side_effect=observe):
        selected = attach_headcount(history, proxy)
    selected = selected.loc[selected.headcount.notna() & selected._selected_employee_asof.notna()].copy()
    selected = selected.drop(columns=["_source_as_of", "accession_number"], errors="ignore")
    source = prepared.rename(columns={"as_of": "_selected_employee_asof"})
    selected = selected.merge(
        source[["ticker", "_selected_employee_asof", "_source_as_of", "accession_number"]],
        on=["ticker", "_selected_employee_asof"],
        validate="many_to_one",
    )
    selected["_projection_parent_asof"] = selected.as_of
    daily = fiscal_values_to_daily(selected, selected.headcount, sessions, max_age_days=460)
    return [
        _State(
            table,
            Tables.cube_part_fundamentals,
            selected,
            "headcount",
            daily,
            age_days=460,
            exception="Actual headcount proxy, backward financial-history attachment and fiscal daily projection; raw projected state before employee ratios/growth and peers.",
        )
    ]


def votes(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.sec_8k_votes
    columns = [
        "ticker",
        "accession_number",
        "proposal_seq",
        "filing_date",
        "proposal_type",
        "votes_for",
        "votes_against",
        "votes_abstain",
        "nominee_votes_json",
    ]
    windows = cube_input_windows(context, Tables.cube_part_governance, sessions)
    raw = load_source(context, table, columns=columns, window=windows[table])
    if raw is None:
        raise ValueError("Pinned vote units are absent")
    prepared = point_in_time(raw, context.config, table, sessions)
    prepared["filing_date"] = prepared["as_of"]
    fields, _ = vote_dissent_fields(prepared, sessions)
    history = _say_on_pay_history(prepared, {})
    if history is None or "sop_dissent" not in fields:
        raise ValueError("No observed say-on-pay feature state for pinned vote units")
    evidence = prepared.loc[prepared.proposal_type.eq("say_on_pay")].sort_values(["ticker", "filing_date", "accession_number", "proposal_seq"])
    units = {key: group[[*table.pk, "_source_filing_date"]].to_dict("records") for key, group in evidence.groupby(["ticker", "filing_date"])}
    evidence = evidence.drop_duplicates(["ticker", "filing_date"], keep="last")
    rows = history.merge(
        evidence[["ticker", "filing_date", "accession_number", "proposal_seq", "_source_filing_date"]],
        left_on=["ticker", "as_of"],
        right_on=["ticker", "filing_date"],
        validate="many_to_one",
    )
    rows["_source_units"] = [units[(row.ticker, row.as_of)] for row in rows.itertuples()]
    return [
        _State(
            table,
            Tables.cube_part_governance,
            rows,
            "sop_dissent",
            fields["sop_dissent"],
            age_days=GOVERNANCE_EVENT_MAX_AGE_DAYS,
            exception="Actual say-on-pay dissent and null barriers before peer transforms; complete same-publication ballot units recorded.",
        )
    ]


def ownership_identity(context: Context) -> SubjectIdentity:
    lineage = context.store.load(Tables.universe_cik_symbol, columns=LINEAGE_COLUMNS)
    lines = context.store.load(Tables.universe_cusip, columns=FTD_LINE_COLUMNS)
    return SubjectIdentity.from_tables(lineage, lines)


def ownership(context: Context, sessions: pd.DatetimeIndex, table: Table, identity: SubjectIdentity) -> list[_State]:
    # Identity filtering uses global full-history concentration statistics. Keep
    # the real universe population and all stored versions before selecting output keys.
    interval = cube_input_windows(context, Tables.cube_part_institutionals, sessions)[table]
    scope = source_read_scope(context, table, interval)
    where = {"ticker": context.universe.source_tickers, **scope.pop("where", {})}
    chunks = list(context.store.iter_load(table, columns=SUBJECT_COLUMNS, where=where, markers=True, **scope))
    if not chunks:
        raise ValueError("No full-population ownership source history")
    raw = pd.concat(chunks, ignore_index=True)
    selected = _filings(raw, identity)
    selected = selected.loc[selected.ticker.isin(context.universe.tickers)]
    prepared = point_in_time(selected, context.config, table, sessions)
    prepared["filing_date"] = prepared["as_of"]
    daily = _filing_counts(prepared, sessions, pd.Index(context.universe.tickers))
    units = raw.loc[raw.ticker.isin(context.universe.tickers)].groupby(["ticker", "accession_number"], as_index=False).filing_date.min()
    units["selected_filing"] = (
        pd.MultiIndex.from_frame(units[["ticker", "accession_number"]])
        .isin(pd.MultiIndex.from_frame(selected[["ticker", "accession_number"]]))
        .astype(float)
    )
    rows = point_in_time(units, context.config, table, sessions)
    references = {
        key: group[[*table.pk, "filing_date"]].to_dict("records")
        for key, group in raw.loc[raw.ticker.isin(context.universe.tickers)].groupby(["ticker", "accession_number"])
    }
    rows["_source_units"] = [references[(row.ticker, row.accession_number)] for row in rows.itertuples()]
    rows.attrs["lineage_dependency_scope"] = {
        "population_rows": len(raw),
        "population_tickers": sorted(raw.ticker.unique()),
        "retained_start": None if interval is None else interval.start,
        "retained_end": context.run.as_of,
        "markers": True,
    }
    return [
        _State(
            table,
            Tables.cube_part_institutionals,
            rows,
            "selected_filing",
            daily,
            lifetime_sessions=126,
            exception="Actual full-population subject classifier and126-session selected-accession count before coverage/peer transforms; full-history identity classification remains retrospective.",
        )
    ]


def insiders(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    windows = cube_input_windows(context, Tables.cube_part_institutionals, sessions)
    lines = context.store.load(Tables.universe_cusip, columns=FTD_LINE_COLUMNS, optional=True)
    source = load_insider_transactions(context, context.universe.tickers, lines, window=windows[Tables.insider_transactions])
    prices = load_source(
        context, Tables.cube_part_prices, columns=["date", "ticker", "close_pit", "shares_outstanding_pit"], window=windows[Tables.cube_part_prices]
    )
    if source is None or prices is None:
        raise ValueError("Missing complete insider history or actual saved PIT valuation dependencies")
    source.transactions["_source_filing_date"] = source.transactions.filing_date
    lagged = lag_insider_source(source, context.config, sessions)
    roles = load_insider_role_evidence(context, lagged.transactions, context.universe.tickers, sessions, window=windows[Tables.def14a_executive_comp])
    quotes = prices.pivot(index="date", columns="ticker", values="close_pit")
    shares = prices.pivot(index="date", columns="ticker", values="shares_outstanding_pit")
    trades, _ = clean_transactions(lagged.transactions, df_valuation_close=quotes, df_roles=roles, df_shares_outstanding=shares)
    trades = trades.loc[~trades.quality_uncertain.fillna(False)].copy()
    seen = _first_filing(trades.loc[trades.code.eq("P")], trades.loc[trades.code.eq("S")], sessions)
    daily = _rolling(trades, sessions, 20, "shares_n", seen)
    trades["as_of"] = trades.visible_from
    return [
        _State(
            Tables.insider_transactions,
            Tables.cube_part_institutionals,
            trades,
            "shares_n",
            daily,
            age_days=20,
            exception="Actual complete-version/copy/quality selector and20-calendar-day disclosed-share state; before economic ratios, account conviction, population and peer transforms.",
        )
    ]
