"""Bounded real-source replay of governance expiry and invalidation."""

from __future__ import annotations

from copy import copy
from typing import Any

import pandas as pd
import pytest

from src.context import get_config_context
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.governance.clean import GovernanceInputs, clean_governance_inputs
from src.data_aggregate.utils.governance.panel import _ceo_pay_growth, build_governance_feature_panel
from src.data_aggregate.utils.governance.pay_features import _comp_history, pay_fields
from src.data_aggregate.utils.governance.staleness import expire_stale
from src.data_aggregate.utils.governance.vote_dissent_features import vote_dissent_fields
from src.data_store.schema import Tables
from src.validate.checks.bounds import check_bounds
from src.validate.io import write_result


@pytest.fixture(scope="module")
def governance_sources() -> dict[str, Any]:
    config, context = get_config_context("./configs", use_cache=False, save=False)
    store = context.store
    names = ["AAPL", "MSFT", "AME", "EIX", "ETN", "GWW", "OXY", "TROW", "WFC", "JPM", "BAC", "JNJ"]
    if not store.exists(Tables.def14a_llm):
        pytest.skip("Real governance sources are unavailable")
    where = {"ticker": names}
    parent_cols = [c for c in store.columns(Tables.def14a_llm) if c != "def14a_json"]
    parent = store.load(Tables.def14a_llm, columns=parent_cols, where=where, since="2008-01-01", until="2026-10-08")
    raw = GovernanceInputs(
        def14a=parent,
        exec_comp=store.load(
            Tables.def14a_executive_comp, columns=store.columns(Tables.def14a_executive_comp), where=where, since="2008-01-01", until="2026-10-08"
        ),
        directors=store.load(Tables.def14a_directors, project=True, where=where, since="2008-01-01", until="2026-10-08"),
        director_comp=store.load(Tables.def14a_director_comp, project=True, where=where, since="2008-01-01", until="2026-10-08"),
        votes=store.load(Tables.sec_8k_votes, columns=store.columns(Tables.sec_8k_votes), where=where, since="2008-01-01", until="2026-10-08"),
    )
    wanted = ["ticker", "as_of", "fiscal_end", "fiscal_year", "fiscal_quarter", "totalRevenue", "sharesOutstandingPit"]
    fundamentals = store.load(
        Tables.fundamentals_history,
        columns=[c for c in wanted if c in store.columns(Tables.fundamentals_history)],
        where=where,
        since="2007-01-01",
        until="2026-10-08",
    )
    prices = store.load(Tables.cube_part_prices, columns=["date", "ticker", "close_total"], where=where, since="2010-01-01", until="2026-10-08")
    close = prices.pivot(index="date", columns="ticker", values="close_total").sort_index()
    close.index = pd.DatetimeIndex(close.index)
    inputs, _ = clean_governance_inputs(raw, fundamentals)
    return {
        "config": config,
        "context": context,
        "inputs": inputs,
        "fundamentals": fundamentals,
        "close": close,
        "peers": load_peers_or_raise(context, config),
    }


def test_real_ceo_changes_end_pay_measurements(governance_sources: dict[str, Any]) -> None:
    source = governance_sources
    inputs, close = source["inputs"], source["close"]
    hist = _comp_history(inputs.def14a, {})
    frames, _ = pay_fields(inputs.def14a, inputs.exec_comp, source["fundamentals"], close, source["peers"], close.index)
    legacy = _ceo_pay_growth(inputs.def14a, close.index)
    changes = hist.loc[(hist["ceo_turnover_flag"] == 1) & hist["as_of"].between(close.index.min(), close.index.max())]
    assert len(changes) >= 10
    for row in changes.itertuples():
        day = close.index[close.index >= row.as_of][0]
        assert pd.isna(frames["ceo_comp_growth_1y"].loc[day, row.ticker]), (row.ticker, day)
        assert pd.isna(legacy.loc[day, row.ticker]), (row.ticker, day)
        assert pd.isna(frames["pay_return_gap"].loc[day, row.ticker]), (row.ticker, day)
    print(f"SANITY: {len(changes)} real CEO transitions immediately stop old pay growth and pay-return gaps.")


def test_real_votes_are_prefix_invariant(governance_sources: dict[str, Any]) -> None:
    source = governance_sources
    votes, idx = source["inputs"].votes, source["close"].index
    cutoff = pd.Timestamp("2021-12-31")
    full, _ = vote_dissent_fields(votes, idx)
    prefix, _ = vote_dissent_fields(votes.loc[pd.to_datetime(votes["filing_date"]) <= cutoff], idx[idx <= cutoff])
    for field, frame in prefix.items():
        pd.testing.assert_frame_equal(frame, full[field].reindex(index=frame.index, columns=frame.columns))
    print(f"SANITY: {len(prefix)} real vote features are unchanged when later filings are removed.")


def test_real_carry_keeps_the_original_source_clock(governance_sources: dict[str, Any]) -> None:
    hist = governance_sources["inputs"].def14a
    field = "avg_board_tenure"
    rows = hist.loc[hist[field].notna() & hist["ticker"].eq("WFC")].sort_values("as_of")
    row = rows.iloc[-1:]
    source = pd.to_datetime(row[f"{field}_source_as_of"].iloc[0])
    idx = pd.date_range(source, periods=397)
    daily = pd.DataFrame(float(row[field].iloc[0]), index=idx, columns=["WFC"])
    capped = expire_stale(daily, row, field)
    # A carried row can be newer than its ultimate source, and must not restart the clock.
    assert capped.loc[source + pd.Timedelta(days=395), "WFC"] == pytest.approx(row[field].iloc[0])
    assert pd.isna(capped.loc[source + pd.Timedelta(days=396), "WFC"])
    print("SANITY: an actual WFC board observation expires from its original public source on day 396.")


@pytest.fixture(scope="module")
def rebuilt_governance(governance_sources: dict[str, Any]) -> pd.DataFrame:
    source = governance_sources
    inputs, close = source["inputs"], source["close"]
    return build_governance_feature_panel(
        inputs.def14a,
        source["peers"],
        close.index,
        fundamentals_history=source["fundamentals"],
        votes=inputs.votes,
        exec_comp=inputs.exec_comp,
        directors=inputs.directors,
        director_comp=inputs.director_comp,
        close_total=close,
        availability=close.notna(),
    )[0]


def test_real_panel_window_matches_full_tail(governance_sources: dict[str, Any], rebuilt_governance: pd.DataFrame) -> None:
    source = governance_sources
    inputs, close = source["inputs"], source["close"]
    window = close.iloc[-(1260 + 756) :]
    panel, _ = build_governance_feature_panel(
        inputs.def14a,
        source["peers"],
        window.index,
        fundamentals_history=source["fundamentals"],
        votes=inputs.votes,
        exec_comp=inputs.exec_comp,
        directors=inputs.directors,
        director_comp=inputs.director_comp,
        close_total=window,
        availability=window.notna(),
    )
    cutoff = close.index[-756]
    full = rebuilt_governance.loc[rebuilt_governance["date"] >= cutoff].set_index(["date", "ticker"]).sort_index()
    tail = panel.loc[panel["date"] >= cutoff].set_index(["date", "ticker"]).sort_index()
    pd.testing.assert_frame_equal(full, tail.reindex(index=full.index, columns=full.columns), check_exact=False, atol=1e-6, rtol=1e-6)
    print(f"SANITY: the real three-year incremental tail matches a full in-memory rebuild on {len(full):,} rows.")


def test_rebuilt_panel_passes_read_only_bounds_validator(
    governance_sources: dict[str, Any], rebuilt_governance: pd.DataFrame, sqlite_store: Any
) -> None:
    context = copy(governance_sources["context"])
    context.store = sqlite_store
    context.store.replace(Tables.cube_part_governance, rebuilt_governance)
    bounded = {
        field: (0.0, 1.0)
        for field in [
            "f_founder_ceo",
            "f_pct_long_tenured",
            "f_board_dissent_breadth_10",
            "f_sop_dissent_gt_10",
            "f_ceo_pay_slice",
            "f_board_independence_drop_10pp",
        ]
        if field in rebuilt_governance
    }
    result = check_bounds(context, Tables.cube_part_governance, config=governance_sources["config"], bounds=bounded)
    write_result("reports/validate/2026-10-15-governance-freshness", result)
    assert result.status == "pass", result.summary()
    print(f"SANITY: existing read-only bounds validator: {result.summary()}; only an isolated SQLite store was written.")
