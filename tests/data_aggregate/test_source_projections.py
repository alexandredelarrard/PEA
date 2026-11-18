"""Real streaming-store contracts for the governance and fundamental event sources."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.utils.common.inputs import load_source
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.window import Window, WindowRun
from tests.support.config import fixture_config
from tests.support.universe import universe_of


def _payload(table: Table) -> dict[str, object]:
    """Keep independent known-truth raw fields, including the producer's role buckets."""
    if table == Tables.dividends:
        return {"ticker": "AAPL", "date": "2026-01-15", "dividends": 0.25}
    if table == Tables.earnings_surprises:
        return {"ticker": "AAPL", "earnings_date": "2026-01-15", "eps_estimate": 1.5, "eps_actual": 2.0, "surprise_pct": None}
    row: dict[str, object] = {
        "ticker": "AAPL",
        "accession_number": "a-1",
        "proposal_seq": 1.0,
        "cik": "0000320193",
        "form": "8-K/A",
        "filing_date": "2026-01-15",
        "period_of_report": "2026-01-12",
        "meeting_date": "2026-01-10",
        "is_amendment": 1.0,
        "mentions_preliminary": 0.0,
        "is_preliminary_stated": None,
        "proposal_number": "4b",
        "proposal_type": "director_election",
        "description": "Election of directors",
        "vote_standard": "majority",
        "votes_for": 80.0,
        "votes_against": 15.0,
        "votes_abstain": None,
        "votes_broker_non_votes": 5.0,
        "nominee_sum_matches": 0.0,
        "nominee_votes_json": '[{"name":"Jane Doe","votes_for":80,"votes_against":15,"votes_abstain":0}]',
        "exec_officer_titles": "Chief Executive Officer",
        "n_nominees": 2.0,
        "min_support_pct": 0.8,
        "min_support_name": "Jane Doe",
        "n_nominees_below_70pct": 0.0,
        "source": "8k",
    }
    row.update({f"n_nominees_{bucket}": float(i) for i, bucket in enumerate(("ceo", "exec_officer", "non_employee", "unmatched"))})
    row.update(
        {
            f"votes_{leg}_{bucket}": float(i)
            for i, bucket in enumerate(("ceo", "exec_officer", "non_employee", "unmatched"))
            for leg in ("for", "against", "abstain", "broker_non_votes")
        }
    )
    return row


@pytest.mark.parametrize("table", [Tables.sec_8k_votes, Tables.earnings_surprises, Tables.dividends], ids=lambda table: table.name)
@pytest.mark.parametrize("legacy", [False, True], ids=["complete", "legacy"])
def test_source_projection_preserves_scoped_raw_payload(sqlite_store: DataStore, table: Table, legacy: bool) -> None:
    """The migrated loader retains raw cells, keys and clocks while excluding extra payload."""
    # Seed the real store with in-scope rows, both date bounds, another ticker and an empty marker.
    payload = _payload(table)
    if legacy:
        optional = {"eps_estimate", "surprise_pct", "nominee_votes_json", "votes_for_ceo", "n_nominees_non_employee", "source"}
        payload = {key: value for key, value in payload.items() if key not in optional}
    assert table.date_col is not None
    clock = table.date_col
    rows = [payload, {**payload, clock: "2026-02-15", "ticker": "AAPL"}]
    rows.extend([{**payload, clock: "2026-01-14"}, {**payload, clock: "2026-02-16"}, {**payload, "ticker": "OUTSIDE"}])
    if table == Tables.sec_8k_votes:
        rows = [{**row, "accession_number": f"a-{i}"} for i, row in enumerate(rows)]
    if table.empty_marker is not None:
        rows.append({**payload, "accession_number": "empty-1", table.empty_marker[0]: table.empty_marker[1]})
    df_rows = pd.DataFrame(rows).assign(unused_payload="must not be streamed")
    df_rows[clock] = pd.to_datetime(df_rows[clock])
    value = {Tables.sec_8k_votes: "votes_for", Tables.earnings_surprises: "eps_actual", Tables.dividends: "dividends"}[table]
    df_rows.loc[1, value] = None
    sqlite_store.save(table, df_rows)
    window = Window(pd.Timestamp("2026-01-15"), pd.Timestamp("2026-02-15"))
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=fixture_config(),
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "source projection contract", pd.Timestamp("2026-11-16")),
        ),
    )

    # Compare actual streaming to the original ordinary-read payload, with and without marker visibility.
    for markers in (False, True):
        df_actual = load_source(context, table, window=window, markers=markers)
        df_expected = sqlite_store.load(table, where={"ticker": ["AAPL"], clock: window}, markers=markers)
        assert df_actual is not None and df_expected is not None
        assert set(df_actual.columns) == set(payload)
        assert len(df_actual) == 2 + int(markers and table.empty_marker is not None)
        pd.testing.assert_frame_equal(
            df_actual.sort_values(list(table.pk)).reset_index(drop=True).sort_index(axis=1),
            df_expected.drop(columns="unused_payload").sort_values(list(table.pk)).reset_index(drop=True).sort_index(axis=1),
        )
    with pytest.raises(ValueError, match="needs an explicit"):
        next(sqlite_store.iter_load(table))
    print(
        f"SANITY: {table.name} {legacy=}: streamed raw cells/keys/clocks/nulls/markers match scoped load; extra payload excluded; strict guard retained."
    )


@pytest.mark.parametrize(
    ("end_date", "as_of", "read_window", "expected_end"),
    [
        ("2026-01-18", "2026-01-20", None, "2026-01-18"),
        ("2026-01-20", "2026-01-18", None, "2026-01-18"),
        ("2026-01-20", "2026-01-20", Window(pd.Timestamp("2026-01-14"), pd.Timestamp("2026-01-16")), "2026-01-16"),
    ],
    ids=["configured-end", "run-as-of", "requested-window"],
)
def test_dated_snapshot_uses_declared_date(sqlite_store: DataStore, end_date: str, as_of: str, read_window: Window | None, expected_end: str) -> None:
    """A snapshot resume contract without a frontier still retains its declared date scope."""
    # Seed the registered snapshot payload across both sides of the source bounds.
    table = Tables.superinvestor_roster
    config = fixture_config()
    item = config.data.tables[table.name]
    item.start_date, item.source_start, item.end_date = "2026-01-15", None, end_date
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store, config=config, universe=universe_of(["AAPL"]), run=WindowRun("full", "dated snapshot contract", pd.Timestamp(as_of))
        ),
    )
    df_rows = pd.DataFrame(
        {
            "snapshot_date": pd.date_range("2026-01-14", "2026-01-21"),
            "dataroma_code": "BRK",
            "manager_name": "Berkshire Hathaway",
            "cik": "0001067983",
            "resolution": "override",
            "source_url": "https://www.dataroma.com/",
            "unused_payload": "must not be streamed",
        }
    )
    df_rows.loc[2, "manager_name"] = None
    context.store.save(table, df_rows)

    # Check exact retained keys and raw nulls through the real projected stream.
    df_actual = load_source(context, table, window=read_window)
    expected_dates = pd.date_range("2026-01-15", expected_end)
    assert table.resume is not None and table.resume.frontier_col is None and table.date_col == "snapshot_date"
    assert df_actual is not None and set(df_actual.columns) == set(table.read_columns)
    assert pd.DatetimeIndex(df_actual.snapshot_date).sort_values().equals(expected_dates)
    assert df_actual.manager_name.isna().sum() == 1
    print(
        f"SANITY: dated roster snapshots retain their declared date, inclusive hard bounds, as-of cap and requested intersection through {expected_end}."
    )


def test_explicit_source_frontier_precedes_table_date(sqlite_store: DataStore) -> None:
    """A filing frontier keeps late-reported periods inside the source interval."""
    # Seed old economic periods whose filing dates bracket the configured interval.
    table = Tables.sec13f_hr
    config = fixture_config()
    item = config.data.tables[table.name]
    item.start_date, item.source_start, item.end_date = "2026-01-15", None, "2026-01-18"
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("full", "filing frontier contract", pd.Timestamp("2026-01-20")),
        ),
    )
    df_rows = pd.DataFrame(
        {
            "cik": "0001067983",
            "period": pd.Timestamp("2025-12-31"),
            "accession_number": [f"a-{i}" for i in range(4)],
            "ticker": "AAPL",
            "cusip": "037833100",
            "filing_date": pd.to_datetime(["2026-01-14", "2026-01-15", "2026-01-18", "2026-01-19"]),
        }
    )
    context.store.save(table, df_rows)

    # An explicit filing clock wins over the earlier economic period.
    df_actual = load_source(context, table, columns=list(df_rows.columns))
    assert df_actual is not None and set(df_actual.accession_number) == {"a-1", "a-2"}
    assert set(pd.to_datetime(df_actual.period)) == {pd.Timestamp("2025-12-31")}
    print("SANITY: explicit filing_date frontier retains the in-bounds filings even when their economic period precedes the source floor.")


def test_undated_source_requires_parent_keys(sqlite_store: DataStore) -> None:
    """A child without either intrinsic date cannot silently become an unbounded read."""
    # Seed a real registered child whose date eligibility belongs to its parent.
    table = Tables.insider_footnotes
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=fixture_config(),
            universe=universe_of(["AAPL"]),
            run=WindowRun("full", "undated child refusal", pd.Timestamp("2026-01-20")),
        ),
    )
    context.store.save(table, pd.DataFrame({"accession_number": ["a-1"], "footnote_id": ["F1"], "footnote_text": ["Known source text"]}))

    # Parent-key inheritance remains mandatory for a genuinely undated source.
    assert table.date_col is None and table.resume is None
    with pytest.raises(ValueError, match="undated source rows require inherited parent keys"):
        load_source(context, table)
    print("SANITY: genuinely undated source children still require inherited parent keys before streaming.")
