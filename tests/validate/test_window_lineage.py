"""Known-truth controls for the permanent source-to-assembly lineage audit."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.context import Context
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.institutionals.short_flow_features import _publish_ftd_zip_states
from src.data_store.schema import Tables
from src.utils.publication import point_in_time
from src.utils.window import Window, WindowRun
from src.validate.checks.window_lineage import (
    _boundaries,
    _combine_routes,
    _combine_table,
    _expected_state,
    _State,
    check_window_lineage,
    expected_publication,
)
from src.validate.checks.window_plan import _check_part, _subtract_days, check_window_plans
from src.validate.cli import cli
from src.validate.result import CheckResult, Finding, jsonable
from tests.support.config import fixture_config
from tests.support.universe import universe_of
from tests.support.window_lineage import (
    _assemble_fields,
    _finish_group,
    _ftd_metadata_periods,
    _ftd_selected_rows,
    _Journal,
    _packet_rows,
    _probe_rows,
    _short_metadata_window,
    _stage_family,
    measure_window_lineage,
)
from tests.support.window_samples import WindowSamples, assert_lineage_results


def test_lineage_command_is_permanent() -> None:
    assert "window-lineage" in cli.commands, "Persisted-cell hashes do not measure producer or assembled-output lineage"
    print("SANITY: selected-source lineage has a permanent read-only validation entry point.")


@pytest.mark.parametrize(
    ("lag", "expected"),
    [(0, "2026-07-02"), (1, "2026-07-06"), (2, "2026-07-06"), (3, "2026-07-06"), (12, "2026-07-14"), (21, "2026-07-23")],
)
def test_source_column_calendar_policy_oracle(lag: int, expected: str) -> None:
    sessions = pd.bdate_range("2026-07-01", "2026-08-05").difference(pd.to_datetime(["2026-07-03"]))
    policy = {"column": "as_of", "lag_days": lag}
    actual = expected_publication(policy, {"as_of": "2026-07-02", "accepted_at": "2026-08-01"}, sessions)
    assert actual == pd.Timestamp(expected)
    assert pd.isna(expected_publication(policy, {"as_of": "invalid"}, sessions))
    assert pd.isna(expected_publication(policy, {"as_of": "2026-09-01"}, sessions))
    print(
        f"SANITY: independent source-column oracle applies exactly {lag} calendar days and forward market-session roll; invalid/future stays unavailable."
    )


@pytest.mark.parametrize(
    ("period", "lag", "expected"),
    [("2026_03", 12, "2026-04-13"), ("2026q1", 12, "2026-04-13"), ("202607a", 21, "2026-08-05"), ("202607b", 21, "2026-08-21")],
)
def test_archive_source_period_oracle(period: str, lag: int, expected: str) -> None:
    sessions = pd.bdate_range("2026-01-01", "2026-09-01").difference(pd.to_datetime(["2026-04-03", "2026-07-03"]))
    policy = {"column": "period", "lag_days": lag}
    assert expected_publication(policy, {"period": period, "filed": "2026-09-01", "available_at": "2026-09-01"}, sessions) == pd.Timestamp(expected)
    assert pd.isna(expected_publication(policy, {"period": "2026_13"}, sessions))
    assert pd.isna(expected_publication(policy, {"period": None}, sessions))
    print(f"SANITY: independent {period}+{lag}CD oracle uses the archive end, preserving other raw clocks without an eligibility override.")


def _known_state() -> tuple[Context, _State, pd.DatetimeIndex]:
    config = fixture_config()
    sessions = pd.DatetimeIndex(["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08"])
    context = cast(
        Context,
        SimpleNamespace(
            config=config, universe=SimpleNamespace(tickers=["AAA"]), run=WindowRun("incremental", "known truth lineage clock", sessions[-1])
        ),
    )
    raw = pd.DataFrame({"ticker": ["AAA"], "as_of": pd.to_datetime(["2026-07-02"]), "board_size": [11.0], "accession_number": ["source-1"]})
    rows = point_in_time(raw, config, Tables.def14a_llm, sessions)
    state = _State(Tables.def14a_llm, Tables.cube_part_governance, rows, "board_size", fundamentals_to_daily(rows, "board_size", sessions))
    return context, state, sessions


def test_real_calendar_boundary_and_raw_ancestry() -> None:
    context, state, sessions = _known_state()
    records, findings = _boundaries(context.config, context.universe.tickers, state, sessions)
    assert not findings
    assert state.rows.as_of.iloc[0] == pd.Timestamp("2026-07-06")
    assert records[0]["value"] != records[0]["value"]
    assert records[1]["source"]["accession_number"] == "source-1"
    assert records[1]["source"]["as_of"] == pd.Timestamp("2026-07-02")
    assert records[1]["usable_on"] == pd.Timestamp("2026-07-06")
    print(
        "SANITY: a Thursday filing waits across the July3 market holiday; the assembled-boundary ancestry retains its actual accession and filing day."
    )


def test_early_production_selection_is_a_finding() -> None:
    context, state, sessions = _known_state()
    state.rows["as_of"] = pd.Timestamp("2026-07-02")
    state.daily = fundamentals_to_daily(state.rows, state.field, sessions)
    _, findings = _boundaries(context.config, context.universe.tickers, state, sessions)
    assert findings and any("differs" in finding.observed for finding in findings)
    print("SANITY: an early production selector produces a high-severity finding even when its stored scalar value is unchanged.")


def test_other_raw_clocks_do_not_override_the_configured_source_column() -> None:
    context, _, sessions = _known_state()
    row = pd.Series({"filing_date": "2026-07-01", "accepted_at": "2026-08-01T01:00:00Z"})
    policy = dict(OmegaConf.to_container(context.config.data.tables[Tables.sec13f_hr.name].pit, resolve=True))
    expected = expected_publication(policy, row, sessions)
    assert expected == pd.Timestamp("2026-07-06")
    print("SANITY: filing_date plus two calendar days controls eligibility; preserved accepted_at evidence cannot silently override that column.")


def test_selected_state_reaches_actual_assembly_and_bounded_writer(sqlite_store: object, monkeypatch: pytest.MonkeyPatch) -> None:
    context, state, sessions = _known_state()
    context.store = sqlite_store
    context.config_dir = Path("configs")
    context.paths = {}
    context.universe = universe_of(["AAA"])
    context.run = WindowRun("incremental", "known truth assembly scope", sessions[-1])
    context.config.validate.window_samples.keys_per_table = 1
    sqlite_store.replace(Tables.cube_part_prices, pd.DataFrame({"date": sessions, "ticker": "AAA", "close_split": 100.0}))
    monkeypatch.setattr("src.data_aggregate.transformers.step_assemble_cube.load_peers_or_raise", lambda *_: {"AAA": {"BBB": 1.0}})
    records, findings = _boundaries(context.config, context.universe.tickers, state, sessions)
    result = _assemble_fields(context, [state], {state.field: records})[state.field]
    assert not findings and result["outside_write_rows_preserved"] == 2
    assert len(result["assembled"]) == 3
    replay = {
        "sessions": jsonable(list(sessions)),
        "routes": [
            {
                "table": state.table.name,
                "part": state.part.name,
                "tickers": ["AAA"],
                "requested_tickers": ["AAA"],
                "field": state.field,
                "clock": state.clock,
                "age_days": None,
                "lifetime_sessions": None,
                "exception": None,
                "rows": jsonable(state.rows.to_dict("records")),
                "daily": jsonable(records),
                "assembly": jsonable(result),
            }
        ],
    }
    manifest = {"tables": {state.table.name: {"sample_scope": {"groups": ["AAA"]}}}}
    checked = check_window_lineage(context, replay, manifest)
    assert checked[state.table.name].status == "pass"
    replay["routes"][0]["assembly"]["assembled"][1][result["column"]] = 99.0
    assert check_window_lineage(context, replay, manifest)[state.table.name].status == "fail"
    # A genuine independent part payload establishes wide-part presence; its
    # selected all-null neighbor remains unknown through the real projection.
    sqlite_store.replace(state.part, pd.DataFrame({"date": sessions, "ticker": "AAA", "f_actual_other_state": 7.0}))
    state.daily.loc[:, :] = float("nan")
    empty = _assemble_fields(context, [state], {state.field: records})[state.field]
    assert all(row[empty["column"]] is None for row in empty["assembled"])
    assert empty["contextual_presence"]["columns"] == ["_lineage_context_f_actual_other_state"]
    assert not any("_lineage_context" in key for row in empty["assembled"] for key in row)
    state.daily.loc[:, :] = 11.0
    companion = _State(state.table, state.part, state.rows, "known_empty_companion", state.daily * float("nan"))
    wide = _assemble_fields(context, [state, companion], {state.field: records, companion.field: records})
    assert len(wide[state.field]["projected_family_columns"]) == 2
    assert all(row[wide[state.field]["column"]] == 11 for row in wide[state.field]["assembled"])
    assert all(row[wide[companion.field]["column"]] is None for row in wide[companion.field]["assembled"])
    print(
        "SANITY: actual part projection and assembly retain boundary values/nulls, and the real finite writer preserves both neighboring rows outside its authorized interval."
    )


def test_targetpart_batch_keeps_source_identities(sqlite_store: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context, template, sessions = _known_state()
    context.store, context.config_dir, context.paths = sqlite_store, Path("configs"), {}
    context.universe = universe_of(["AAA"])
    context.run = WindowRun("incremental", "known truth grouped assembly", sessions[-1])
    sqlite_store.replace(Tables.cube_part_prices, pd.DataFrame({"date": sessions, "ticker": "AAA", "close_split": 100.0}))
    monkeypatch.setattr("src.data_aggregate.transformers.step_assemble_cube.load_peers_or_raise", lambda *_: {"AAA": {"BBB": 1.0}})
    raw = template.rows.rename(columns={"board_size": "selected_filing"}).assign(
        selected_filing=1.0, rp_seq=0, filing_date=pd.Timestamp("2026-07-02")
    )
    daily = fundamentals_to_daily(raw, "selected_filing", sessions)
    states = [
        _State(table, Tables.cube_part_institutionals, raw.assign(accession_number=table.name), "selected_filing", daily)
        for table in (Tables.sec_13d, Tables.sec_13g)
    ]
    journal = _Journal({"routes": []}, tmp_path / "grouped.jsonl")
    pending = _stage_family(context, states, sessions, journal)
    assert all("pending" in entry["reason"] for entry in journal.report["routes"])
    _finish_group(context, pending, journal)
    entries = journal.report["routes"]
    assert all("reason" not in entry and "failure" not in entry for entry in entries)
    assert len({entry["assembly"]["column"] for entry in entries}) == 2
    assert all(len(entry["assembly"]["projected_family_columns"]) == 2 and entry["requested_tickers"] == ["AAA"] for entry in entries)
    assert [entry["rows"][0]["accession_number"] for entry in entries] == [table.name for table in (Tables.sec_13d, Tables.sec_13g)]
    print(
        "SANITY: one actual targetpart assembly namespaces equal-named states from two sources while retaining each original accession/requested group; pending evidence is not PASS."
    )


def test_pinned_source_lineage(window_samples: WindowSamples) -> None:
    results = measure_window_lineage(window_samples)
    assert_lineage_results(results, window_samples.planning)
    measured = [name for name, result in results.items() if result.status == "pass"]
    abstained = [name for name, result in results.items() if result.status == "abstain"]
    print(f"SANITY: actual source-to-state-to-assembly measured for {measured}; unmeasured tables remain explicit ABSTAIN: {abstained}.")


def test_required_routes_cannot_mask_partial_coverage_or_downgrade_failure() -> None:
    wanted = {"AAA", "BBB"}
    complete = CheckResult("window_lineage", "broad", "pass", scope={"tickers": sorted(wanted)})
    partial = CheckResult("window_lineage", "elite", "pass", scope={"tickers": ["AAA"]})
    broad = _combine_routes("broad", [complete], wanted, 2)
    elite = _combine_routes("elite", [partial], wanted, 2)
    assert broad.status == "pass" and elite.status == "abstain"
    assert _combine_table(Tables.sec13f_hr.name, [broad, elite]).status == "abstain"
    failed = CheckResult.measured("window_lineage", "broad", [Finding.at(8, "early accession", "eligible accession")], scope={}, metrics={})
    reason = CheckResult.abstained("window_lineage", "broad", "later field unavailable")
    assert _combine_routes("broad", [failed, complete, reason], wanted, 2).status == "fail"
    assert _combine_routes("broad", [complete, reason], wanted, 2).status == "abstain"
    print("SANITY: full broad coverage cannot hide partial elite coverage, and later missing evidence cannot downgrade FAIL.")


def test_insider_visibility_survives_json_nat_and_exclusive_expiry() -> None:
    _, state, sessions = _known_state()
    state.table, state.field = Tables.insider_transactions, "shares_n"
    ready = pd.Timestamp("2026-07-06")
    row = pd.Series({"anchor": ready, "visible_until": "NaT", "shares_n": 7000.0})
    observed = [(ready, row, {"accession_number": "actual-selected"})]
    assert _expected_state(state, observed, pd.Timestamp("2026-07-07"), sessions)[0] == 7000
    assert _expected_state(state, observed, pd.Timestamp("2026-07-26"), sessions)[0] == 0
    row["visible_until"] = "2026-07-07"
    assert _expected_state(state, observed, pd.Timestamp("2026-07-07"), sessions)[0] == 0
    print("SANITY: JSON NaT retains an unbounded insider interval; both correction and20-calendar-day window ends remain exclusive.")


def test_short_metadata_retains_prior_year_publication_boundary() -> None:
    context, _, _ = _known_state()
    sessions = pd.DatetimeIndex(["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"])
    retained = Window(pd.Timestamp("2009-01-01"), sessions[-1])
    actual = _short_metadata_window(context, sessions, retained)
    assert actual.start == pd.Timestamp("2025-12-31")
    assert actual.end == retained.end
    print(
        "SANITY: genuine global short-volume metadata retains the prior-year raw session publishing into2026; scoped economic history remains separate and complete."
    )


def test_ftd_null_publication_barrier_is_source_evidence() -> None:
    context, _, _ = _known_state()
    sessions = pd.bdate_range("2025-11-03", "2026-01-09").difference(pd.to_datetime(["2025-11-27", "2025-12-25", "2026-01-01"]))
    raw = pd.DataFrame(
        {
            "ticker": "AAA",
            "period": ["202511b", "202512a"],
            "date": pd.to_datetime(["2025-11-28", "2025-12-15"]),
            "fails_quantity": [10.0, float("nan")],
        }
    )
    settlement = raw.pivot(index="date", columns="ticker", values="fails_quantity").reindex(sessions)
    rows = point_in_time(raw, context.config, Tables.sec_fails_to_deliver_security, sessions)
    daily = _publish_ftd_zip_states(settlement, rows, sessions, rows.groupby("period")["as_of"].first().to_dict())
    state = _State(Tables.sec_fails_to_deliver_security, Tables.cube_part_institutionals, rows, "fails_quantity", daily)
    records, findings = _boundaries(context.config, context.universe.tickers, state, sessions)
    assert not findings and len(records) == 3 and len(_probe_rows(context, state, sessions)) == 3
    assert records[0]["value"] == 10.0 and pd.isna(records[1]["value"]) and pd.isna(records[2]["value"])
    assert records[1]["source"]["period"] == "202512a"
    print(
        "SANITY: an actual published NULL FTD vector supersedes the earlier positive ZIP; NULL is measured evidence, never a search for an older non-null state."
    )


def test_ftd_metadata_retains_actual_null_seed(sqlite_store: object) -> None:
    context, _, _ = _known_state()
    context.store = sqlite_store
    sessions = pd.bdate_range("2025-11-03", "2026-01-09").difference(pd.to_datetime(["2025-11-27", "2025-12-25", "2026-01-01"]))
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-10-31", "2025-11-14", "2025-11-28", "2025-12-15", "2025-12-31"]),
            "cusip": "111111111",
            "period": ["202510b", "202511a", "202511b", "202512a", "202512b"],
            "fails_quantity": [5.0, 10.0, float("nan"), 20.0, 30.0],
        }
    )
    sqlite_store.replace(Tables.sec_fails_to_deliver_security, source)
    periods = _ftd_metadata_periods(context, sessions, Window(pd.Timestamp("2009-07-01"), sessions[-1]))
    assert periods == {"202511b", "202512a"}
    assert sqlite_store.load(Tables.sec_fails_to_deliver_security).shape[0] == len(source)
    print(
        "SANITY: actual persisted ZIP metadata keeps the latest prior NULL seed and publishing2026 period, without choosing an older positive ZIP or trimming economic rows."
    )


def test_ftd_capture_uses_latest_global_served_row_even_when_absent() -> None:
    sessions = pd.bdate_range("2025-12-01", "2026-01-09").difference(pd.to_datetime(["2025-12-25", "2026-01-01"]))
    series = pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2025-12-01"]), "period": ["202512a"], "fails_quantity": [10.0]})
    coverage = pd.DataFrame({"date": pd.to_datetime(["2025-12-01", "2025-12-15"]), "period": "202512a"})
    rows = _ftd_selected_rows(series, coverage, sessions, ["AAA"])
    assert len(rows) == 1 and rows.date.iloc[0] == pd.Timestamp("2025-12-15") and pd.isna(rows.fails_quantity.iloc[0])
    print("SANITY: FTD ancestry points to the actual latest global served session's NULL vector cell, never the last surviving positive ticker row.")


def test_explicit_zero_calendar_lag_is_not_an_implicit_default() -> None:
    context, _, sessions = _known_state()
    policy = dict(OmegaConf.to_container(context.config.data.tables[Tables.def14a_llm.name].pit, resolve=True))
    policy["lag_days"] = 0
    assert expected_publication(policy, {"as_of": "2026-07-02"}, sessions) == pd.Timestamp("2026-07-02")
    del policy["lag_days"]
    with pytest.raises(KeyError):
        expected_publication(policy, {"as_of": "2026-07-02"}, sessions)
    print("SANITY: an explicit zero lag is honored; an omitted configured lag remains a contract error.")


def test_actual_wrong_newer_packet_id_fails_even_with_same_known_flag() -> None:
    context, _, _ = _known_state()
    sessions = pd.DatetimeIndex(["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"])
    pins = [{"ticker": "AAA", "cik": "manager", "period": "2025-12-31"}]
    manifests = pd.DataFrame(
        {
            "cik": "manager",
            "period": pd.Timestamp("2025-12-31"),
            "accession_number": ["old", "new"],
            "filing_date": pd.to_datetime(["2026-07-01", "2026-07-07"]),
            "accepted_at": [None, None],
            "public_at": pd.to_datetime(["2026-07-06", "2026-07-09"]),
        }
    )
    states = pd.DataFrame(
        {
            "cik": ["manager"],
            "period": pd.to_datetime(["2025-12-31"]),
            "_day": pd.to_datetime(["2026-07-06"]),
            "book_known": [True],
            "_selected_accessions": [["new"]],
        }
    )
    rows = _packet_rows(states, manifests, pins, "packet_known_broad")
    assert rows.accession_number.tolist() == ["new"], "Capture must observe the actual wrong selection, not reconstruct the eligible old book"
    daily = fundamentals_to_daily(rows, "packet_known_broad", sessions)
    state = _State(Tables.sec13f_hr, Tables.cube_part_sec_thirteen, rows, "packet_known_broad", daily)
    _, findings = _boundaries(context.config, context.universe.tickers, state, sessions)
    assert findings and daily.loc["2026-07-06", "AAA"] == 1.0
    print("SANITY: actual newer accession chosen early fails even though both old/new packet-known values equal1.")


def test_independent_rewrite_anchored_calendar_input_oracle() -> None:
    context, _, sessions = _known_state()
    assert _subtract_days(pd.Timestamp("2026-07-06"), 2) == pd.Timestamp("2026-07-04")
    name = Tables.cube_part_text.name
    item = context.config.data.tables[name]
    item.overlap.days = 2
    item.warmup[Tables.cube_part_prices.name] = 0
    plan = {
        "requested": {"start": "2026-07-01", "end": "2026-07-08"},
        "read": {"start": "2026-07-04", "end": "2026-07-08"},
        "write": {"start": "2026-07-04", "end": "2026-07-08"},
        "frontier": "2026-07-06",
        "mode": "incremental",
        "keys": {"ticker": ["AAA"]},
    }
    assert _check_part(context, name, plan, {"as_of": "2026-07-08", "tickers": ["AAA"]}, sessions).status == "pass"
    item.warmup[Tables.cube_part_prices.name] = 2
    plan["read"]["start"] = "2026-07-02"
    assert _check_part(context, name, plan, {"as_of": "2026-07-08", "tickers": ["AAA"]}, sessions).status == "pass"
    plan["read"]["start"] = "2026-07-06"
    assert _check_part(context, name, plan, {"as_of": "2026-07-08", "tickers": ["AAA"]}, sessions).status == "fail"
    item.warmup[Tables.cube_part_prices.name] = None
    plan["read"]["start"] = "2026-07-01"
    assert _check_part(context, name, plan, {"as_of": "2026-07-08", "tickers": ["AAA"]}, sessions).status == "pass"
    plan["write"]["start"] = "2026-07-02"
    assert _check_part(context, name, plan, {"as_of": "2026-07-08", "tickers": ["AAA"]}, sessions).status == "fail"
    print(
        "SANITY: independent calendar oracle distinguishes null/full, zero and rewrite-minus-days; a current-source anchor or changed write footprint fails."
    )


def test_independent_assembly_oracle_uses_observed_price_dates() -> None:
    context, _, sessions = _known_state()
    prices, text = Tables.cube_part_prices.name, Tables.cube_part_text.name
    context.config.data_aggregate.assembly.compulsory_parts = [prices, text]
    context.config.data_aggregate.assembly.optional_parts = []
    context.config.data.tables.cube.warmup = {prices: 0, text: 0}
    parts = {}
    for name in (prices, text):
        item = context.config.data.tables[name]
        item.overlap.days = 2
        parts[name] = {
            "requested": {"start": "2026-07-01", "end": "2026-07-08"},
            "write": {"start": "2026-07-04", "end": "2026-07-08"},
            "read": {"start": "2026-07-01" if name == prices else "2025-04-10", "end": "2026-07-08"},
            "frontier": "2026-07-06",
            "mode": "incremental",
            "keys": {"ticker": ["AAA"]},
        }
    assembly = {
        "requested": {"start": "2026-07-01", "end": "2026-07-08"},
        "write": {"start": "2026-07-02", "end": "2026-07-08"},
        "read": {"start": "2026-07-02", "end": "2026-07-08"},
        "frontier": "2026-07-02",
        "mode": "incremental",
        "keys": {"ticker": ["AAA"]},
    }
    replay = {
        "sessions": sessions,
        "planning": {"parts": parts, "assembly": assembly, "selected_components": [prices, text], "as_of": "2026-07-08", "tickers": ["AAA"]},
    }
    assert all(result.status == "pass" for result in check_window_plans(context, replay).values())
    assembly["requested"]["start"] = "2026-07-02"
    assert check_window_plans(context, replay)[Tables.cube.name].status == "fail"
    print(
        "SANITY: assembly derives its first possible output from observed prices, inherits selected writes and catches up its older frontier; a fabricated cube floor fails."
    )
