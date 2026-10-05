"""Static contract for the extraction DAG; Airflow is not installed in the test venv."""

from __future__ import annotations

import ast
from pathlib import Path

from src.data_extract import cli as extraction_cli
from tests.dags.dag_harness import ALL_DONE, ALL_SUCCESS, load_dag, replay

DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_extraction.py"
AGG_DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_aggregation.py"
EDGAR_DRIVER_FILE = "src/data_extract/utils/common/edgar_driver.py"
STRICT_EDGAR_FILES = (
    "src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py",
    "src/data_extract/utils/institutionals/fetch_8k_edgar.py",
    "src/data_extract/utils/institutionals/fetch_13d_edgar.py",
    "src/data_extract/utils/institutionals/fetch_13g_edgar.py",
    "src/data_extract/utils/institutionals/fetch_insider_edgar.py",
    "src/data_extract/utils/structure/fetch_def14a_edgar.py",
    "src/data_extract/utils/structure/fetch_filing_text.py",
)
REQUIRED_COMMANDS = {
    "seed-universe",
    "macro",
    "short-interest",
    "earnings-surprises",
    "price-history",
    "fails-to-deliver",
    "thirteen-f",
    "thirteen-f-backfill",
    "financial-statements",
    "insider-transactions",
    "financial-notes",
    "identity-tables",
    "superinvestors",
    "thirteen-f-managers",
    "fundamentals",
    "fundamentals-employees",
    "fundamentals-sharadar",
    "def14a",
    "def14a-edgar",
    "sec-8k-items",
    "sec-8k-votes",
    "sec-13d",
    "sec-13g",
    "filing-text",
    "extract-earnings-calls",
    "extraction-status",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_required_sources_are_scheduled_without_retired_attention_sources():
    source = _source(DAG_FILE)
    tree = ast.parse(source)
    scheduled = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "fetch"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    assert scheduled == REQUIRED_COMMANDS
    assert "google-trends" not in scheduled

    print("\n=== SANITY CHECK: extraction DAG sources ===")
    print(f"  all {len(REQUIRED_COMMANDS) - 1} extraction commands are scheduled")
    print("  OK: the retired Google Trends source is absent and the final gate is present")


def test_freshness_inventory_comes_only_from_schema():
    root = DAG_FILE.parents[2]
    cli_source = _source(root / "src/data_extract/cli.py")
    assert "freshness_tables()" in cli_source
    assert "SOURCE_TABLES" not in cli_source
    assert not (root / "src/data_extract/freshness.py").exists()

    print("\n=== SANITY CHECK: one freshness inventory ===")
    print("  extraction-status consumes schema.freshness_tables(); no parallel registry exists")
    print("  OK: schema.py is the sole table/cadence/date-column source of truth")


def test_retries_dependencies_and_hard_gates_are_wired():
    source = _source(DAG_FILE)
    tree = ast.parse(source)
    identity_consumer_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "identity_consumers" for target in node.targets)
    )
    assert isinstance(identity_consumer_assignment.value, ast.List)
    identity_consumers = {element.id for element in identity_consumer_assignment.value.elts if isinstance(element, ast.Name)}
    expected_identity_consumers = {
        "short_interest",
        "fails_to_deliver",
        "financial_statements",
        "financial_notes",
        "fundamentals",
        "fundamentals_employees",
        "def14a",
        "def14a_edgar",
        "sec_8k_items",
        "sec_13d",
        "sec_13g",
        "filing_text",
    }
    identity_independent = {
        "insider_transactions",
        "thirteen_f",
        "thirteen_f_backfill",
        "superinvestors",
        "thirteen_f_managers",
        "macro",
        "earnings_surprises",
        "price_history",
        "extract_earnings_calls",
    }
    assert '"retries": 3' in source
    assert 'pool_slots=2 if pool == "sec_api" else 1' in source
    assert "thirteen_f >> thirteen_f_backfill" in source  # one EDGAR walk at a time: the backfill follows the nightly walk
    assert 'thirteen_f_backfill = fetch("thirteen-f-backfill", pool="sec_api")' in source
    assert identity_consumers == expected_identity_consumers
    assert identity_consumers.isdisjoint(identity_independent)
    assert "sec_8k_votes" not in identity_consumers
    assert "insider_transactions >> identity_tables >> identity_consumers" in source
    assert "[fundamentals, fundamentals_employees] >> fundamentals_sharadar" in source
    assert "[sec_8k_items, def14a] >> sec_8k_votes" in source
    assert "all_fetchers >> extraction_status >> trigger_aggregation" in source

    print("\n=== SANITY CHECK: extraction retry + gate wiring ===")
    print("  identity producer -> 12 direct consumers; 8-K votes inherit through item/proxy parents")
    print("  fundamentals facts + employees are siblings; merged history waits for both")
    print("  OK: independent manager-CIK and non-SEC sources stay outside the identity barrier")


def test_one_price_task_and_no_alias_commands():
    dag = load_dag(DAG_FILE)
    price_tasks = [task_id for task_id, task in dag.tasks.items() if "price-history" in str(task.kwargs.get("bash_command", ""))]
    assert price_tasks == ["price_history"]
    assert not {"splits", "dividends"} & set(dag.tasks), "the merged download has one task"
    assert not {"splits", "dividends"} & set(extraction_cli.cli.commands), "the CLI aliases are gone"
    assert "price-history" in extraction_cli.cli.commands

    print("\n=== SANITY CHECK: one yfinance task ===")
    print("  price_history is the only task writing prices, prices_dividends and prices_splits; the splits/dividends aliases are removed")


def test_gate_and_trigger_run_all_done_and_a_failed_fetcher_never_blocks_aggregation():
    dag = load_dag(DAG_FILE)
    gate, trigger = dag.tasks["extraction_status"], dag.tasks["trigger_data_aggregation"]
    assert gate.trigger_rule == ALL_DONE and trigger.trigger_rule == ALL_DONE
    assert trigger.upstream == {"extraction_status"}
    fetchers = {task_id for task_id in dag.tasks if task_id not in {"seed_universe", "extraction_status", "trigger_data_aggregation"}}
    assert gate.upstream == fetchers, "the report waits for every fetcher"
    assert {"thirteen_f"} <= dag.tasks["thirteen_f_backfill"].upstream, "one EDGAR walk at a time: the backfill follows the nightly 13F walk"
    assert all(dag.tasks[task_id].trigger_rule == ALL_SUCCESS for task_id in fetchers), "a fetcher still waits for its own sources"

    for failed in (["price_history"], ["identity_tables"], ["thirteen_f", "insider_transactions", "extraction_status"]):
        states = replay(dag, failed)
        assert states["trigger_data_aggregation"] == "success", (failed, states)
    states = replay(dag, ["identity_tables"])
    assert states["fundamentals"] == "upstream_failed" and states["extraction_status"] == "success"

    print("\n=== SANITY CHECK: non-blocking extraction DAG (AC-015) ===")
    print(f"  {len(fetchers)} fetchers -> extraction_status (ALL_DONE) -> trigger_data_aggregation (ALL_DONE)")
    print("  replay: price_history failed / identity_tables failed (its 12 consumers upstream_failed) / 13F + insider + the report failed")
    print("  -> aggregation is triggered every time. Validated.")


def _raises(node: ast.AST, name: str) -> bool:
    return any(isinstance(sub, ast.Raise) and isinstance(sub.exc, ast.Call) and getattr(sub.exc.func, "id", None) == name for sub in ast.walk(node))


def _calls(node: ast.AST) -> list[str]:
    return [sub.func.id for sub in ast.walk(node) if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)]


def test_scheduled_edgar_walks_save_what_they_read_and_exit_zero():
    root = DAG_FILE.parents[2]
    driver = ast.parse(_source(root / EDGAR_DRIVER_FILE))
    run = next(f for f in ast.walk(driver) if isinstance(f, ast.FunctionDef) and f.name == "run_edgar_fetch")
    assert not _raises(run, "IncompleteEdgarRunError"), "a failed document must not fail the task"
    calls = _calls(run)
    assert "record_run" not in calls and "_record_tables" not in calls, "the driver must not write the run manifest"
    assert calls.index("_run_pass") < calls.index("_retry_rounds") < calls.index("_log_coverage")

    missing = [
        path
        for path in STRICT_EDGAR_FILES
        if not any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "EdgarFetch"
            for call in ast.walk(ast.parse(_source(root / path)))
        )
    ]
    assert not missing, f"scheduled EDGAR walks no longer declare an EdgarFetch spec: {missing}"

    print("\n=== SANITY CHECK: EDGAR walks save what they read ===")
    print(f"  all {len(STRICT_EDGAR_FILES)} scheduled EDGAR fetchers declare an EdgarFetch; run_edgar_fetch reads, retries in")
    print("  rounds and logs coverage, never raises IncompleteEdgarRunError and writes no run manifest.")


def test_earnings_calls_are_one_task_outside_the_retired_scrape_pool():
    source = _source(DAG_FILE)
    compose = _source(DAG_FILE.parents[2] / "docker-compose.yml")
    assert 'extract_earnings_calls = fetch("extract-earnings-calls")' in source
    assert "download-earnings-calls" not in source and "ingest-earnings-calls" not in source
    assert '"scrape"' not in source
    assert "pools set scrape" not in compose

    print("\n=== SANITY CHECK: earnings-call extraction task ===")
    print("  one extract-earnings-calls task in the default pool replaces download + ingest")
    print("  OK: no task or airflow-init pool definition references the scrape pool")
