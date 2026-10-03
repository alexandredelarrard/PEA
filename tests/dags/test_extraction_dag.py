"""Static contract for the extraction DAG; Airflow is not installed in the test venv."""

from __future__ import annotations

import ast
from pathlib import Path

DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_extraction.py"
AGG_DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_aggregation.py"
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
    "splits",
    "price-history",
    "dividends",
    "fails-to-deliver",
    "thirteen-f",
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
        "superinvestors",
        "thirteen_f_managers",
        "macro",
        "earnings_surprises",
        "splits",
        "price_history",
        "dividends",
        "extract_earnings_calls",
    }
    assert '"retries": 3' in source
    assert 'pool_slots=2 if pool == "sec_api" else 1' in source
    assert "splits >> price_history" in source
    assert identity_consumers == expected_identity_consumers
    assert identity_consumers.isdisjoint(identity_independent)
    assert "sec_8k_votes" not in identity_consumers
    assert "insider_transactions >> identity_tables >> identity_consumers" in source
    assert "[fundamentals, fundamentals_employees] >> fundamentals_sharadar" in source
    assert "[sec_8k_items, def14a] >> sec_8k_votes" in source
    assert "all_fetchers >> extraction_status >> trigger_aggregation" in source
    assert "trigger_rule=TriggerRule.ALL_SUCCESS" in source
    assert "trigger_rule=TriggerRule.ALL_SUCCESS" in _source(AGG_DAG_FILE)

    print("\n=== SANITY CHECK: extraction retry + gate wiring ===")
    print("  identity producer -> 12 direct consumers; 8-K votes inherit through item/proxy parents")
    print("  fundamentals facts + employees are siblings; merged history waits for both")
    print("  OK: independent manager-CIK and non-SEC sources stay outside the identity barrier")


def test_scheduled_edgar_walks_require_complete_ticker_coverage():
    root = DAG_FILE.parents[2]
    missing = [path for path in STRICT_EDGAR_FILES if "require_complete=True" not in _source(root / path)]
    assert not missing, f"scheduled EDGAR walks still allow partial success: {missing}"

    print("\n=== SANITY CHECK: strict EDGAR walks ===")
    print(f"  all {len(STRICT_EDGAR_FILES)} scheduled per-ticker EDGAR fetchers require complete coverage")
    print("  OK: one failed ticker makes the source task retry without advancing its manifest")


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
