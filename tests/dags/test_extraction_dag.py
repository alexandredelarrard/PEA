"""Static contract for the extraction DAG; Airflow is not installed in the test venv."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
    "splits",
    "price-history",
    "dividends",
    "fails-to-deliver",
    "thirteen-f",
    "financial-statements",
    "insider-transactions",
    "insider-download",
    "notes-download",
    "financial-notes",
    "identity-tables",
    "identity-propagate",
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


#: identity producers, in stage order: the two cache downloads, the build, the propagation.
IDENTITY_DOWNLOADS = {"insider_download", "notes_download"}
#: every task that reads the lineage; each must start after `identity_propagate`.
IDENTITY_CONSUMERS = {
    "insider_transactions",
    "financial_notes",
    "financial_statements",
    "fails_to_deliver",
    "short_interest",
    "fundamentals",
    "fundamentals_employees",
    "def14a",
    "def14a_edgar",
    "sec_8k_items",
    "sec_13d",
    "sec_13g",
    "filing_text",
}
#: derived from consumer tables; each must start after its parents.
DERIVED_PARENTS = {
    "sec_8k_votes": {"sec_8k_items", "def14a"},
    "fundamentals_sharadar": {"fundamentals", "fundamentals_employees"},
}
IDENTITY_INDEPENDENT = {
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


class _StubTask:
    """Records `>>` edges the way Airflow does for a task or a list of tasks."""

    def __init__(self, graph: dict[str, set[str]], params: dict[str, dict], **kwargs: object) -> None:
        self.task_id = str(kwargs["task_id"])
        self._graph = graph
        graph.setdefault(self.task_id, set())
        params[self.task_id] = dict(kwargs)

    def _link(self, upstream: object, downstream: object) -> None:
        for up in upstream if isinstance(upstream, list) else [upstream]:
            for down in downstream if isinstance(downstream, list) else [downstream]:
                self._graph[up.task_id].add(down.task_id)

    def __rshift__(self, other: object) -> object:
        self._link(self, other)
        return other

    def __rrshift__(self, other: object) -> object:
        self._link(other, self)
        return self


def _load_dag_graph(monkeypatch) -> tuple[dict[str, set[str]], dict[str, dict]]:
    """Execute the DAG module against a stub `airflow` package; returns (edges, task kwargs)."""
    graph: dict[str, set[str]] = {}
    params: dict[str, dict] = {}

    def operator(**kwargs: object) -> _StubTask:
        return _StubTask(graph, params, **kwargs)

    airflow = types.ModuleType("airflow")
    airflow.DAG = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    bash = types.ModuleType("airflow.operators.bash")
    bash.BashOperator = operator  # type: ignore[attr-defined]
    trigger = types.ModuleType("airflow.operators.trigger_dagrun")
    trigger.TriggerDagRunOperator = operator  # type: ignore[attr-defined]
    rule = types.ModuleType("airflow.utils.trigger_rule")
    rule.TriggerRule = SimpleNamespace(ALL_SUCCESS="all_success")  # type: ignore[attr-defined]
    modules = {
        "airflow": airflow,
        "airflow.operators": types.ModuleType("airflow.operators"),
        "airflow.operators.bash": bash,
        "airflow.operators.trigger_dagrun": trigger,
        "airflow.utils": types.ModuleType("airflow.utils"),
        "airflow.utils.trigger_rule": rule,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_extraction_dag_under_test", DAG_FILE)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return graph, params


def _descendants(graph: dict[str, set[str]], start: str) -> set[str]:
    seen: set[str] = set()
    stack = list(graph[start])
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(graph[node])
    return seen


def _topological_order(graph: dict[str, set[str]]) -> list[str]:
    indegree = dict.fromkeys(graph, 0)
    for downs in graph.values():
        for down in downs:
            indegree[down] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for down in sorted(graph[node]):
            indegree[down] -= 1
            if indegree[down] == 0:
                ready.append(down)
    return order


def test_retries_dependencies_and_hard_gates_are_wired():
    source = _source(DAG_FILE)
    assert '"retries": 3' in source
    assert 'pool_slots=2 if pool == "sec_api" else 1' in source
    assert "splits >> price_history" in source
    assert "trigger_rule=TriggerRule.ALL_SUCCESS" in source
    assert "trigger_rule=TriggerRule.ALL_SUCCESS" in _source(AGG_DAG_FILE)

    print("\n=== SANITY CHECK: extraction retry + gate wiring ===")
    print("  three retries per task, sec_api tasks take both pool slots, both DAG triggers wait for ALL_SUCCESS")
    print("  OK: a failed source never lets aggregation start")


def test_identity_stage_orders_downloads_build_propagation_consumers_and_status(monkeypatch):
    graph, params = _load_dag_graph(monkeypatch)
    order = _topological_order(graph)
    assert len(order) == len(graph), "the extraction DAG has a cycle"
    position = {task: index for index, task in enumerate(order)}
    after_build = _descendants(graph, "identity_tables")
    after_propagation = _descendants(graph, "identity_propagate")

    # downloads -> identity build -> propagation -> every consumer
    for download in IDENTITY_DOWNLOADS:
        assert "identity_tables" in graph[download], f"{download} must feed identity_tables"
    assert graph["identity_tables"] == {"identity_propagate", "identity_check"}
    assert IDENTITY_CONSUMERS <= graph["identity_propagate"], sorted(IDENTITY_CONSUMERS - graph["identity_propagate"])
    for derived, parents in DERIVED_PARENTS.items():
        assert all(derived in graph[parent] for parent in parents), f"{derived} must wait for {sorted(parents)}"
        assert derived in after_propagation
    # sources that read no lineage keep running beside the identity stage
    assert IDENTITY_INDEPENDENT.isdisjoint(after_build), sorted(IDENTITY_INDEPENDENT & after_build)
    assert not (IDENTITY_DOWNLOADS | {"identity_tables"}) & after_propagation

    # the hard gate: every task upstream, default ALL_SUCCESS, so an identity or propagation failure blocks it
    upstream_of_status = {task for task in graph if "extraction_status" in _descendants(graph, task)}
    assert upstream_of_status == set(graph) - {"extraction_status", "trigger_data_aggregation"}
    assert "trigger_rule" not in params["extraction_status"]
    assert params["trigger_data_aggregation"]["trigger_rule"] == "all_success"
    assert graph["extraction_status"] == {"trigger_data_aggregation"}
    # AC-039 foreign-row half: the identity validator runs after every fetcher and before the freshness gate
    fetchers = set(graph) - {"identity_check", "extraction_status", "trigger_data_aggregation", "seed_universe"}
    assert all("identity_check" in graph[task] for task in fetchers), sorted(t for t in fetchers if "identity_check" not in graph[t])
    assert graph["identity_check"] == {"extraction_status"} and "trigger_rule" not in params["identity_check"]
    assert " -m src validate identity -o " in str(params["identity_check"]["bash_command"])
    for task in IDENTITY_DOWNLOADS:
        assert params[task]["pool"] == "sec_bulk"

    print("\n=== SANITY CHECK: identity stage order (AC-011, AC-037-039) ===")
    print(f"  {len(graph)} tasks, acyclic; downloads {sorted(IDENTITY_DOWNLOADS)} -> identity_tables -> identity_propagate")
    print(f"  -> {len(IDENTITY_CONSUMERS)} consumers -> sec_8k_votes / fundamentals_sharadar -> identity_check -> extraction_status")
    first = min(position[task] for task in IDENTITY_DOWNLOADS)
    print(f"  first identity task at topological position {first}, gate at {position['extraction_status']}")
    print(f"  OK: {len(IDENTITY_INDEPENDENT)} non-identity sources run beside the stage; a failed build or propagation leaves the gate unrun")


def _raises_incomplete(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Raise) and isinstance(sub.exc, ast.Call) and getattr(sub.exc.func, "id", None) == "IncompleteEdgarRunError"
        for sub in ast.walk(node)
    )


def test_scheduled_edgar_walks_require_complete_ticker_coverage():
    root = DAG_FILE.parents[2]
    driver = ast.parse(_source(root / EDGAR_DRIVER_FILE))
    fields = {
        node.target.id
        for cls in ast.walk(driver)
        if isinstance(cls, ast.ClassDef) and cls.name == "EdgarFetch"
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "require_complete" not in fields, "EdgarFetch must not offer a partial-success mode"
    run = next(f for f in ast.walk(driver) if isinstance(f, ast.FunctionDef) and f.name == "run_edgar_fetch")
    order = [
        "raise"
        if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "failed" and _raises_incomplete(stmt)
        else "record"
        for stmt in run.body
        if (isinstance(stmt, ast.If) and _raises_incomplete(stmt))
        or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and getattr(stmt.value.func, "id", None) == "_record_tables")
    ]
    assert order == ["raise", "record"], f"a failed ticker must raise IncompleteEdgarRunError before any manifest entry is recorded: {order}"
    record = next(f for f in ast.walk(driver) if isinstance(f, ast.FunctionDef) and f.name == "_record_tables")
    coverage = [kw.value for call in ast.walk(record) if isinstance(call, ast.Call) for kw in call.keywords if kw.arg == "coverage_complete"]
    assert len(coverage) == 1 and isinstance(coverage[0], ast.Constant) and coverage[0].value is True

    missing = [
        path
        for path in STRICT_EDGAR_FILES
        if not any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "EdgarFetch"
            for call in ast.walk(ast.parse(_source(root / path)))
        )
    ]
    assert not missing, f"scheduled EDGAR walks no longer declare an EdgarFetch spec: {missing}"

    print("\n=== SANITY CHECK: strict EDGAR walks ===")
    print(f"  all {len(STRICT_EDGAR_FILES)} scheduled per-ticker EDGAR fetchers run the one strict driver path: no flag, a failed")
    print("  ticker raises IncompleteEdgarRunError before _record_tables, and recorded runs are coverage_complete=True")
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
