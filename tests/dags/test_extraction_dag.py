"""Static contract for the extraction DAG; Airflow is not installed in the test venv."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

from src.data_extract import cli as extraction_cli
from src.data_store.schema import Tables
from tests.dags.dag_harness import ALL_DONE, ALL_SUCCESS, load_dag, replay

DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_extraction.py"
AGG_DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_aggregation.py"
EDGAR_DRIVER_FILE = "src/data_extract/utils/edgar_extract/edgar_driver.py"
STRICT_EDGAR_FILES = (
    "src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py",
    "src/data_extract/utils/edgar_extract/fetch_8k_edgar.py",
    "src/data_extract/utils/edgar_extract/fetch_13d_edgar.py",
    "src/data_extract/utils/edgar_extract/fetch_13g_edgar.py",
    "src/data_extract/utils/edgar_extract/fetch_insider_edgar.py",
    "src/data_extract/utils/edgar_extract/fetch_filing_text.py",
    "src/data_extract/utils/edgar_extract/fetch_employees_text.py",
    "src/data_extract/utils/edgar_extract/fetch_def14a_text.py",
)
#: the universe step: the two ticker lists and the two identity evidence refreshes, then the sync (AC-020).
EVIDENCE_TASKS = ("sec_company_tickers", "sharadar_tickers", "sharadar_actions", "sec_submissions")
UNIVERSE_TASKS = (*EVIDENCE_TASKS, "universe_sync")
REQUIRED_COMMANDS = {
    "universe",
    "macro",
    "short-interest",
    "earnings-surprises",
    "price-history",
    "zips-download",
    "identity",
    "zips-ingest",
    "edgar-index",
    "thirteen-f",
    "thirteen-f-backfill",
    "insider-edgar",
    "superinvestors",
    "sec-fundamentals-facts",
    "sec-fundamentals-history",
    "fundamentals-check",
    "sec-employees-llm",
    "fundamentals-sharadar",
    "sec-def14a-llm",
    "sec-8k-items",
    "sec-8k-votes",
    "sec-13d",
    "sec-13g",
    "filing-text",
    "sec-employees-text",
    "sec-def14a-text",
    "extract-earnings-calls",
    "extraction-status",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_required_sources_are_scheduled_without_retired_attention_sources():
    source = _source(DAG_FILE)
    tree = ast.parse(source)
    scheduled = {
        call.args[0].value.split()[0]
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


#: identity producers, in stage order: the zip downloads (Form 3/4/5, Notes, FTD, SEC current tickers),
#: then the identity step (captures, build, sync).
IDENTITY_DOWNLOADS = {"zips_download"}
#: the two zip ingests (D-16a): the insider quarters alone, and notes + FTD.
ZIP_INGESTS = {"zips_ingest_insider": "insider", "zips_ingest_datasets": "notes --only ftd"}
#: the eight EDGAR document walks; each waits for the one nightly `edgar_index` refresh (P7b).
EDGAR_WALKS = {
    "insider_edgar",
    "sec_13d",
    "sec_13g",
    "sec_8k_items",
    "filing_text",
    "sec_employees_text",
    "sec_def14a_text",
    "sec_fundamentals_facts",
}
#: the LLM passes over the stored text tables (no SEC read): default pool, after the walk storing their text (D-05, M4).
LLM_TEXT_TASKS = {"sec_employees_llm": {"sec_employees_text"}, "sec_def14a_llm": {"sec_def14a_text", "sec_8k_items"}}
#: the SEC fundamentals producer; no path may join it to `fundamentals_sharadar` (AC-009).
SEC_FUNDAMENTALS = {"sec_fundamentals_facts", "sec_fundamentals_history"}
#: every task that reads the lineage; each must start after `identity`.
IDENTITY_CONSUMERS = {
    *ZIP_INGESTS,
    "insider_edgar",
    "short_interest",
    "thirteen_f",
    "extract_earnings_calls",
    "sec_fundamentals_facts",
    "fundamentals_sharadar",  # D16: the vendor history is keyed through the lineage
    "sec_employees_llm",
    "sec_def14a_llm",
    "sec_8k_items",
    "sec_13d",
    "sec_13g",
    "filing_text",
    "sec_employees_text",
    "sec_def14a_text",
}
#: derived from consumer tables; each must start after its parents.
DERIVED_PARENTS = {
    "sec_8k_votes": {"sec_def14a_llm"},
    **LLM_TEXT_TASKS,
    "sec_fundamentals_history": {"sec_fundamentals_facts"},
    "fundamentals_check": {"sec_fundamentals_history", "fundamentals_sharadar"},
    "thirteen_f_backfill": {"thirteen_f"},
    "superinvestors": {"thirteen_f"},
}
IDENTITY_INDEPENDENT = {"macro"}


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
    rule.TriggerRule = SimpleNamespace(ALL_SUCCESS="all_success", ALL_DONE="all_done")  # type: ignore[attr-defined]
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
    assert "thirteen_f >> thirteen_f_backfill" in source  # one EDGAR walk at a time: the backfill follows the nightly walk
    assert 'thirteen_f_backfill = fetch("thirteen-f-backfill", pool="sec_api")' in source
    assert "fundamentals >> fundamentals_sharadar" not in source  # the two producers are independent (AC-009)
    assert "sec_employees_llm] >> fundamentals_sharadar" not in source  # the cube reads headcount directly
    assert "[sec_def14a_text, sec_8k_items] >> sec_def14a_llm >> sec_8k_votes" in source
    assert "all_fetchers >> extraction_status >> identity_check >> trigger_aggregation" in source

    print("\n=== SANITY CHECK: extraction retry + gate wiring ===")
    print("  three retries per task, sec_api tasks take both pool slots, the backfill follows the 13F walk")
    print("  OK: every fetcher -> extraction_status -> identity_check -> trigger_aggregation")


def test_identity_stage_orders_downloads_identity_consumers_and_status(monkeypatch):
    graph, params = _load_dag_graph(monkeypatch)
    order = _topological_order(graph)
    assert len(order) == len(graph), "the extraction DAG has a cycle"
    position = {task: index for index, task in enumerate(order)}
    after_identity = _descendants(graph, "identity")

    # zip downloads -> identity (captures, build, sync) -> every consumer, the two zip ingests included
    for download in IDENTITY_DOWNLOADS:
        assert "identity" in graph[download], f"{download} must feed identity"
    assert graph["identity"] == IDENTITY_CONSUMERS | {"edgar_index", "extraction_status", "price_history", "earnings_surprises"}, sorted(
        graph["identity"]
    )
    for derived, parents in DERIVED_PARENTS.items():
        assert all(derived in graph[parent] for parent in parents), f"{derived} must wait for {sorted(parents)}"
        assert derived in after_identity
    # sources that read no lineage keep running beside the identity stage
    assert IDENTITY_INDEPENDENT.isdisjoint(after_identity), sorted(IDENTITY_INDEPENDENT & after_identity)
    assert not (IDENTITY_DOWNLOADS | {"identity"}) & after_identity
    for task, only in ZIP_INGESTS.items():
        assert params[task]["pool"] == "sec_bulk" and f" zips-ingest --only {only} -c " in str(params[task]["bash_command"]), task
    assert "insider_edgar" in graph["zips_ingest_insider"] and "insider_edgar" not in graph["zips_ingest_datasets"], "D-16a"

    # every fetcher upstream of the report; the report and the check run ALL_DONE (no source blocks the night)
    upstream_of_status = {task for task in graph if "extraction_status" in _descendants(graph, task)}
    assert upstream_of_status == set(graph) - {"extraction_status", "identity_check", "trigger_data_aggregation"}
    assert params["extraction_status"]["trigger_rule"] == "all_done"
    assert graph["extraction_status"] == {"identity_check"}
    # AC-039 / P37: the identity validator runs after the report and is the hard gate on aggregation
    fetchers = set(graph) - {"identity_check", "extraction_status", "trigger_data_aggregation", *UNIVERSE_TASKS}
    assert all("extraction_status" in graph[task] for task in fetchers), sorted(t for t in fetchers if "extraction_status" not in graph[t])
    assert graph["identity_check"] == {"trigger_data_aggregation"} and params["identity_check"]["trigger_rule"] == "all_done"
    assert params["trigger_data_aggregation"]["trigger_rule"] == "all_success"
    assert " -m src validate identity -o " in str(params["identity_check"]["bash_command"])
    for task in IDENTITY_DOWNLOADS:
        assert params[task]["pool"] == "sec_bulk"

    print("\n=== SANITY CHECK: identity stage order (AC-011, AC-037-039) ===")
    print(f"  {len(graph)} tasks, acyclic; {sorted(IDENTITY_DOWNLOADS)} -> identity -> {len(IDENTITY_CONSUMERS)} consumers")
    print(f"  zip ingests {sorted(ZIP_INGESTS)} in sec_bulk; only the insider one feeds insider_edgar (D-16a)")
    print(f"  first identity task at topological position {position['zips_download']}, gate at {position['extraction_status']}")
    print(
        f"  OK: {len(IDENTITY_INDEPENDENT)} non-identity sources run beside the stage; a failed identity run leaves its consumers unrun, the check still runs"
    )


def test_the_edgar_index_refreshes_once_before_the_seven_edgar_walks(monkeypatch):
    """P7b / D-17: one `edgar-index` task in the sec_api pool, after identity; the seven document walks wait for it."""
    graph, params = _load_dag_graph(monkeypatch)
    task = params["edgar_index"]
    assert task["pool"] == "sec_api" and task["pool_slots"] == 2 and " edgar-index -c " in str(task["bash_command"])
    assert "edgar_index" in graph["identity"]
    assert graph["edgar_index"] == EDGAR_WALKS | {"extraction_status"}, sorted(graph["edgar_index"])
    assert all(params[walk]["pool"] == "sec_api" for walk in EDGAR_WALKS)
    print("\n=== SANITY CHECK: one nightly EDGAR index refresh (P7b) ===")
    print(f"  identity -> edgar_index (sec_api, 2 slots) -> {sorted(EDGAR_WALKS)}")


def test_sec_and_sharadar_fundamentals_are_independent_producers(monkeypatch):
    """AC-009 / D16: no path joins `fundamentals_sharadar` to an SEC fundamentals task in either direction; both wait for
    identity; the check runs after both histories on ALL_DONE; headcount stays off both producers."""
    graph, params = _load_dag_graph(monkeypatch)
    after_sharadar = _descendants(graph, "fundamentals_sharadar")
    joined = {task for task in SEC_FUNDAMENTALS if task in after_sharadar or "fundamentals_sharadar" in _descendants(graph, task)}
    assert not joined, f"a path joins fundamentals_sharadar and {sorted(joined)}"
    assert "fundamentals" not in graph, "the combined fundamentals task is gone"
    assert {"fundamentals_sharadar", "sec_fundamentals_facts"} <= graph["identity"], "both producers wait for identity"
    assert "sec_fundamentals_facts" in graph["edgar_index"] and "sec_fundamentals_history" in graph["sec_fundamentals_facts"]
    assert params["sec_fundamentals_facts"]["pool"] == "sec_api" and params["sec_fundamentals_history"]["pool"] == "default_pool"
    check_parents = {task for task in graph if "fundamentals_check" in graph[task]}
    assert check_parents == {"universe_sync", "sec_fundamentals_history", "fundamentals_sharadar"}, sorted(check_parents)
    assert params["fundamentals_check"]["trigger_rule"] == "all_done" and graph["fundamentals_check"] == {"extraction_status"}
    assert " fundamentals-check -c " in str(params["fundamentals_check"]["bash_command"])
    assert " sec-fundamentals-history -c " in str(params["sec_fundamentals_history"]["bash_command"])
    assert not {"fundamentals_sharadar", *SEC_FUNDAMENTALS} & _descendants(graph, "sec_employees_llm"), "a producer waits for headcount"
    assert graph["sec_employees_llm"] == {"extraction_status"}

    dag = load_dag(DAG_FILE)
    sec_down, sharadar_down = replay(dag, ["sec_fundamentals_facts"]), replay(dag, ["fundamentals_sharadar"])
    assert sec_down["fundamentals_sharadar"] == "success" and sec_down["sec_fundamentals_history"] == "upstream_failed"
    assert sharadar_down["sec_fundamentals_history"] == "success"
    assert sec_down["fundamentals_check"] == sharadar_down["fundamentals_check"] == "success", "the check runs whatever either side did"

    print("\n=== SANITY CHECK: independent fundamentals producers (AC-009, D16) ===")
    print("  identity -> edgar_index -> sec_fundamentals_facts (sec_api) -> sec_fundamentals_history -> fundamentals_check")
    print("  identity -> fundamentals_sharadar -> fundamentals_check (ALL_DONE); no path between the two producers")
    print("  replay: facts failed -> Sharadar still succeeds; Sharadar failed -> SEC history still succeeds; the check runs both times. Validated.")


def test_the_llm_passes_run_in_the_default_pool_after_the_walk_storing_their_text(monkeypatch):
    """D-05 / M4: the text walks hold both sec_api slots; the LLM passes read only stored text, so they run in the
    default pool (no new pool), each after its walk; DEF 14A also waits for the 8-K items and feeds the vote parser."""
    graph, params = _load_dag_graph(monkeypatch)
    upstream = {task: {up for up, downs in graph.items() if task in downs} for task in graph}
    for walk, command in (("sec_employees_text", "sec-employees-text"), ("sec_def14a_text", "sec-def14a-text")):
        assert params[walk]["pool"] == "sec_api" and params[walk]["pool_slots"] == 2 and f" {command} -c " in str(params[walk]["bash_command"])
    for task, parents in LLM_TEXT_TASKS.items():
        assert params[task]["pool"] == "default_pool" and params[task]["pool_slots"] == 1, params[task]
        assert upstream[task] == {"universe_sync", "identity", *parents}, sorted(upstream[task])
    assert graph["sec_employees_text"] == {"sec_employees_llm", "extraction_status"}
    assert graph["sec_def14a_text"] == {"sec_def14a_llm", "extraction_status"}
    assert graph["sec_def14a_llm"] == {"sec_8k_votes", "extraction_status"} and upstream["sec_8k_votes"] == {"universe_sync", "sec_def14a_llm"}
    assert params["sec_8k_votes"]["pool"] == "default_pool" and params["sec_8k_votes"]["pool_slots"] == 1, params["sec_8k_votes"]
    assert {str(kwargs.get("pool")) for kwargs in params.values() if "pool" in kwargs} == {"default_pool", "sec_api", "sec_bulk"}, "no new pool"
    print("\n=== SANITY CHECK: text walks -> LLM passes (D-05, M4) ===")
    print("  sec_employees_text, sec_def14a_text: sec_api (2 slots); sec_employees_llm, sec_def14a_llm, sec_8k_votes: default_pool (1 slot)")
    print("  sec_employees_text -> sec_employees_llm; [sec_def14a_text, sec_8k_items] -> sec_def14a_llm -> sec_8k_votes; no new pool")


def test_price_history_waits_for_the_security_master(monkeypatch):
    """Q2d: the price fetch list adds the master's current secondary classes, so it runs after `identity`."""
    graph, params = _load_dag_graph(monkeypatch)
    assert "price_history" in graph["identity"] and params["price_history"]["trigger_rule"] == "all_done"
    waits_for = {task for task in graph if "price_history" in graph[task]}
    assert not {"zips_ingest_insider", "zips_ingest_datasets", "edgar_index"} & waits_for, "prices wait only for the identity step"
    print("\n=== SANITY CHECK: price_history order ===\n  identity -> price_history (ALL_DONE); no wait on the zip ingests")


def test_the_live_insider_walk_runs_in_the_sec_api_pool_after_the_bulk_parse(monkeypatch):
    """D-8: the Form 3/4/5 bulk parse stays in `sec_bulk`; its live EDGAR tail is its own `sec_api` task, so it never
    overlaps another EDGAR walk."""
    graph, params = _load_dag_graph(monkeypatch)
    bulk, walk = params["zips_ingest_insider"], params["insider_edgar"]

    assert bulk["pool"] == "sec_bulk" and " zips-ingest --only insider -c " in str(bulk["bash_command"])
    assert walk["pool"] == "sec_api" and walk["pool_slots"] == 2 and " insider-edgar -c " in str(walk["bash_command"])
    assert "insider_edgar" in graph["zips_ingest_insider"], "the live tail resumes from the bulk table's latest quarter"
    assert "insider_edgar" in _descendants(graph, "identity") and "extraction_status" in graph["insider_edgar"]
    edgar_in_bulk = [task for task, kwargs in params.items() if kwargs.get("pool") == "sec_bulk" and "edgar" in str(kwargs.get("bash_command"))]
    assert not edgar_in_bulk, edgar_in_bulk

    print("\n=== SANITY CHECK: insider tasks by pool (D-8) ===")
    print(f"  zips_ingest_insider: {bulk['pool']} -> {bulk['bash_command'].split(' -m src data_extract ')[1]}")
    print(f"  insider_edgar: {walk['pool']} ({walk['pool_slots']} slots) -> {walk['bash_command'].split(' -m src data_extract ')[1]}")
    print("  OK: the live EDGAR walk holds both sec_api slots and starts after the bulk parse and the identity sync.")


def test_one_price_task_and_no_alias_commands():
    dag = load_dag(DAG_FILE)
    price_tasks = [task_id for task_id, task in dag.tasks.items() if "price-history" in str(task.kwargs.get("bash_command", ""))]
    assert price_tasks == ["price_history"]
    assert not {"splits", "dividends"} & set(dag.tasks), "the merged download has one task"
    assert not {"splits", "dividends"} & set(extraction_cli.cli.commands), "the CLI aliases are gone"
    assert "price-history" in extraction_cli.cli.commands

    print("\n=== SANITY CHECK: one yfinance task ===")
    print("  price_history is the only task writing prices, prices_dividends and prices_splits; the splits/dividends aliases are removed")


def test_a_failed_fetcher_never_blocks_aggregation_but_a_failed_identity_check_does():
    """AC-015 for sources, P37 for identity: the report and the check run ALL_DONE; the trigger needs the check's success."""
    dag = load_dag(DAG_FILE)
    gate, trigger, check = dag.tasks["extraction_status"], dag.tasks["trigger_data_aggregation"], dag.tasks["identity_check"]
    assert gate.trigger_rule == ALL_DONE and check.trigger_rule == ALL_DONE and trigger.trigger_rule == ALL_SUCCESS
    assert trigger.upstream == {"identity_check"} and check.upstream == {"extraction_status"}
    fetchers = {
        task_id for task_id in dag.tasks if task_id not in {*UNIVERSE_TASKS, "identity_check", "extraction_status", "trigger_data_aggregation"}
    }
    assert gate.upstream == fetchers, "the report, then the identity check, wait for every fetcher"
    assert {"thirteen_f"} <= dag.tasks["thirteen_f_backfill"].upstream, "one EDGAR walk at a time: the backfill follows the nightly 13F walk"
    lenient = {"identity", "price_history", "earnings_surprises", "fundamentals_check"}
    assert all(dag.tasks[task_id].trigger_rule == ALL_SUCCESS for task_id in fetchers - lenient), "a fetcher still waits for its own sources"
    assert all(dag.tasks[task_id].trigger_rule == ALL_DONE for task_id in lenient), "F-106: a failed SEC download never skips prices"

    for failed in (["price_history"], ["identity"], ["thirteen_f", "zips_ingest_insider", "extraction_status"], ["edgar_index"]):
        states = replay(dag, failed)
        assert states["trigger_data_aggregation"] == "success", (failed, states)
    states = replay(dag, ["identity"])
    assert states["sec_fundamentals_facts"] == states["fundamentals_sharadar"] == states["zips_ingest_datasets"] == "upstream_failed"
    assert states["extraction_status"] == "success"
    assert states["price_history"] == "success", "prices read the stored master when the build fails"
    assert states["earnings_surprises"] == "success", "earnings read the stored trading symbols when the build fails"
    for download in sorted(IDENTITY_DOWNLOADS):
        states = replay(dag, [download])
        assert states["price_history"] == states["identity"] == states["zips_ingest_insider"] == states["sec_fundamentals_facts"] == "success", (
            download
        )
    for evidence in ("sharadar_actions", "sec_submissions"):
        states = replay(dag, [evidence])
        assert states["universe_sync"] == states["identity"] == states["sec_fundamentals_facts"] == "success", evidence
    held = replay(dag, ["identity_check"])
    assert held["trigger_data_aggregation"] == "upstream_failed", held
    assert held["extraction_status"] == "success" and all(held[task_id] == "success" for task_id in fetchers), (
        "extraction is never blocked by the check"
    )

    print("\n=== SANITY CHECK: extraction never blocked, aggregation gated on identity (AC-015, P37) ===")
    print(f"  {len(fetchers)} fetchers -> extraction_status (ALL_DONE) -> identity_check (ALL_DONE) -> trigger_data_aggregation (ALL_SUCCESS)")
    print(
        f"  replay: price_history / identity (its {len(IDENTITY_CONSUMERS)} consumers upstream_failed) / 13F + insider zips + the report / the index failed"
    )
    print("  -> aggregation triggered every time; identity_check failed -> every fetcher and the report succeed, the trigger is upstream_failed")
    print("  F-106: a failed zip download still runs identity, the consumers and price_history; a failed build still runs prices")


#: Modules whose entry points walk EDGAR filing by filing (the document walks, the EDGAR step running them, the 13F walks).
EDGAR_WALK_FILES = (
    *STRICT_EDGAR_FILES,
    "src/data_extract/transformers/step_extract_edgar.py",
    "src/data_extract/utils/institutionals/fetch_13f.py",
    "src/data_extract/utils/institutionals/fetch_13f_backfill.py",
    "src/data_extract/utils/institutionals/fetch_13f_vintages.py",
)


def _edgar_walk_names() -> set[str]:
    """Names `cli.py` imports from an EDGAR-walk module."""
    modules = {path.removesuffix(".py").replace("/", ".") for path in EDGAR_WALK_FILES}
    tree = ast.parse(_source(DAG_FILE.parents[2] / "src/data_extract/cli.py"))
    return {
        alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module in modules for alias in node.names
    }


def _command_names(command: str) -> set[str]:
    """Every name the CLI command's function refers to, through the `cli.py` helpers it calls."""
    seen: set[str] = set()
    pending = [extraction_cli.cli.commands[command].callback]
    while pending:
        tree = ast.parse(textwrap.dedent(inspect.getsource(pending.pop())))
        for name in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - seen:
            seen.add(name)
            helper = getattr(extraction_cli, name, None)
            if inspect.isfunction(helper) and helper.__module__ == extraction_cli.__name__:
                pending.append(helper)
    return seen


def test_every_edgar_walk_runs_in_the_sec_api_pool():
    """A task whose command walks EDGAR holds both sec_api slots, so only one EDGAR walk runs at a time;
    a walk in another pool would run beside it, two rate limiters against SEC's one 10 req/s."""
    dag = load_dag(DAG_FILE)
    walk_names = _edgar_walk_names()
    walks: dict[str, str] = {}
    for task_id, task in dag.tasks.items():
        bash = str(task.kwargs.get("bash_command", ""))
        if " data_extract " not in bash:
            continue
        command = bash.split(" data_extract ", 1)[1].split()[0]
        if _command_names(command) & walk_names:
            walks[task_id] = str(task.kwargs.get("pool"))
    assert {"thirteen_f", "sec_fundamentals_facts", "sec_13d"} <= set(walks), f"the walk detection lost known walks: {sorted(walks)}"
    outside = {task_id: pool for task_id, pool in walks.items() if pool != "sec_api"}
    assert not outside, f"EDGAR walks outside the sec_api pool: {outside}"
    # insiders: the zip leg is no walk; the EDGAR leg follows it, the identity step and the nightly index refresh
    assert "insider_edgar" in walks and not {"zips_ingest_insider", "zips_ingest_datasets", "zips_download", "identity"} & set(walks)
    assert {"zips_ingest_insider", "identity", "edgar_index"} <= dag.tasks["insider_edgar"].upstream
    # by hand the two legs are two commands: `zips-ingest --only insider` (no walk), then `insider-edgar` (the walk)
    assert "StepExtractZips" in _command_names("zips-ingest") and not _command_names("zips-ingest") & walk_names
    assert "StepExtractEdgar" in _command_names("insider-edgar")

    print("\n=== SANITY CHECK: one EDGAR walk at a time ===")
    print(f"  {len(walks)} tasks walk EDGAR (derived from the CLI code), all in sec_api: {sorted(walks)}")


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


def test_the_evidence_lands_before_the_universe_sync_and_never_stops_the_night():
    """AC-020 / PD-15: [ticker lists, sharadar_actions, sec_submissions] >> universe_sync >> zips_download >> identity >> consumers;
    the sync runs on ALL_DONE, so a failed evidence refresh leaves identity on the stored inputs."""
    dag = load_dag(DAG_FILE)
    parts = {task_id: str(dag.tasks[task_id].kwargs["bash_command"]).split(" data_extract ", 1)[1].split(" -c ")[0] for task_id in UNIVERSE_TASKS}

    assert parts == {
        "sec_company_tickers": "universe --only sec-tickers",
        "sharadar_tickers": "universe --only sharadar-tickers",
        "sharadar_actions": "universe --only sharadar-actions",
        "sec_submissions": "universe --only sec-submissions",
        "universe_sync": "universe --only sync",
    }
    assert dag.tasks["universe_sync"].upstream == set(EVIDENCE_TASKS) and dag.tasks["universe_sync"].trigger_rule == ALL_DONE
    assert all(dag.tasks[task_id].upstream == set() for task_id in EVIDENCE_TASKS), "the evidence tasks are the DAG roots"
    assert dag.tasks["sec_submissions"].kwargs["pool"] == "sec_bulk", "the bulk SEC download shares the bulk pool"
    assert {"universe_sync"} <= dag.tasks["zips_download"].upstream and {"zips_download"} <= dag.tasks["identity"].upstream
    roots = {task_id for task_id, task in dag.tasks.items() if task.upstream == {"universe_sync"}}
    assert "zips_download" in roots and not set(EVIDENCE_TASKS) & roots, "every fetcher hangs off the universe sync"
    assert "sharadar_actions" not in dag.tasks["fundamentals_sharadar"].upstream and "sharadar-actions" not in extraction_cli.cli.commands

    for failed in EVIDENCE_TASKS[2:]:
        states = replay(dag, [failed])
        assert states["universe_sync"] == states["identity"] == states["sec_fundamentals_facts"] == "success", (failed, states)
        assert states["trigger_data_aggregation"] == "success", (failed, states)
    agg = _source(AGG_DAG_FILE)
    assert "sharadar_actions" not in agg and "sec_submissions" not in agg, "the aggregation DAG is untouched"

    print("\n=== SANITY CHECK: evidence before identity (AC-020, PD-15) ===")
    print(f"  {sorted(EVIDENCE_TASKS)} (roots) -> universe_sync (ALL_DONE) -> zips_download -> identity; {len(roots)} tasks hang off the sync")
    print("  a failed sharadar_actions or sec_submissions -> sync, identity, consumers and the aggregation trigger still succeed. Validated.")


#: identity build modules: every `Tables.<name>` they name is an input of the build (captures write, the sync rewrites).
IDENTITY_BUILD_FILES = (
    "src/data_extract/utils/identity/symbol_tenure.py",
    "src/data_extract/transformers/step_extract_identity.py",
    "src/data_extract/utils/identity/evidence.py",
    "src/data_extract/utils/identity/company_anchor.py",
    "src/data_extract/utils/identity/cik_history.py",
    "src/data_extract/utils/identity/symbol_history.py",
    "src/data_extract/utils/identity/cusip_history.py",
)
#: the DAG task(s) that add rows to each identity input (keys are `Tables` attributes); `identity` writes its own outputs
#: and the captured evidence.
INPUT_PRODUCERS = {
    "universe_tickers": {"universe_sync"},
    "sec_company_tickers": {"sec_company_tickers"},
    "sharadar_tickers": {"sharadar_tickers"},
    "sharadar_actions": {"sharadar_actions"},
    "sec_cik_profile": {"sec_submissions"},
    "symbol_tenure": {"identity"},
    "universe_cik_symbol": {"identity"},
    "universe_cusip": {"identity"},
    "sec_fails_to_deliver_security": {"zips_ingest_datasets"},  # the FTD ingest stores the lines as filed
    "sec_short_volume_security": {"short_interest"},
    "prices": {"price_history"},
    "sec13f_hr": {"thirteen_f", "thirteen_f_backfill"},
    "sec_13d": {"sec_13d"},
    "sec_13g": {"sec_13g"},
}
#: inputs read one night old, as history only. U-3: prices, 13F and 13D/G CUSIPs. X15 (2026-11-04): the CUSIP history's
#: FINRA presence check (`sec_short_volume_security`). Q4 (2026-11-09): the CUSIP history reads the FTD lines the ingest
#: stored last night (`sec_fails_to_deliver_security`).
ONE_NIGHT_LAG = {"prices", "sec13f_hr", "sec_13d", "sec_13g", "sec_short_volume_security", "sec_fails_to_deliver_security"}


def test_identity_reads_no_input_written_later_in_the_night():
    """AC-020: every table the identity build reads is produced upstream of `identity`, by `identity` itself, or is a declared one-night lag."""
    dag = load_dag(DAG_FILE)
    root = DAG_FILE.parents[2]
    reads = {"universe_tickers"}  # `context.universe`
    for path in IDENTITY_BUILD_FILES:
        tree = ast.parse(_source(root / path))
        reads |= {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "Tables"
        }
    ancestors: set[str] = set()
    pending = list(dag.tasks["identity"].upstream)
    while pending:
        task_id = pending.pop()
        if task_id not in ancestors:
            ancestors.add(task_id)
            pending.extend(dag.tasks[task_id].upstream)

    undeclared = sorted(reads - set(INPUT_PRODUCERS))
    late = {table: sorted(INPUT_PRODUCERS[table] - ancestors - {"identity"}) for table in sorted(reads - set(undeclared))}
    late = {table: tasks for table, tasks in late.items() if tasks and table not in ONE_NIGHT_LAG}
    assert not undeclared, f"identity reads tables with no declared producer: {undeclared}"
    assert not late, f"identity reads tables written later in the night: {late}"
    assert {"sharadar_actions", "sec_submissions", "universe_sync", "zips_download"} <= ancestors
    assert set(INPUT_PRODUCERS) <= set(vars(Tables)), "producer keys are `Tables` attributes"

    print("\n=== SANITY CHECK: identity inputs land before identity (AC-020) ===")
    print(f"  build reads {sorted(reads)}")
    print(f"  {len(ancestors)} upstream tasks; one-night lag only for {sorted(reads & ONE_NIGHT_LAG)}; nothing else written later. Validated.")
