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
    "src/data_extract/utils/edgar_extract/fetch_def14a_edgar.py",
    "src/data_extract/utils/edgar_extract/fetch_filing_text.py",
)
REQUIRED_COMMANDS = {
    "sharadar-tickers",
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
    "thirteen-f-managers",
    "fundamentals",
    "fundamentals-employees",
    "fundamentals-sharadar",
    "sharadar-actions",
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


#: identity producers, in stage order: the zip downloads (Form 3/4/5, Notes, Statements, FTD, SEC current tickers),
#: then the identity step (captures, build, propagation).
IDENTITY_DOWNLOADS = {"zips_download"}
#: the two zip ingests (D-16a): the insider quarters alone, and notes + statements + FTD.
ZIP_INGESTS = {"zips_ingest_insider": "insider", "zips_ingest_datasets": "notes --only statements --only ftd"}
#: the six EDGAR document walks; each waits for the one nightly `edgar_index` refresh (P7b).
EDGAR_WALKS = {"insider_edgar", "sec_13d", "sec_13g", "sec_8k_items", "filing_text", "def14a_edgar"}
#: every task that reads the lineage; each must start after `identity`.
IDENTITY_CONSUMERS = {
    *ZIP_INGESTS,
    "insider_edgar",
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
    "fundamentals_sharadar": {"fundamentals"},
}
IDENTITY_INDEPENDENT = {
    "thirteen_f",
    "superinvestors",
    "thirteen_f_managers",
    "macro",
    "earnings_surprises",
    "thirteen_f_backfill",
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
    assert "fundamentals >> fundamentals_sharadar" in source
    assert "fundamentals_employees] >> fundamentals_sharadar" not in source  # the cube reads headcount directly
    assert "[sec_8k_items, def14a] >> sec_8k_votes" in source
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

    # zip downloads -> identity (captures, build, propagation) -> every consumer, the two zip ingests included
    for download in IDENTITY_DOWNLOADS:
        assert "identity" in graph[download], f"{download} must feed identity"
    assert graph["identity"] == IDENTITY_CONSUMERS | {"edgar_index", "extraction_status", "price_history"}, sorted(graph["identity"])
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
    fetchers = set(graph) - {"identity_check", "extraction_status", "trigger_data_aggregation", "sharadar_tickers", "universe"}
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


def test_the_edgar_index_refreshes_once_before_the_six_edgar_walks(monkeypatch):
    """P7b / D-17: one `edgar-index` task in the sec_api pool, after identity; the six document walks wait for it."""
    graph, params = _load_dag_graph(monkeypatch)
    task = params["edgar_index"]
    assert task["pool"] == "sec_api" and task["pool_slots"] == 2 and " edgar-index -c " in str(task["bash_command"])
    assert "edgar_index" in graph["identity"]
    assert graph["edgar_index"] == EDGAR_WALKS | {"extraction_status"}, sorted(graph["edgar_index"])
    assert all(params[walk]["pool"] == "sec_api" for walk in EDGAR_WALKS)
    print("\n=== SANITY CHECK: one nightly EDGAR index refresh (P7b) ===")
    print(f"  identity -> edgar_index (sec_api, 2 slots) -> {sorted(EDGAR_WALKS)}")


def test_the_sharadar_merge_does_not_wait_for_headcount(monkeypatch):
    """Headcount is read by the cube from `fundamentals_employees`, so the merged history does not wait for it."""
    graph, _ = _load_dag_graph(monkeypatch)
    assert "fundamentals_sharadar" in graph["fundamentals"]
    assert "fundamentals_sharadar" not in _descendants(graph, "fundamentals_employees"), "the merge still waits for headcount"
    assert "extraction_status" in graph["fundamentals_employees"]
    print("\n=== SANITY CHECK: headcount is not a merge input ===")
    print("  fundamentals -> fundamentals_sharadar; fundamentals_employees -> extraction_status only")


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
    print("  OK: the live EDGAR walk holds both sec_api slots and starts after the bulk parse and the identity propagation.")


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
        task_id for task_id in dag.tasks if task_id not in {"sharadar_tickers", "universe", "identity_check", "extraction_status", "trigger_data_aggregation"}
    }
    assert gate.upstream == fetchers, "the report, then the identity check, wait for every fetcher"
    assert {"thirteen_f"} <= dag.tasks["thirteen_f_backfill"].upstream, "one EDGAR walk at a time: the backfill follows the nightly 13F walk"
    lenient = {"identity", "price_history"}
    assert all(dag.tasks[task_id].trigger_rule == ALL_SUCCESS for task_id in fetchers - lenient), "a fetcher still waits for its own sources"
    assert all(dag.tasks[task_id].trigger_rule == ALL_DONE for task_id in lenient), "F-106: a failed SEC download never skips prices"

    for failed in (["price_history"], ["identity"], ["thirteen_f", "zips_ingest_insider", "extraction_status"], ["edgar_index"]):
        states = replay(dag, failed)
        assert states["trigger_data_aggregation"] == "success", (failed, states)
    states = replay(dag, ["identity"])
    assert states["fundamentals"] == states["zips_ingest_datasets"] == "upstream_failed" and states["extraction_status"] == "success"
    assert states["price_history"] == "success", "prices read the stored master when the build fails"
    for download in sorted(IDENTITY_DOWNLOADS):
        states = replay(dag, [download])
        assert states["price_history"] == states["identity"] == states["zips_ingest_insider"] == states["fundamentals"] == "success", download
    held = replay(dag, ["identity_check"])
    assert held["trigger_data_aggregation"] == "upstream_failed", held
    assert held["extraction_status"] == "success" and all(held[task_id] == "success" for task_id in fetchers), (
        "extraction is never blocked by the check"
    )

    print("\n=== SANITY CHECK: extraction never blocked, aggregation gated on identity (AC-015, P37) ===")
    print(f"  {len(fetchers)} fetchers -> extraction_status (ALL_DONE) -> identity_check (ALL_DONE) -> trigger_data_aggregation (ALL_SUCCESS)")
    print("  replay: price_history / identity (its 12 consumers upstream_failed) / 13F + insider zips + the report / the index failed")
    print("  -> aggregation triggered every time; identity_check failed -> every fetcher and the report succeed, the trigger is upstream_failed")
    print("  F-106: a failed zip download still runs identity, the consumers and price_history; a failed build still runs prices")


#: Modules whose entry points walk EDGAR filing by filing (the document walks, the EDGAR step running them, the 13F walks).
EDGAR_WALK_FILES = (
    *STRICT_EDGAR_FILES,
    "src/data_extract/transformers/step_extract_edgar.py",
    "src/data_extract/utils/institutionals/fetch_13f.py",
    "src/data_extract/utils/institutionals/fetch_13f_backfill.py",
    "src/data_extract/utils/institutionals/fetch_13f_managers.py",
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
    assert {"thirteen_f", "fundamentals", "sec_13d"} <= set(walks), f"the walk detection lost known walks: {sorted(walks)}"
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


def test_sharadar_actions_is_an_sec_independent_task():
    """AC-010: the market-wide Sharadar actions table refreshes nightly; an SEC walk failure never blocks it."""
    dag = load_dag(DAG_FILE)
    task = dag.tasks["sharadar_actions"]
    assert str(task.kwargs["bash_command"]).split(" data_extract ", 1)[1].split()[0] == "sharadar-actions"
    assert task.upstream == {"universe"}, "no SEC edge: it waits for the universe sync only"
    assert "sharadar_actions" in dag.tasks["extraction_status"].upstream
    assert "sharadar_actions" not in dag.tasks["fundamentals_sharadar"].upstream
    assert dag.tasks["fundamentals_sharadar"].upstream >= {"fundamentals"}
    assert "fundamentals >> fundamentals_sharadar" in _source(DAG_FILE)

    states = replay(dag, ["fundamentals"])
    assert states["sharadar_actions"] == "success", states
    assert states["fundamentals_sharadar"] == "upstream_failed", states
    assert states["trigger_data_aggregation"] == "success", states

    agg = _source(AGG_DAG_FILE)
    assert "sharadar_actions" not in agg and "sharadar-actions" not in agg, "the aggregation DAG is untouched"
    assert {"fundamentals", "fundamentals_employees", "def14a"} <= IDENTITY_CONSUMERS and "sharadar_actions" not in IDENTITY_CONSUMERS
    assert DERIVED_PARENTS["fundamentals_sharadar"] == {"fundamentals"}

    print("\n=== SANITY CHECK: sharadar_actions task (AC-010) ===")
    print(f"  upstream {sorted(task.upstream)}; feeds extraction_status; not an ancestor of fundamentals_sharadar")
    print("  fundamentals failure -> sharadar_actions success, fundamentals_sharadar upstream_failed; schedule/start_date unchanged")


def test_the_universe_is_synced_after_sharadar_tickers_and_before_every_fetcher():
    """AC-016: sharadar-tickers -> universe -> every other extract task; no seed-universe task."""
    dag = load_dag(DAG_FILE)
    sync = dag.tasks["universe"]
    assert str(sync.kwargs["bash_command"]).split(" data_extract ", 1)[1].split()[0] == "universe"
    assert sync.upstream == {"sharadar_tickers"} and dag.tasks["sharadar_tickers"].upstream == set()
    roots = {task_id for task_id, task in dag.tasks.items() if task.upstream == {"universe"}}
    assert "seed_universe" not in dag.tasks and roots, "fetchers hang off the universe sync"
    print("\n=== SANITY CHECK: universe ordering ===")
    print(f"  sharadar_tickers -> universe -> {len(roots)} root fetchers; seed_universe is gone.")
