"""Code-shape contract of the extraction steps: the flows a step's roots reach, and their strict call depth.

The walk resolves calls statically: a bare name defined in or imported from a `src` module, `self.<method>`, and
`<src module alias>.<name>`; a repository function passed as an argument counts as a call. Levels count from the step:
0 `run`, 1 a `run_*` sub-run (wherever it is called from), 2 a domain function, 3 a flat helper (the coding
standards number these 1-4). An `EdgarFetch` callback (`parse`, `is_subject`, `filings`, `run_scope`, and the
`row_fn` a parse binds) is a level-2 root wherever it is named: the driver, not the step, calls it; so is a flow's
per-ticker worker, which `run_per_ticker` calls once per ticker (checked to be bound and handed to it). Inside the
flow's `scope` a level-3 helper may call no repository function at all. The depth gate counts depths (1 `run`, 2 a
sub-run, 3 a function it calls or a callback/worker root, 4 a flat helper) along the deepest chain through the step
flow and domain modules; shared code (every other module) is called like a package and not counted. It holds every
step-flow function (`STEP_FLOW_MODULES`) to depth 4, with no allowlist; the domain modules' own depth is recorded
debt (wiki TODO). Every function of the domain modules (`DOMAIN_MODULES`), reached or not, keeps the per-function
contract below except the `Tables.<x>` literal rule.

Each reached function: at most 6 inputs, 90 lines and control-flow nesting 2; no pass-through; full annotations; no
logger parameter and no `log=` / `logger=` argument; no `print`; no import inside the body; a docstring of at most 4 lines; every blank-line-separated
block of a multi-block body opens with a comment; every table read (`.store.load`) in the first block, each table
once; store calls name `Tables.<x>`.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.data_extract.utils.edgar_extract import fetch_def14a_text, fetch_employees_text

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
MAX_INPUTS, MAX_LINES, MAX_NESTING, MAX_LEVEL, MAX_DOC_LINES, MAX_DEPTH = 6, 90, 2, 3, 4, 4
_BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)
_STORE_READS = {"load", "iter_load", "distinct", "summarize", "load_cohort"}
_STORE_CALLS = _STORE_READS | {"save", "replace", "update", "delete", "row_count"}
#: Keywords of an `EdgarFetch(...)` or `replace(<fetch>, ...)` declaration that hold a driver callback.
_CALLBACK_KEYS = {"parse", "is_subject", "filings", "run_scope"}
_CALLBACK_DECLARATIONS = {"EdgarFetch", "replace"}

_EDGAR = "src.data_extract.utils.edgar_extract."
_EMPLOYEES = "src.data_extract.utils.fundamentals.fundamentals_employees"
_DEF14A = "src.data_extract.utils.structure.def14a."


@dataclass(frozen=True)
class Flow:
    """One step's walk: its roots (`run` is level 0, any other root a level-1 sub-run), the modules whose callback
    declarations add level-2 roots, the per-ticker workers `run_per_ticker` calls (level-2 roots too), the in-scope
    prefixes and the expected `run()` order."""

    name: str
    step_module: str
    step_class: str
    roots: tuple[str, ...]
    scope: tuple[str, ...]
    callback_modules: tuple[str, ...] = ()
    workers: tuple[str, ...] = ()
    run_order: tuple[str, ...] = ()


IDENTITY = Flow(
    name="identity",
    step_module="src.data_extract.transformers.step_extract_identity",
    step_class="StepExtractIdentity",
    roots=("run",),
    scope=(
        "src.data_extract.transformers.step_extract_identity",
        "src.data_extract.utils.identity.",
        "src.utils.identity_flags",
        "src.data_extract.utils.common.identity",
    ),
    run_order=(
        "run_capture_evidence",
        "run_company_anchor",
        "run_cik_history",
        "run_symbol_history",
        "run_cusip_history",
        "run_lifecycle",
        "run_publish",
        "run_sync",
    ),
)
EDGAR_TEXT = Flow(
    name="edgar-text",
    step_module="src.data_extract.transformers.step_extract_edgar",
    step_class="StepExtractEdgar",
    roots=("run_employees_text", "run_def14a_text"),
    scope=(
        "src.data_extract.transformers.step_extract_edgar",
        f"{_EDGAR}fetch_employees_text",
        f"{_EDGAR}fetch_def14a_text",
        "src.data_extract.utils.common.workforce",
    ),
    callback_modules=(f"{_EDGAR}fetch_employees_text", f"{_EDGAR}fetch_def14a_text"),
    run_order=(
        "run_index",
        "run_insider",
        "run_13d",
        "run_13g",
        "run_8k",
        "run_filing_text",
        "run_employees_text",
        "run_def14a_text",
        "run_fundamentals_facts",
    ),
)
LLM_TEXT = Flow(
    name="llm-text",
    step_module="src.data_extract.transformers.step_extract_llm",
    step_class="StepExtractLlm",
    roots=("run_employees_llm", "run_def14a_llm"),
    scope=("src.data_extract.transformers.step_extract_llm", _EMPLOYEES, f"{_DEF14A}fetch"),
    workers=(f"{_EMPLOYEES}._ticker_employees",),
    run_order=("run_def14a_llm", "run_8k_votes", "run_employees_llm"),
)
FLOWS = (IDENTITY, EDGAR_TEXT, LLM_TEXT)
#: Domain modules of the text flows: every function keeps the per-function contract, reached or not, except the
#: `Tables.<x>` literal rule (D-06: the DEF 14A modules store through the `SECTIONS` registry tables). Their depth is
#: counted in the chains the depth gate walks.
DOMAIN_MODULES = (
    f"{_EDGAR}employee_document",
    "src.data_extract.utils.fundamentals.employee_guard",
    *(f"{_DEF14A}{name}" for name in ("work", "postprocess", "basis", "carve", "holders", "retry", "gender")),
)
#: The step-flow code of the text flows (the steps and their fetch/LLM orchestrators): held to depth 4.
STEP_FLOW_MODULES = (
    EDGAR_TEXT.step_module,
    LLM_TEXT.step_module,
    f"{_EDGAR}fetch_employees_text",
    f"{_EDGAR}fetch_def14a_text",
    _EMPLOYEES,
    f"{_DEF14A}fetch",
)


@dataclass
class _Module:
    """One parsed `src` module: its functions (`name` or `Class.method`), classes, imported names and module aliases."""

    name: str
    path: Path
    lines: list[str]
    tree: ast.Module
    functions: dict[str, ast.FunctionDef] = field(default_factory=dict)
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)
    names: dict[str, tuple[str, str]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


_MODULES: dict[str, _Module | None] = {}


def _module(name: str) -> _Module | None:
    """The parsed module `name` (None outside `src` or when it is a package without a file)."""
    if name not in _MODULES:
        path = REPO / Path(*name.split(".")).with_suffix(".py")
        _MODULES[name] = _parse(name, path) if name.startswith("src.") and path.exists() else None
    return _MODULES[name]


def _parse(name: str, path: Path) -> _Module:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    module = _Module(name=name, path=path, lines=text.splitlines(), tree=tree)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            module.functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            module.classes[node.name] = node
            module.functions.update({f"{node.name}.{item.name}": item for item in node.body if isinstance(item, ast.FunctionDef)})
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                target = f"{node.module}.{alias.name}"
                if (REPO / Path(*target.split(".")).with_suffix(".py")).exists():
                    module.aliases[alias.asname or alias.name] = target
                else:
                    module.names[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, ast.Import):
            module.aliases.update({alias.asname or alias.name: alias.name for alias in node.names})
    return module


def _resolve(module: _Module, owner: str | None, node: ast.expr) -> tuple[_Module, str] | None:
    """The repository function a call target (or a passed function) names, with its module; None for anything else."""
    if isinstance(node, ast.Name):
        if node.id in module.functions:
            return module, node.id
        if node.id in module.classes:
            return (module, f"{node.id}.__init__") if f"{node.id}.__init__" in module.functions else None
        if node.id in module.names:
            source, name = module.names[node.id]
            target = _module(source)
            return _resolve(target, None, ast.Name(id=name)) if target is not None else None
        return None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "self" and owner is not None and f"{owner}.{node.attr}" in module.functions:
            return module, f"{owner}.{node.attr}"
        if node.value.id in module.aliases and (target := _module(module.aliases[node.value.id])) is not None:
            return _resolve(target, None, ast.Name(id=node.attr))
    return None


def _calls(module: _Module, qualname: str, function: ast.FunctionDef) -> list[tuple[_Module, str]]:
    """Every repository function `function` calls or passes on, in source order (lambdas and inner bodies included)."""
    owner = qualname.split(".")[0] if "." in qualname else None
    out: list[tuple[_Module, str]] = []
    for node in ast.walk(function):
        targets = [node.func, *node.args, *(k.value for k in node.keywords)] if isinstance(node, ast.Call) else []
        out += [found for target in targets if (found := _resolve(module, owner, target)) is not None]
    return out


def _qualified(module: _Module, name: str) -> str:
    return f"{module.name}.{name}"


def _matches(name: str, prefixes: tuple[str, ...]) -> str | None:
    """The first prefix naming `name` itself, or a module/class prefix (`a.b.` or `a.b` + `.`) holding it."""
    return next((prefix for prefix in prefixes if name == prefix or name.startswith(prefix if prefix.endswith(".") else f"{prefix}.")), None)


def _callbacks(flow: Flow) -> set[str]:
    """The in-scope functions an `EdgarFetch(...)` or `replace(<fetch>, ...)` declaration of the flow's callback
    modules names under a callback keyword, bound through `partial` included."""
    found: set[str] = set()
    for name in flow.callback_modules:
        module = _module(name)
        assert module is not None, name
        for call in ast.walk(module.tree):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in _CALLBACK_DECLARATIONS):
                continue
            for keyword in (k for k in call.keywords if k.arg in _CALLBACK_KEYS):
                for node in ast.walk(keyword.value):
                    if isinstance(node, ast.Name) and (target := _resolve(module, None, node)) is not None:
                        found.add(_qualified(*target))
    return {name for name in found if _matches(name.rsplit(".", 1)[0], flow.scope)}


def _passed_to_run_per_ticker(worker: str) -> bool:
    """Whether a function of the worker's module binds it with `partial` and hands work to `run_per_ticker`."""
    module = _module(worker.rsplit(".", 1)[0])
    assert module is not None, worker
    name = worker.rsplit(".", 1)[1]
    for function in module.functions.values():
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        drives = any(call.func.id == "run_per_ticker" for call in calls)
        binds = any(call.func.id == "partial" and call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == name for call in calls)
        if drives and binds:
            return True
    return False


def _nesting(nodes: list[ast.stmt], depth: int = 0) -> int:
    """The deepest control-flow nesting of `nodes` (an `elif` counts as its `if`)."""
    deepest = depth
    for node in nodes:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(node, _BRANCHES):
            orelse = getattr(node, "orelse", [])
            elif_chain = isinstance(node, ast.If) and len(orelse) == 1 and isinstance(orelse[0], ast.If)
            bodies = [
                getattr(node, "body", []),
                [] if elif_chain else orelse,
                *(h.body for h in getattr(node, "handlers", [])),
                getattr(node, "finalbody", []),
            ]
            bodies += [case.body for case in getattr(node, "cases", [])]
            deepest = max(deepest, *(_nesting(body, depth + 1) for body in bodies), _nesting(orelse, depth) if elif_chain else depth)
    return deepest


def _blocks(module: _Module, function: ast.FunctionDef) -> list[list[str]]:
    """The body's source lines after the docstring, split at blank lines."""
    body = function.body[1:] if ast.get_docstring(function) is not None else function.body
    if not body:
        return []
    first = min(getattr(node, "lineno", 0) for node in body)
    comment_start = first
    while comment_start - 2 >= 0 and module.lines[comment_start - 2].strip().startswith("#"):
        comment_start -= 1
    lines = module.lines[comment_start - 1 : function.end_lineno]
    blocks: list[list[str]] = [[]]
    for line in lines:
        if line.strip():
            blocks[-1].append(line.strip())
        elif blocks[-1]:
            blocks.append([])
    return [block for block in blocks if block]


def _store_tables(node: ast.AST) -> list[tuple[str, str | None]]:
    """(method, `Tables.<x>` name or None) of every `<...>.store.<method>(table, ...)` call under `node`."""
    out = []
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in _STORE_CALLS:
            value = call.func.value
            if isinstance(value, ast.Attribute) and value.attr == "store" and call.args:
                table = call.args[0]
                named = isinstance(table, ast.Attribute) and isinstance(table.value, ast.Name) and table.value.id == "Tables"
                out.append((call.func.attr, table.attr if named else None))
    return out


def _is_pass_through(function: ast.FunctionDef) -> bool:
    """A body (docstring aside) that only returns or makes one call whose arguments are the function's own parameters."""
    body = function.body[1:] if ast.get_docstring(function) is not None else function.body
    if len(body) != 1 or not isinstance(body[0], ast.Return | ast.Expr) or not isinstance(body[0].value, ast.Call):
        return False
    params = {arg.arg for arg in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]}
    call = body[0].value
    values = [*call.args, *(k.value for k in call.keywords)]
    return bool(values) and all(isinstance(v, ast.Name) and v.id in params for v in values)


def _shape(module: _Module, qualname: str, function: ast.FunctionDef) -> list[str]:
    """Every contract violation of one function (not the call-depth rules)."""
    where = f"{_qualified(module, qualname)} ({module.path.relative_to(REPO).as_posix()}:{function.lineno})"
    args = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    inputs = [a for a in args if a.arg not in ("self", "cls")] + [a for a in (function.args.vararg, function.args.kwarg) if a is not None]
    problems = []
    if len(inputs) > MAX_INPUTS:
        problems.append(f"{where}: {len(inputs)} inputs > {MAX_INPUTS}")
    if (n := function.end_lineno - function.lineno + 1) > MAX_LINES:
        problems.append(f"{where}: {n} lines > {MAX_LINES}")
    if (depth := _nesting(function.body)) > MAX_NESTING:
        problems.append(f"{where}: nesting {depth} > {MAX_NESTING}")
    if _is_pass_through(function):
        problems.append(f"{where}: pass-through")
    if function.returns is None or any(a.annotation is None for a in inputs):
        problems.append(f"{where}: incomplete annotations")
    if any(a.arg in ("log", "logger") or "Logger" in ast.unparse(a.annotation or ast.Constant(None)) for a in inputs):
        problems.append(f"{where}: logger parameter")
    if any(isinstance(n, ast.Call) and any(k.arg in ("log", "logger") for k in n.keywords) for n in ast.walk(function)):
        problems.append(f"{where}: logger passed as an argument")
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print" for n in ast.walk(function)):
        problems.append(f"{where}: print")
    if any(isinstance(n, ast.Import | ast.ImportFrom) for n in ast.walk(function)):
        problems.append(f"{where}: import inside the function body")
    doc = ast.get_docstring(function)
    if not doc or len(doc.splitlines()) > MAX_DOC_LINES:
        problems.append(f"{where}: docstring missing or over {MAX_DOC_LINES} lines")
    blocks = _blocks(module, function)
    if len(blocks) > 1 and any(not block[0].startswith("#") for block in blocks):
        problems.append(f"{where}: a block does not open with a comment")
    problems += _read_problems(where, function, blocks)
    return problems


def _read_problems(where: str, function: ast.FunctionDef, blocks: list[list[str]]) -> list[str]:
    """Store calls name a `Tables.<x>`, each table is read once, every read sits in the first block."""
    calls = _store_tables(function)
    reads = [table for method, table in calls if method in _STORE_READS]
    problems = [f"{where}: store.{method} without Tables.<name>" for method, table in calls if table is None]
    if len(reads) != len(set(reads)):
        problems.append(f"{where}: a table read twice {sorted(reads)}")
    first = "\n".join(blocks[0]) if blocks else ""
    late = [table for table in reads if table is not None and f"Tables.{table}" not in first]
    if len(blocks) > 1 and late:
        problems.append(f"{where}: table read(s) outside the first block {late}")
    return problems


@dataclass
class _Walk:
    flow: Flow
    callbacks: set[str]
    reached: dict[str, int] = field(default_factory=dict)
    infrastructure: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)


def _root_level(walk: _Walk, module: _Module, name: str) -> int | None:
    """1 for a step `run_*` sub-run, 2 for a driver callback, wherever either is named; None for a plain call."""
    if module.name == walk.flow.step_module and name.startswith(f"{walk.flow.step_class}.run_"):
        return 1
    return 2 if _qualified(module, name) in walk.callbacks else None


def _visit(walk: _Walk, module: _Module, qualname: str, level: int, path: tuple[str, ...]) -> None:
    """Check one reached function and walk its calls one level down."""
    qualified = _qualified(module, qualname)
    if walk.reached.get(qualified, -1) >= level or qualified in path:
        return
    walk.reached[qualified] = level
    function = module.functions[qualname]
    walk.problems += _shape(module, qualname, function)
    for target, name in _calls(module, qualname, function):
        callee = _qualified(target, name)
        root = _root_level(walk, target, name)
        if root is not None:
            _visit(walk, target, name, root, (*path, qualified))
        elif level >= MAX_LEVEL:
            walk.problems.append(f"{qualified}: level-{level} helper calls repository function {callee}")
        elif not _matches(target.name, walk.flow.scope):
            walk.infrastructure.add(callee)
        else:
            _visit(walk, target, name, level + 1, (*path, qualified))


def _walk(flow: Flow) -> _Walk:
    module = _module(flow.step_module)
    assert module is not None
    assert all(_passed_to_run_per_ticker(worker) for worker in flow.workers), flow.workers
    walk = _Walk(flow=flow, callbacks=_callbacks(flow) | set(flow.workers))
    for root in flow.roots:
        _visit(walk, module, f"{flow.step_class}.{root}", 0 if root == "run" else 1, ())
    for callback in sorted(walk.callbacks):
        target = _module(callback.rsplit(".", 1)[0])
        assert target is not None
        _visit(walk, target, callback.rsplit(".", 1)[1], 2, ())
    return walk


def _bound_callbacks(value: object) -> set[str]:
    """The functions a declared callback runs: a function, or a `partial` and every function it binds."""
    if isinstance(value, functools.partial):
        bound = [value.func, *value.args, *value.keywords.values()]
        return set().union(*(_bound_callbacks(item) for item in bound))
    if callable(value) and hasattr(value, "__qualname__") and hasattr(value, "__module__"):
        return {f"{value.__module__}.{value.__qualname__}"}
    return set()


@pytest.mark.parametrize("flow", [flow for flow in FLOWS if flow.run_order], ids=lambda flow: flow.name)
def test_run_is_the_flow(flow: Flow) -> None:
    """`run()` is the named sub-runs in order, one call per line, nothing else."""
    module = _module(flow.step_module)
    assert module is not None
    run = module.functions[f"{flow.step_class}.run"]
    body = run.body[1:] if ast.get_docstring(run) is not None else run.body
    calls = [
        stmt.value.func.attr
        for stmt in body
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) and not stmt.value.args
    ]
    assert tuple(calls) == flow.run_order and len(body) == len(flow.run_order), [ast.unparse(stmt) for stmt in body]
    print(f"\n=== SANITY CHECK: {flow.step_class}.run() is the flow ===")
    print(f"  {' -> '.join(calls)}")


def test_the_walk_callbacks_are_the_declared_ones() -> None:
    """D-16: the statically enumerated callback roots are exactly the in-scope functions the two text walks run,
    the `run_scope`-bound DEF 14A parse and listing included."""
    declared = {fetch_employees_text.EMPLOYEES_TEXT_FETCH, fetch_def14a_text.DEF14A_TEXT_FETCH}
    with fetch_def14a_text.DEF14A_TEXT_FETCH.run_scope(None, [], None) as bound:  # binds partials only, reads nothing
        declared.add(bound)
    attributes: list[Callable | None] = [getattr(fetch, key) for fetch in declared for key in sorted(_CALLBACK_KEYS)]
    run_time = {name for value in attributes for name in _bound_callbacks(value) if _matches(name.rsplit(".", 1)[0], EDGAR_TEXT.scope)}

    static = _callbacks(EDGAR_TEXT)
    assert static == run_time, (sorted(static), sorted(run_time))
    assert {name.rsplit(".", 1)[1] for name in static} == {"_employee_rows", "proxy_is_subject", "_def14a_run_scope", "_def14a_rows", "proxy_filings"}
    print("\n=== SANITY CHECK: EdgarFetch callback roots (D-16) ===")
    print(f"  {len(static)} level-2 callback roots, static == bound at run time: {sorted(name.rsplit('.', 1)[1] for name in static)}")


@pytest.mark.parametrize("flow", FLOWS, ids=lambda flow: flow.name)
def test_every_function_the_flow_reaches_keeps_the_code_shape_contract(flow: Flow) -> None:
    """Plan 3c on the AST: depth, flat helpers, limits, blocks, reads, annotations, no inline import, and no
    exception: every reached function is checked."""
    walk = _walk(flow)

    assert not walk.problems, "\n".join(walk.problems)
    assert all(f"{flow.step_module}.{flow.step_class}.{root}" in walk.reached for root in flow.roots)
    levels: dict[int, int] = {}
    for level in walk.reached.values():
        levels[level] = levels.get(level, 0) + 1
    print(f"\n=== SANITY CHECK: {flow.name} code shape ===")
    print(f"  {len(walk.reached)} function(s) reached from {', '.join(flow.roots)}; per level {dict(sorted(levels.items()))}; 0 violations")
    print(f"  callback roots: {len(walk.callbacks)}; infrastructure leaves called from levels 1-2: {len(walk.infrastructure)}")
    print("  OK: every reached function keeps the contract; level-3 helpers call no repository function")


@pytest.mark.parametrize("name", DOMAIN_MODULES, ids=lambda name: name.rsplit(".", 1)[1])
def test_every_domain_function_keeps_the_function_contract(name: str) -> None:
    """D-06: every function of a domain module, reached or not, keeps the per-function contract (limits, nesting,
    blocks, docstring, annotations, no logger input, reads in the first block)."""
    module = _module(name)
    assert module is not None, name
    problems = [problem for qualname, function in module.functions.items() for problem in _shape(module, qualname, function)]
    problems = [problem for problem in problems if "without Tables.<name>" not in problem]

    assert not problems, "\n".join(problems)
    print(f"\n=== SANITY CHECK: domain module {name.rsplit('.', 1)[1]} keeps the function contract ===")
    print(f"  {len(module.functions)} function(s), {len(module.lines)} lines, 0 violations")


def _owner(qualified: str) -> tuple[_Module, str]:
    """The parsed module and in-module name (`Class.method` included) of a qualified function."""
    parts = qualified.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        module = _module(".".join(parts[:cut]))
        if module is not None and ".".join(parts[cut:]) in module.functions:
            return module, ".".join(parts[cut:])
    raise KeyError(qualified)


def _depths(flows: tuple[Flow, ...]) -> tuple[dict[str, tuple[int, tuple[str, ...]]], dict[str, list[str]]]:
    """Deepest depth (with its chain) of every step-flow or domain function the flows' roots reach, and each one's
    counted callees; shared code is a leaf. A root a caller names (a callback, a worker) runs at its own depth."""
    counted = (*STEP_FLOW_MODULES, *DOMAIN_MODULES)
    roots = [(f"{flow.step_module}.{flow.step_class}.{root}", 2) for flow in flows for root in flow.roots]
    roots += [(name, 3) for flow in flows for name in sorted(_callbacks(flow) | set(flow.workers))]
    named = {name for name, _ in roots}
    depth: dict[str, tuple[int, tuple[str, ...]]] = {}
    callees: dict[str, list[str]] = {}
    stack = [(root, level, ()) for root, level in reversed(roots)]
    while stack:
        qualified, level, chain = stack.pop()
        if qualified in chain or depth.get(qualified, (0, ()))[0] >= level:
            continue
        module, qualname = _owner(qualified)
        depth[qualified] = (level, (*chain, qualified))
        found = [_qualified(t, n) for t, n in _calls(module, qualname, module.functions[qualname]) if t.name in counted]
        callees[qualified] = [name for name in dict.fromkeys(found) if name not in named]
        stack += [(name, level + 1, (*chain, qualified)) for name in reversed(callees[qualified])]
    return depth, callees


def test_every_step_flow_function_sits_at_depth_4() -> None:
    """The coding-standards depth rule on the EDGAR text and LLM text step flows, along the deepest chain: `run` 1, a
    sub-run 2, a function it calls 3, a flat helper 4; shared code is a leaf, step-flow code has no allowlist."""
    depth, callees = _depths((EDGAR_TEXT, LLM_TEXT))
    flow = {name: value for name, value in depth.items() if _owner(name)[0].name in STEP_FLOW_MODULES}
    deeper = [f"depth {d} > {MAX_DEPTH}: {' -> '.join(chain)}" for d, chain in flow.values() if d > MAX_DEPTH]
    calling = [f"depth-{d} {name} calls {', '.join(callees[name])}" for name, (d, _) in flow.items() if d == MAX_DEPTH and callees[name]]
    debt = sum(1 for name, (d, _) in depth.items() if name not in flow and (d > MAX_DEPTH or (d == MAX_DEPTH and callees[name])))

    assert not deeper and not calling, "\n".join(sorted(calling) + sorted(deeper)).replace("src.data_extract.", "")
    print("\n=== SANITY CHECK: the text step flows sit at depth <= 4 ===")
    print(f"  {len(flow)} step-flow function(s), deepest {max(d for d, _ in flow.values())}; 0 violations")
    print(f"  domain-module depth debt (wiki TODO): {debt} function(s) of {len(depth) - len(flow)} reached")
