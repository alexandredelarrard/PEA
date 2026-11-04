"""Code-shape contract of the identity build (plan 3c), enforced on every function `StepExtractIdentity.run` reaches.

The walk resolves calls statically: a bare name defined in or imported from a `src` module, `self.<method>`, and
`<src module alias>.<name>`; a repository function passed as an argument counts as a call. Levels count from the step:
0 `run`, 1 a `run_*` sub-run, 2 a domain function, 3 a flat helper. A function outside the identity build (`SCOPE`) is
infrastructure: a level-1 or level-2 function may call it, a level-3 helper may call no repository function at all.
`ALLOWLIST` names the stages phases 4b-4d still rewrite; their subtrees are not walked, and the list is empty after 4d.

Each reached function: at most 6 inputs, 90 lines and control-flow nesting 2; no pass-through; full annotations; no
logger parameter; no `print`; a docstring of at most 4 lines; every blank-line-separated block of a multi-block body
opens with a comment; every table read (`.store.load`) in the first block, each table once; store calls name `Tables.<x>`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
STEP_MODULE = "src.data_extract.transformers.step_extract_identity"
STEP_CLASS = "StepExtractIdentity"
FLOW = (
    "run_capture_evidence",
    "run_company_anchor",
    "run_cik_history",
    "run_symbol_history",
    "run_cusip_history",
    "run_lifecycle",
    "run_publish",
    "run_propagate",
)
#: The identity build: the step, the identity package, the flags and the `Identity` reader (plan 3c).
SCOPE = (STEP_MODULE, "src.data_extract.utils.identity.", "src.utils.identity_flags", "src.data_extract.utils.common.identity")
#: Not yet rewritten (prefix of a qualified name -> the sub-phase that rewrites it). Shrinks each sub-phase; empty after 4d.
ALLOWLIST = {
    f"{STEP_MODULE}.{STEP_CLASS}.run_symbol_history": "4b",
    "src.data_extract.utils.identity.symbol_tenure.": "4b",
    "src.data_extract.utils.identity.entity_lineage.": "4b",
    f"{STEP_MODULE}.{STEP_CLASS}.run_cusip_history": "4c",
    "src.data_extract.utils.identity.capture_ftd.": "4c",
    "src.data_extract.utils.identity.security_master.": "4c",
    f"{STEP_MODULE}.{STEP_CLASS}.run_lifecycle": "4d",
    f"{STEP_MODULE}.{STEP_CLASS}.run_publish": "4d",
    f"{STEP_MODULE}.{STEP_CLASS}.run_propagate": "4d",
    "src.data_extract.utils.identity.identity_propagate.": "4d",
    "src.data_extract.utils.common.identity.": "4d",
}
MAX_INPUTS, MAX_LINES, MAX_NESTING, MAX_LEVEL, MAX_DOC_LINES = 6, 90, 2, 3, 4
_BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)
_STORE_READS = {"load", "iter_load", "distinct", "summarize", "load_cohort"}
_STORE_CALLS = _STORE_READS | {"save", "replace", "update", "delete", "row_count"}


@dataclass
class _Module:
    """One parsed `src` module: its functions (`name` or `Class.method`), classes, imported names and module aliases."""

    name: str
    path: Path
    lines: list[str]
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
    module = _Module(name=name, path=path, lines=text.splitlines())
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


def _allowlisted(qualified: str) -> str | None:
    return next((phase for prefix, phase in ALLOWLIST.items() if qualified == prefix or qualified.startswith(prefix) and prefix.endswith(".")), None)


def _in_scope(module: _Module) -> bool:
    return any(module.name == prefix or prefix.endswith(".") and module.name.startswith(prefix) for prefix in SCOPE)


def _nesting(nodes: list[ast.stmt], depth: int = 0) -> int:
    """The deepest control-flow nesting of `nodes` (an `elif` counts as its `if`)."""
    deepest = depth
    for node in nodes:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(node, _BRANCHES):
            elif_chain = isinstance(node, ast.If) and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
            bodies = [node.body, [] if elif_chain else node.orelse, *(h.body for h in getattr(node, "handlers", [])), getattr(node, "finalbody", [])]
            bodies += [case.body for case in getattr(node, "cases", [])]
            deepest = max(deepest, *(_nesting(body, depth + 1) for body in bodies), _nesting(node.orelse, depth) if elif_chain else depth)
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
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print" for n in ast.walk(function)):
        problems.append(f"{where}: print")
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
    reached: dict[str, int] = field(default_factory=dict)
    allowlisted: dict[str, str] = field(default_factory=dict)
    infrastructure: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)


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
        if (phase := _allowlisted(callee)) is not None:
            walk.allowlisted[callee] = phase
        elif level >= MAX_LEVEL:
            walk.problems.append(f"{qualified}: level-{level} helper calls repository function {callee}")
        elif not _in_scope(target):
            walk.infrastructure.add(callee)
        else:
            _visit(walk, target, name, level + 1, (*path, qualified))


def _walk() -> _Walk:
    module = _module(STEP_MODULE)
    assert module is not None
    walk = _Walk()
    _visit(walk, module, f"{STEP_CLASS}.run", 0, ())
    return walk


def test_run_is_the_flow() -> None:
    """REQ-009: `run()` is the eight named sub-runs in order, one call per line, nothing else."""
    module = _module(STEP_MODULE)
    assert module is not None
    run = module.functions[f"{STEP_CLASS}.run"]
    body = run.body[1:] if ast.get_docstring(run) is not None else run.body
    calls = [
        stmt.value.func.attr
        for stmt in body
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) and not stmt.value.args
    ]
    assert tuple(calls) == FLOW and len(body) == len(FLOW), [ast.unparse(stmt) for stmt in body]
    print("\n=== SANITY CHECK: run() is the flow ===")
    print(f"  {' -> '.join(calls)}")


def test_every_function_run_reaches_keeps_the_code_shape_contract() -> None:
    """Plan 3c on the AST: depth, flat helpers, limits, blocks, reads, annotations; the allowlist names what 4b-4d rewrite."""
    walk = _walk()

    assert not walk.problems, "\n".join(walk.problems)
    levels: dict[int, int] = {}
    for level in walk.reached.values():
        levels[level] = levels.get(level, 0) + 1
    allowed_prefixes = sorted(set(ALLOWLIST.values()))
    print("\n=== SANITY CHECK: identity code shape (plan 3c) ===")
    print(f"  {len(walk.reached)} identity function(s) reached from run(); per level {dict(sorted(levels.items()))}; 0 violations")
    print(f"  infrastructure leaves called from levels 1-2: {len(walk.infrastructure)}; level-3 helpers call no repository function")
    for phase in allowed_prefixes:
        names = sorted(name for name, p in walk.allowlisted.items() if p == phase)
        print(f"  allowlisted for {phase} ({len(names)} reached): {', '.join(n.rsplit('.', 2)[-2] + '.' + n.rsplit('.', 1)[-1] for n in names)}")
    print("  OK: the rewritten stages keep the contract; the allowlist is the 4b-4d backlog and must be empty after 4d")
