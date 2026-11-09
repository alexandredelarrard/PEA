"""Helper reuse contract (plan 3d) over `src/utils/` (minus `step.py`) and `src/data_extract/`.

1. No two function bodies of at least `MIN_NODES` AST nodes are identical once normalised: docstring, own name,
   annotations and decorators dropped, arguments and locals renamed in first-use order (globals, attributes and called
   names kept, so two bodies calling different helpers never collide). Shorter bodies are one call or one return (X9).
2. A `data_extract` function or class imported by the `data_extract/utils` code of two or more steps lives in
   `data_extract/utils/common/`. A step is the `data_extract/utils/<package>` it sits in (`structure.<sub>` for the
   structure sub-packages); the step orchestrators (`transformers/`) and the CLI call entry points and do not count.
   `PLACEMENT_EXCEPTIONS` lists the accepted ones with their reason; an entry nothing needs any more fails the test.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from collections import defaultdict
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
SCOPE = (REPO / "src" / "utils", REPO / "src" / "data_extract")
EXCLUDED = {REPO / "src" / "utils" / "step.py"}
COMMON = "src.data_extract.utils.common."
MIN_NODES = 25
#: Identical bodies accepted, as sorted `module.qualname` pairs -> reason. Empty.
DUPLICATE_EXCEPTIONS: dict[tuple[str, str], str] = {}
_DRIVER = "the EDGAR fetch driver and its parsers, owned by edgar_extract; fundamentals and 13F run their fetches through it"
_INSIDER = "the insider Form 4 pipeline shared by the EDGAR fetch and the bulk zip reader (moving it is a follow-up)"
_DEF14A = "a DEF 14A / 8-K vote parser owned by its structure package; the EDGAR fetch calls it on each filing"
_TAPE = "the security-tape contract shared by the FTD and short-volume tapes (moving it is a follow-up)"
_PROPAGATE = "a source's own re-stamp / re-parse entry point, called by identity propagation"
#: Accepted placement exceptions: owner module -> (names, reason). Phase 9b measured these; moving the shared ones
#: into `common/` is a separate decision (02-plan Phase 9b record).
PLACEMENT_EXCEPTIONS: dict[str, tuple[frozenset[str], str]] = {
    "src.data_extract.utils.edgar_extract.edgar_driver": (
        frozenset({"EdgarFetch", "EdgarScope", "FilingStamp", "RetryRounds", "header_subject_ciks", "universe_rows"}),
        _DRIVER,
    ),
    "src.data_extract.utils.edgar_extract.edgar_fillings": (frozenset({"list_filings"}), _DRIVER),
    "src.data_extract.utils.edgar_extract.edgar_index": (frozenset({"index_filing"}), _DRIVER),
    "src.data_extract.utils.edgar_extract.empty_markers": (frozenset({"marker_frame", "marker_mask"}), _DRIVER),
    "src.data_extract.utils.edgar_extract.html_text": (frozenset({"html_to_text"}), _DRIVER),
    "src.data_extract.utils.edgar_extract.parallel_fetch": (frozenset({"run_per_ticker"}), _DRIVER),
    "src.data_extract.utils.edgar_extract.insider_common": (
        frozenset(
            {"accession_batches", "build_insider_frame", "empty_footnotes", "exclusion_rows", "filter_footnotes", "insider_verdicts"}
            | {"log_exclusions", "screen_insider_rows", "screened_accessions", "stamp_lineage", "top_counts"}
        ),
        _INSIDER,
    ),
    "src.data_extract.utils.edgar_extract.insider_store": (frozenset({"patch_insider_source", "save_insider_source"}), _INSIDER),
    "src.data_extract.utils.fundamentals.build_history": (frozenset({"build_fundamentals_history"}), _PROPAGATE),
    "src.data_extract.utils.fundamentals.kpi_catalogue": (
        frozenset({"Catalogue", "load_catalogue"}),
        "the KPI catalogue owned by fundamentals; the Sharadar merge reads the same catalogue",
    ),
    "src.data_extract.utils.identity.symbol_tenure": (
        frozenset({"captured_dei_periods"}),
        "the DEI period ledger owned by identity; the Notes reader skips periods it holds",
    ),
    "src.data_extract.utils.institutionals.fetch_13f_backfill": (frozenset({"reread_hr_filings"}), _PROPAGATE),
    "src.data_extract.utils.institutionals.fetch_short_interest": (frozenset({"change_stamps", "restamp_short_volume"}), _PROPAGATE),
    "src.data_extract.utils.institutionals.security_tape": (
        frozenset(
            {"assert_tape_owners", "assert_tape_source", "commit_tape_source", "load_tape_cohorts", "nullable", "restamp_targets"}
            | {"scope_tape_rows", "stamp_changed", "summed_lines", "tape_key_matches", "validate_tape_frames"}
        ),
        _TAPE,
    ),
    "src.data_extract.utils.structure.def14a.ecd": (frozenset({"ecd_facts", "ecd_row", "has_ecd_block"}), _DEF14A),
    "src.data_extract.utils.structure.def14a.validate": (frozenset({"repair_main_row"}), _DEF14A),
    "src.data_extract.utils.structure.votes.guard": (frozenset({"has_vote_table"}), _DEF14A),
    "src.data_extract.utils.zips.fetch_fails_to_deliver": (frozenset({"restamp_fails"}), _PROPAGATE),
    "src.data_extract.utils.zips.fetch_financial_notes": (frozenset({"cached_notes_periods", "reparse_financial_notes"}), _PROPAGATE),
    "src.data_extract.utils.zips.fetch_insider_transactions": (frozenset({"reparse_insider_transactions", "restamp_insider_lineage"}), _PROPAGATE),
}


def _files() -> list[Path]:
    """Every in-scope module, sorted."""
    return sorted(p for root in SCOPE for p in root.rglob("*.py") if p not in EXCLUDED)


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(REPO).with_suffix("").parts)


def _local_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The function's arguments and the names it binds."""
    names = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node is not fn:
            names.add(node.name)
    return names - {fn.name}


class _Normaliser(ast.NodeTransformer):
    """Renames locals to `v0, v1, ...` in first-use order and drops annotations, decorators and function names."""

    def __init__(self, local: set[str]) -> None:
        self.local, self.map = local, {}

    def _alias(self, name: str) -> str:
        return self.map.setdefault(name, f"v{len(self.map)}")

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg, node.annotation = self._alias(node.arg), None
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._alias(node.id) if node.id in self.local else node.id
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        node.annotation = ast.Constant(None)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.returns, node.decorator_list = None, []
        node.name = self._alias(node.name) if node.name in self.local else "f"
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]  # noqa: N815


def _body_hash(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, int]:
    """The normalised body's digest and its AST node count."""
    node = copy.deepcopy(fn)
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        node.body = node.body[1:] or [ast.Pass()]
    node = _Normaliser(_local_names(fn)).visit(node)
    node.name = "f"
    dumped = ast.dump(node, annotate_fields=False, include_attributes=False)
    return hashlib.sha1(dumped.encode()).hexdigest(), sum(1 for _ in ast.walk(node))


def _functions(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Module-level functions and class methods (dunders left out), by qualified name."""
    out: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            out.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            methods = (m for m in node.body if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef) and not m.name.startswith("__"))
            out += [(f"{node.name}.{m.name}", m) for m in methods]
    return out


def _step(module: str) -> str | None:
    """The `data_extract/utils` package a module belongs to; None outside `data_extract/utils`."""
    parts = module.split(".")
    if parts[:3] != ["src", "data_extract", "utils"] or len(parts) < 5:
        return None
    return f"structure.{parts[4]}" if parts[3] == "structure" and len(parts) >= 6 else parts[3]


def _imports() -> dict[tuple[str, str], set[str]]:
    """(module, name) -> the `src/` modules importing it by `from module import name`."""
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in sorted((REPO / "src").rglob("*.py")):
        importer = _module_name(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    out[(node.module, alias.name)].add(importer)
    return out


def test_no_two_helper_bodies_are_identical() -> None:
    # compute: every body of MIN_NODES nodes or more, grouped by its normalised digest
    groups: dict[str, list[str]] = defaultdict(list)
    counted = 0
    for path in _files():
        module = _module_name(path)
        for qualname, fn in _functions(ast.parse(path.read_text(encoding="utf-8"))):
            digest, nodes = _body_hash(fn)
            if nodes >= MIN_NODES:
                groups[digest].append(f"{module}.{qualname}")
                counted += 1

    # check: no group of two or more, except the accepted pairs
    duplicates = [sorted(members) for members in groups.values() if len(members) > 1]
    open_ = [m for m in duplicates if not (len(m) == 2 and (m[0], m[1]) in DUPLICATE_EXCEPTIONS)]
    print("\n=== SANITY CHECK: identical helper bodies (plan 3d) ===")
    print(f"  {len(_files())} module(s), {counted} body(ies) of >= {MIN_NODES} AST nodes, {len(groups)} distinct digest(s)")
    for members in open_:
        print(f"  DUPLICATE: {', '.join(members)}")
    assert not open_, f"{len(open_)} identical helper body group(s): {open_}"
    print("  OK: no two non-trivial helper bodies are identical")


def test_helpers_shared_by_two_steps_live_in_common() -> None:
    # load: the import sites of every data_extract module-level function and class outside common/
    imports = _imports()
    owners = [p for p in _files() if _module_name(p).startswith("src.data_extract.utils.") and not _module_name(p).startswith(COMMON)]

    # compute: each name imported by the utils code of a step other than its own
    shared: dict[str, set[str]] = defaultdict(set)
    for path in owners:
        module = _module_name(path)
        defined = (n.name for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef | ast.ClassDef))
        for name in defined:
            steps = {_step(user) for user in imports.get((module, name), set())} | {_step(module)}
            if len(steps - {None}) >= 2:
                shared[module].add(name)

    # check: every shared name is an accepted exception, and every exception is still needed
    misplaced = {m: sorted(names - PLACEMENT_EXCEPTIONS.get(m, (frozenset(), ""))[0]) for m, names in shared.items()}
    misplaced = {m: names for m, names in misplaced.items() if names}
    stale = {m: sorted(names - shared.get(m, set())) for m, (names, _) in PLACEMENT_EXCEPTIONS.items()}
    stale = {m: names for m, names in stale.items() if names}
    print("\n=== SANITY CHECK: helper placement (plan 3d) ===")
    print(f"  {sum(len(v) for v in shared.values())} name(s) in {len(shared)} module(s) shared across steps outside common/")
    print(f"  accepted exceptions: {sum(len(v[0]) for v in PLACEMENT_EXCEPTIONS.values())} in {len(PLACEMENT_EXCEPTIONS)} module(s)")
    for module, names in sorted(misplaced.items()):
        print(f"  MISPLACED: {module}: {', '.join(names)}")
    for module, names in sorted(stale.items()):
        print(f"  STALE EXCEPTION: {module}: {', '.join(names)}")
    assert not misplaced, f"shared helpers outside data_extract/utils/common/: {misplaced}"
    assert not stale, f"placement exceptions nothing needs any more: {stale}"
    print("  OK: every helper shared by two steps lives in common/ or is a listed exception")
