"""
refactor_metrics.py  (scripts/dod/refactor_metrics.py)
------------------------------------------------------
The REFACTOR Definition-of-Done report: what the restructure actually did to the code, and
seven gates that catch the ways a "cleanup" silently breaks this repo.

    "$PY" scripts/dod/refactor_metrics.py --slug drop-legacy-peers
    "$PY" scripts/dod/refactor_metrics.py --slug fix-imports --tests tests/utils/test_x.py

Gates
    G1  the tests you named are green AND print a sanity conclusion (AGENTS.md)
    G2  tests/data_store/test_store_boundary.py green -- required iff `data_store/` was touched
    G3  no NEW `print(` under `src/` (logging goes through `self._log` / `context.log`)
    G4  public API of every touched module unchanged, or every call site updated
    G5  doc-sync: a `src/` change with no `AGENTS.md`/`README.md`/`docs/*.md` change
    G6  docstring lines did not SHRINK in a touched file  <- the anti-LOC-minimisation guard
    G7  `AGENTS.md` <= 70 lines

Design notes
  * STDLIB `ast` ONLY. Measuring code must not add a dependency, and `ast` gives exactly what
    is needed: which lines are docstring, which names are public.
  * LOC IS A METRIC, NEVER A GATE. It appears only in §3. G6 is the counterweight: this
    codebase's docstrings are load-bearing (`outliers.detect_level_outliers` documents
    two false-positive bugs a shorter version reintroduces), so an agent rewarded for fewer
    lines would delete precisely the most valuable text. Shrinking is allowed -- it just has
    to be SAID in §5.
  * BASELINE COMES FROM GIT, not from a stored copy. `git show <baseline>:<path>` is the only
    honest "before". A generator may shell out to git; a hook may not.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dod.report_common import (                              # noqa: E402
    Gate, announce, changed_files, head_sha, link_prefix, load_baseline, metrics_table,
    report_path, repo_root, write_report,
)

GENERATOR = "scripts/dod/refactor_metrics.py@1"
AGENTS_MD_MAX_LINES = 70
STORE_BOUNDARY_TEST = "tests/data_store/test_store_boundary.py"
DOC_SYNC_SURFACE = ("AGENTS.md", "README.md", "docs/")
#: A duplicated block shorter than this is a coincidence (imports, `return None`), not a smell.
SHINGLE_LINES = 6


# --------------------------------------------------------------------------- #
# ast measurement                                                             #
# --------------------------------------------------------------------------- #
def _docstring_lines(tree: ast.Module) -> set[int]:
    """1-based line numbers occupied by docstrings (module, class, def, async def).

    Counts the whole span of the string literal, because that is what a "shorter file" would
    delete -- a docstring's value is in its body, not in the quotes that open it."""
    out: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            start = first.lineno
            end = getattr(first, "end_lineno", start) or start
            out.update(range(start, end + 1))
    return out


def loc_profile(source: str) -> dict:
    """`{total, code, docstring, comment, blank}` for one file. Unparseable -> code-only."""
    lines = source.splitlines()
    total = len(lines)
    blank = sum(1 for ln in lines if not ln.strip())
    comment = sum(1 for ln in lines if ln.strip().startswith("#"))
    try:
        doc = _docstring_lines(ast.parse(source))
    except SyntaxError:
        doc = set()
    # a docstring line can also look blank (the empty line inside a triple-quoted block);
    # attribute it to the docstring so the four buckets sum to `total`
    doc_n = len(doc)
    blank = sum(1 for i, ln in enumerate(lines, 1) if not ln.strip() and i not in doc)
    comment = sum(1 for i, ln in enumerate(lines, 1)
                  if ln.strip().startswith("#") and i not in doc)
    return {"total": total, "code": total - blank - comment - doc_n,
            "docstring": doc_n, "comment": comment, "blank": blank}


def public_api(source: str) -> list[str]:
    """Module-level names importable from this module: non-`_` defs, classes AND imports.

    Imports count. `from src.data_store.errors import UnknownTableError` at the top of
    `schema.py` keeps `schema.UnknownTableError` working even though the class now lives
    elsewhere -- so the name never left this module's surface. Ignoring imports made G4 report
    a moved-but-re-exported class as a breaking removal, which is a false positive: the whole
    point of the re-export is that no call site has to change."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                out.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if not name.startswith("_") and name != "*":
                    out.append(name)
    return sorted(set(out))


def _normalise(line: str) -> str:
    return " ".join(line.split())


def duplication(sources: dict[str, str], *, window: int = SHINGLE_LINES) -> dict:
    """Shingle-based near-duplicate scan across `{path: source}`.

    Hashes every `window`-line run of NON-blank, NON-comment, normalised code and counts the
    runs that occur more than once. Deliberately crude: the number is a prompt to go look, not
    a verdict -- `docs/coding_standard.md` documents places where duplication is INTENTIONAL."""
    seen: Counter[str] = Counter()
    where: dict[str, list[str]] = {}
    for path, src in sources.items():
        body = [_normalise(ln) for ln in src.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
        for i in range(max(0, len(body) - window + 1)):
            key = hashlib.sha1("\n".join(body[i:i + window]).encode("utf-8")).hexdigest()[:12]
            seen[key] += 1
            where.setdefault(key, []).append(f"{path}:{i + 1}")
    dups = {k: v for k, v in seen.items() if v > 1}
    worst = sorted(dups.items(), key=lambda kv: -kv[1])[:5]
    return {
        "window_lines": window,
        "shingles": sum(seen.values()),
        "duplicated_shingles": sum(dups.values()),
        "duplicate_ratio": (sum(dups.values()) / sum(seen.values())) if seen else 0.0,
        "top_sites": [{"count": c, "at": where[k][:4]} for k, c in worst],
    }


# --------------------------------------------------------------------------- #
# git "before"                                                                #
# --------------------------------------------------------------------------- #
def baseline_source(path: str, sha: str, root: Path) -> str | None:
    """The file's content at `sha`, or None if it did not exist then (a NEW file)."""
    if not sha or sha == "unknown":
        return None
    try:
        out = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=root,
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _run_pytest(targets: list[str], root: Path) -> tuple[bool, str, str]:
    """(passed, one-line summary, stdout tail). `-s` so a sanity print is visible to G1."""
    py = sys.executable
    try:
        out = subprocess.run([py, "-m", "pytest", *targets, "-q", "-s"], cwd=root,
                             capture_output=True, text=True, timeout=1800, check=False)
    except subprocess.TimeoutExpired:
        return False, "timed out after 1800s", ""
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"could not run pytest: {e}", ""
    text = out.stdout + out.stderr
    tail = [ln for ln in text.splitlines() if ln.strip()][-1:] or ["(no output)"]
    return out.returncode == 0, tail[-1].strip(), text


def _has_sanity_print(pytest_output: str) -> bool:
    """AGENTS.md: a test isn't done until it prints a sanity-check conclusion."""
    lowered = pytest_output.lower()
    return "sanity" in lowered and ("validated" in lowered or "conclusion" in lowered)


# --------------------------------------------------------------------------- #
# Gates                                                                       #
# --------------------------------------------------------------------------- #
def _infer_tests(touched: list[str], root: Path) -> list[str]:
    """`src/a/b.py` -> `tests/a/test_b.py` when that file exists. Mirrors the tests/ layout."""
    found = []
    for p in touched:
        if p.startswith("tests/") and p.endswith(".py"):
            found.append(p)
        elif p.startswith("src/") and p.endswith(".py"):
            parts = Path(p).parts[1:]
            cand = Path("tests", *parts[:-1], f"test_{parts[-1]}")
            if (root / cand).is_file():
                found.append(cand.as_posix())
    return sorted(set(found))


def build_gates(touched: list[str], tests: list[str], root: Path, sha: str,
                per_file: dict[str, dict]) -> tuple[list[Gate], dict]:
    gates: list[Gate] = []
    evidence: dict = {}

    # ---- G1: the named/inferred tests ------------------------------------ #
    if tests:
        ok, summary, out = _run_pytest(tests, root)
        sanity = _has_sanity_print(out)
        gates.append(Gate("G1", f"targeted tests green ({len(tests)} file(s))",
                          ok and sanity,
                          f"{summary}" + ("" if sanity else "; NO sanity-check print found")))
        evidence["pytest_summary"] = summary
        evidence["pytest_targets"] = tests
    else:
        gates.append(Gate("G1", "targeted tests green", None,
                          "no test file named (--tests) and none inferred from the touched paths"))

    # ---- G2: the store boundary ------------------------------------------ #
    touched_store = [p for p in touched if p.startswith("src/data_store/")]
    if touched_store:
        ok, summary, _ = _run_pytest([STORE_BOUNDARY_TEST], root)
        gates.append(Gate("G2", "store boundary test green", ok,
                          f"{len(touched_store)} data_store file(s) touched; {summary}"))
    else:
        gates.append(Gate("G2", "store boundary test green", None,
                          "no `src/data_store/` file touched"))

    # ---- G3: no NEW print( in src/ --------------------------------------- #
    added = 0
    offenders = []
    for p in touched:
        if not (p.startswith("src/") and p.endswith(".py")):
            continue
        now = (root / p).read_text(encoding="utf-8", errors="replace") if (root / p).is_file() else ""
        before = baseline_source(p, sha, root) or ""
        delta = now.count("print(") - before.count("print(")
        if delta > 0:
            added += delta
            offenders.append(f"{p} (+{delta})")
    gates.append(Gate("G3", "no new `print(` under src/", added == 0,
                      ", ".join(offenders) if offenders else "none added"))

    # ---- G4: public API of touched modules ------------------------------- #
    removed: list[str] = []
    for p, prof in per_file.items():
        gone = sorted(set(prof.get("api_before") or []) - set(prof.get("api_after") or []))
        removed.extend(f"{p}::{n}" for n in gone)
    if removed:
        # a removed public name is only a FAIL if something still imports it
        still = []
        for item in removed:
            name = item.split("::", 1)[1]
            try:
                hit = subprocess.run(["git", "grep", "-l", "-w", name, "--", "src", "tests",
                                      "scripts", "app"], cwd=root, capture_output=True,
                                     text=True, timeout=30, check=False)
            except (OSError, subprocess.SubprocessError):
                hit = None
            if hit is not None and hit.stdout.strip():
                still.append(f"{item} (still referenced)")
        gates.append(Gate("G4", "public API stable or call sites updated", not still,
                          "; ".join(still) if still
                          else f"removed {len(removed)}, no dangling reference: "
                               f"{', '.join(removed[:4])}"))
    else:
        gates.append(Gate("G4", "public API stable or call sites updated", True,
                          "no public name removed"))

    # ---- G5: doc-sync ---------------------------------------------------- #
    touched_src = [p for p in touched if p.startswith("src/")]
    touched_doc = [p for p in touched if p.startswith(DOC_SYNC_SURFACE)]
    if touched_src:
        gates.append(Gate("G5", "docs moved with the code", bool(touched_doc),
                          f"{len(touched_src)} src file(s); docs touched: "
                          f"{', '.join(touched_doc) if touched_doc else 'NONE'}"))
    else:
        gates.append(Gate("G5", "docs moved with the code", None, "no `src/` file touched"))

    # ---- G6: docstrings did not shrink ----------------------------------- #
    shrunk = [f"{p} {prof['docstring_before']}->{prof['docstring_after']}"
              for p, prof in per_file.items()
              if prof.get("docstring_before") is not None
              and prof["docstring_after"] < prof["docstring_before"]]
    gates.append(Gate("G6", "docstring lines did not shrink", not shrunk,
                      "; ".join(shrunk) if shrunk
                      else "no touched file lost docstring lines"))
    if shrunk:
        evidence["g6_note"] = ("A shrink is ALLOWED but must be justified in §5 -- say which "
                              "docstring you removed and why it was not load-bearing.")

    # ---- G7: AGENTS.md budget -------------------------------------------- #
    agents = root / "AGENTS.md"
    n = len(agents.read_text(encoding="utf-8").splitlines()) if agents.is_file() else -1
    gates.append(Gate("G7", f"AGENTS.md <= {AGENTS_MD_MAX_LINES} lines",
                      0 <= n <= AGENTS_MD_MAX_LINES, f"{n} lines"))

    return gates, evidence


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="REFACTOR Definition-of-Done report")
    ap.add_argument("--slug", required=True, help="short kebab-case name for this task")
    ap.add_argument("--tests", default="", help="comma/space separated test paths for G1")
    ap.add_argument("--baseline", default=None, help="sha to diff against (default: session baseline)")
    ap.add_argument("--session-id", default=None)
    args = ap.parse_args(argv)

    root = repo_root()
    baseline = load_baseline(args.session_id, root=root)
    sha = args.baseline or baseline.get("head_sha") or head_sha(root)

    touched = changed_files(sha, root=root)
    py_touched = [p for p in touched if p.endswith(".py")]

    per_file: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for p in py_touched:
        fp = root / p
        now = fp.read_text(encoding="utf-8", errors="replace") if fp.is_file() else ""
        before = baseline_source(p, sha, root)
        after_prof = loc_profile(now) if now else {"total": 0, "code": 0, "docstring": 0,
                                                  "comment": 0, "blank": 0}
        before_prof = loc_profile(before) if before is not None else None
        if now:
            sources[p] = now
        per_file[p] = {
            "path": p,
            "status": "new" if before is None else ("deleted" if not now else "modified"),
            "loc_before": before_prof["total"] if before_prof else None,
            "loc_after": after_prof["total"],
            "code_after": after_prof["code"],
            "comment_after": after_prof["comment"],
            "docstring_before": before_prof["docstring"] if before_prof else None,
            "docstring_after": after_prof["docstring"],
            "api_before": public_api(before) if before is not None else [],
            "api_after": public_api(now) if now else [],
        }

    tests = [t for t in args.tests.replace(",", " ").split() if t] or _infer_tests(touched, root)
    gates, evidence = build_gates(touched, tests, root, sha, per_file)
    dup = duplication(sources)

    metrics_rows = [
        {"file": p, "status": d["status"], "loc_before": d["loc_before"],
         "loc_after": d["loc_after"], "code": d["code_after"],
         "docstring": d["docstring_after"], "comment": d["comment_after"],
         "public_api": len(d["api_after"])}
        for p, d in sorted(per_file.items())
    ]
    totals = {
        "files_touched": len(touched),
        "python_files_touched": len(py_touched),
        "loc_after_total": sum(d["loc_after"] for d in per_file.values()),
        "docstring_lines_after_total": sum(d["docstring_after"] for d in per_file.values()),
        "docstring_lines_before_total": sum(d["docstring_before"] or 0
                                            for d in per_file.values()),
    }

    up = link_prefix(report_path("REFACTOR", args.slug, root=root), root)
    metrics_md = "\n\n".join([
        "_Observations only — no verdicts. LOC is never a target (see "
        f"[definition_of_done.md]({up}docs/definition_of_done.md))._",
        "**Per touched Python file**",
        metrics_table(metrics_rows, ["file", "status", "loc_before", "loc_after", "code",
                                    "docstring", "comment", "public_api"]),
        "**Totals**",
        metrics_table([totals], list(totals)),
        f"**Duplication** (shingle = {dup['window_lines']} normalised code lines): "
        f"{dup['duplicated_shingles']:,} of {dup['shingles']:,} "
        f"({dup['duplicate_ratio']:.1%}) recur. "
        f"Some duplication in this repo is deliberate and documented — read the docstring "
        f"before removing any.",
    ])

    non_py = [p for p in touched if not p.endswith(".py")]
    evidence_md = "\n".join([
        f"- baseline: `{sha[:12]}`" if sha != "unknown" else "- baseline: unknown",
        f"- tests run: {', '.join(tests) if tests else 'none'}",
        f"- non-Python files touched ({len(non_py)}): "
        f"{', '.join(non_py[:12]) if non_py else 'none'}",
    ] + ([f"- {k}: {v}" for k, v in evidence.items()]))

    scope_md = "\n".join([
        f"**Files written ({len(touched)}):** "
        + (", ".join(f"`{p}`" for p in touched[:20]) or "none")
        + (f" … +{len(touched) - 20} more" if len(touched) > 20 else ""),
        "",
        f"**Sample scope:** whole repository working tree vs `{sha[:12]}` "
        f"(a refactor's scope is the diff, not a data sample).",
    ])

    # The block is hash-checked and read by humans, so it carries NUMBERS, not the raw API
    # dumps -- embedding every public name made a 480-line report of which 400 lines were JSON.
    # The names only matter when one was REMOVED, which is exactly what G4 reports.
    slim = {
        p: {"status": d["status"], "loc_before": d["loc_before"], "loc_after": d["loc_after"],
            "code": d["code_after"], "comment": d["comment_after"],
            "docstring_before": d["docstring_before"], "docstring_after": d["docstring_after"],
            "public_api_count": len(d["api_after"]),
            "public_api_removed": sorted(set(d["api_before"]) - set(d["api_after"])) or None}
        for p, d in sorted(per_file.items())
    }
    payload = {"scope": {"baseline_sha": sha, "touched": touched, "tests": tests},
               "metrics": {"per_file": slim, "totals": totals, "duplication": dup}}

    path = write_report("REFACTOR", args.slug, generator=GENERATOR, gates=gates,
                        metrics_md=metrics_md, evidence_md=evidence_md, payload=payload,
                        scope_md=scope_md, root=root, session_id=args.session_id)
    announce(path, gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
