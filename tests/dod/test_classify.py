"""
test_classify.py  (tests/dod/test_classify.py)
----------------------------------------------
Table-driven tests for the definition-of-done classifier: every R (required) and N (not
required) rule, both type-resolution paths, and the question-turn exemption.

Parsing math over synthetic known-truth fixtures, per AGENTS.md: the classifier is pure string
logic over a transcript, so a synthetic transcript IS the ground truth here -- no real data
needed or wanted.
"""
from __future__ import annotations

import time

import pytest

from tests.dod.conftest import text_line, tool_line, write_transcript

REPO = "c:/repo"


def _state(lib, tmp_path, lines: list[str]) -> dict:
    """Scan a synthetic transcript into the accumulated session state the hook builds."""
    path = write_transcript(tmp_path, lines)
    from pathlib import Path
    scan = lib.scan_transcript(path, 0)
    return lib.merge_scan({}, scan, Path(REPO))


# --------------------------------------------------------------------------- #
# Requirement rules                                                           #
# --------------------------------------------------------------------------- #
REQUIRED_CASES = [
    ("R1 two src writes",
     [tool_line("Edit", file_path="src/a.py"), tool_line("Edit", file_path="src/b.py")],
     "R1", "REFACTOR"),
    ("R1 one src + one test write",
     [tool_line("Write", file_path="src/utils/x.py"),
      tool_line("Write", file_path="tests/utils/test_x.py")],
     "R1", "REFACTOR"),
    ("R2 single risk-zone write (context.py)",
     [tool_line("Edit", file_path="src/context.py")],
     "R2", "REFACTOR"),
    ("R2 single risk-zone write (data_store -> DATA)",
     [tool_line("Edit", file_path="src/data_store/store.py")],
     "R2", "DATA"),
    ("R2 configs is a risk zone",
     [tool_line("Edit", file_path="configs/portfolio.yml")],
     "R2", "REFACTOR"),
    ("R2 fingerprint baseline",
     [tool_line("Edit", file_path="tests/data_aggregate/aggregate_fingerprint_baseline.json")],
     "R2", "REFACTOR"),
    ("R3 pipeline command, NO edits at all",
     [tool_line("Bash", command="\"$PY\" -m src data_aggregate cube-status")],
     "R3", "DATA"),
    ("R3 modelling pipeline command, no edits",
     [tool_line("Bash", command="\"$PY\" -m src modelling train -F")],
     "R3", "MODELLING"),
    ("R3 main.py counts as a pipeline run",
     [tool_line("Bash", command="\"$PY\" main.py")],
     "R3", "REFACTOR"),
    ("R4 eight writes anywhere",
     [tool_line("Write", file_path=f"notes/n{i}.txt") for i in range(8)],
     "R4", "REFACTOR"),
]


@pytest.mark.parametrize("label,lines,rule,kind", REQUIRED_CASES,
                         ids=[c[0] for c in REQUIRED_CASES])
def test_report_is_required(dod_lib, tmp_path, label, lines, rule, kind):
    verdict = dod_lib.classify(_state(dod_lib, tmp_path, lines))
    assert verdict["required"] is True, f"{label}: expected required, got {verdict}"
    assert any(r.startswith(rule) for r in verdict["reasons"]), \
        f"{label}: expected {rule} to fire, got {verdict['reasons']}"
    assert verdict["kind"] == kind, f"{label}: expected {kind}, got {verdict['kind']}"


NOT_REQUIRED_CASES = [
    ("N-idle pure Q&A, only reads",
     [tool_line("Read", file_path="src/context.py"),
      tool_line("Grep", pattern="def run"),
      text_line("`Context` exposes `.log`, not `.logger`.")],
     "N-idle"),
    ("N-idle no tools at all",
     [text_line("The cube is built by StepBuildCube.")],
     "N-idle"),
    ("N-docs single markdown typo fix",
     [tool_line("Edit", file_path="docs/runbook.md")],
     "N-docs"),
    ("N-question turn ends by asking the user",
     [tool_line("Edit", file_path="src/a.py"), tool_line("Edit", file_path="src/b.py"),
      text_line("I can do either. Which horizon should I optimise for?")],
     "N-question"),
    ("N-question AskUserQuestion was used",
     [tool_line("Edit", file_path="src/a.py"), tool_line("Edit", file_path="src/b.py"),
      tool_line("AskUserQuestion", questions=[])],
     "N-question"),
    ("N-reports only a report was written",
     [tool_line("Write", file_path="reports/2026/08/2026-08-17_x__DATA.md")],
     "N-reports"),
    ("N-small a single src file, no pipeline",
     [tool_line("Edit", file_path="src/utils/xs.py")],
     "N-small"),
]


@pytest.mark.parametrize("label,lines,rule", NOT_REQUIRED_CASES,
                         ids=[c[0] for c in NOT_REQUIRED_CASES])
def test_report_not_required(dod_lib, tmp_path, label, lines, rule):
    verdict = dod_lib.classify(_state(dod_lib, tmp_path, lines))
    assert verdict["required"] is False, f"{label}: expected NOT required, got {verdict}"
    assert verdict["kind"] is None
    assert any(r.startswith(rule) for r in verdict["reasons"]), \
        f"{label}: expected {rule}, got {verdict['reasons']}"


# --------------------------------------------------------------------------- #
# Type resolution: writes outrank commands; commands decide when none exist    #
# --------------------------------------------------------------------------- #
TYPE_CASES = [
    ("writes win over commands", ["src/modelling/step_train.py", "src/modelling/x.py"],
     ["\"$PY\" -m src data_extract prices"], "MODELLING"),
    ("data writes -> DATA", ["src/data_extract/a.py", "src/validate/b.py"], [], "DATA"),
    ("validate/ is DATA", ["src/validate/analyze_history.py", "src/validate/x.py"], [], "DATA"),
    ("strategies -> MODELLING", ["src/strategies/ls_equity.py", "src/strategies/b.py"],
     [], "MODELLING"),
    ("sql/ -> DATA", ["sql/schema.sql", "sql/x.sql"], [], "DATA"),
    ("everything else -> REFACTOR", ["src/utils/http.py", "src/utils/io.py"], [], "REFACTOR"),
    ("no writes -> command decides (DATA)", [], ["\"$PY\" -m src data_peers deduce"], "DATA"),
    ("no writes -> command decides (MODELLING)", [], ["\"$PY\" -m src portfolio blend"],
     "MODELLING"),
    ("reports/ writes are ignored for typing",
     ["reports/x__DATA.md"], ["\"$PY\" -m src modelling train"], "MODELLING"),
]


@pytest.mark.parametrize("label,writes,commands,kind", TYPE_CASES,
                         ids=[c[0] for c in TYPE_CASES])
def test_resolve_kind(dod_lib, label, writes, commands, kind):
    assert dod_lib.resolve_kind(writes, commands) == kind, label


# --------------------------------------------------------------------------- #
# Question detection                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("Which horizon?", True),
    ("Should I proceed?**", True),
    ("Do you want A or B? ", True),
    ("- Shall I continue?)", True),
    ("Done. Tests pass.", False),
    ("I asked whether it was stale, and it is.", False),
    ("", False),
])
def test_ends_in_question(dod_lib, text, expected):
    assert dod_lib.ends_in_question(text) is expected


# --------------------------------------------------------------------------- #
# Incremental scan + performance                                              #
# --------------------------------------------------------------------------- #
def test_scan_is_incremental(dod_lib, tmp_path):
    """A second scan from the stored cursor sees ONLY the new lines."""
    path = write_transcript(tmp_path, [tool_line("Edit", file_path="src/a.py")])
    first = dod_lib.scan_transcript(path, 0)
    assert first["writes"] == ["src/a.py"]

    with path.open("a", encoding="utf-8") as fh:
        fh.write(tool_line("Edit", file_path="src/b.py") + "\n")
    second = dod_lib.scan_transcript(path, first["cursor"])
    assert second["writes"] == ["src/b.py"], "the cursor must skip already-scanned lines"
    assert second["cursor"] > first["cursor"]


def test_scan_recovers_from_a_truncated_transcript(dod_lib, tmp_path):
    """A cursor past EOF (file replaced/compacted) must restart, not return nothing."""
    path = write_transcript(tmp_path, [tool_line("Edit", file_path="src/a.py")])
    scan = dod_lib.scan_transcript(path, 10_000_000)
    assert scan["writes"] == ["src/a.py"]


def test_classification_is_fast_on_a_large_transcript(dod_lib, tmp_path):
    """Perf budget: the hook is one process on every turn, so the scan must stay well under
    the 3 s wall-clock bail. 5k lines is a long session."""
    lines = []
    for i in range(5000):
        lines.append(tool_line("Edit", file_path=f"src/mod_{i % 50}.py")
                     if i % 3 else tool_line("Bash", command=f"echo {i}"))
    path = write_transcript(tmp_path, lines)

    t0 = time.perf_counter()
    scan = dod_lib.scan_transcript(path, 0)
    from pathlib import Path
    verdict = dod_lib.classify(dod_lib.merge_scan({}, scan, Path(REPO)))
    elapsed = time.perf_counter() - t0

    assert verdict["required"] is True
    assert scan["lines"] == 5000
    assert elapsed < 1.0, f"scan+classify took {elapsed:.3f}s on 5k lines"

    print("\n=== SANITY CHECK: definition-of-done classifier ===")
    print(f"  every requirement rule fires on its own evidence (R1 two code writes, R2 a single")
    print(f"  risk-zone write, R3 a pipeline command with NO edits at all, R4 eight writes), and")
    print(f"  every exemption holds (pure Q&A, a lone docs typo, a turn ending in a question or")
    print(f"  an AskUserQuestion, reports-only writes, a single src file).")
    print(f"  Type resolution: writes outrank commands when writes exist, commands decide when")
    print(f"  they do not; reports/** never types a task.")
    print(f"  Scanning is incremental via a byte cursor and recovers when the transcript is")
    print(f"  replaced. 5,000 lines scanned + classified in {elapsed * 1000:.0f} ms")
    print(f"  (budget: 3,000 ms wall clock for the whole hook).")
    print("  Validated.")
