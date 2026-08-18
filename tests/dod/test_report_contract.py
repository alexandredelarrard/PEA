"""
test_report_contract.py  (tests/dod/test_report_contract.py)
------------------------------------------------------------
The validator must accept a generator-produced report and reject every way one can be faked:
a missing section, an empty §5, a hand-edited metrics block, the wrong type, another session's
report.

Built by calling the REAL generator plumbing (`scripts/dod/report_common.write_report`), so the
writer and the hook's reader are pinned to each other. A hand-typed fixture would let them drift.
"""
from __future__ import annotations

import json

import pytest

from scripts.dod.report_common import Gate, link_prefix, report_path, write_report


def _fill_prose(text: str) -> str:
    """What the agent is supposed to do: replace the TODO markers with real sentences."""
    out = []
    for line in text.splitlines():
        if "TODO(agent)" in line:
            continue
        if line.strip() == "-":
            out.append("- Left the peers cache untouched; it is rebuilt by a separate step.")
            continue
        out.append(line)
    return "\n".join(out) + "\n"


@pytest.fixture
def report(tmp_path, monkeypatch):
    """A VALID, fully written REFACTOR report inside a throwaway repo root."""
    (tmp_path / "reports").mkdir()
    gates = [Gate("G1", "targeted tests green", True, "6 passed"),
             Gate("G2", "store boundary", None, "no data_store change")]
    path = write_report("REFACTOR", "unit-test", generator="scripts/dod/refactor_metrics.py@1",
                        gates=gates, metrics_md="| loc |\n|---|\n| 10 |",
                        evidence_md="- baseline: `deadbeef`",
                        payload={"scope": {"baseline_sha": "deadbeef"}, "metrics": {"loc": 10}},
                        scope_md="**Files written (1):** `src/a.py`", root=tmp_path,
                        session_id="sess-1")
    path.write_text(_fill_prose(path.read_text(encoding="utf-8")), encoding="utf-8")
    return path


def test_a_generated_and_completed_report_is_valid(dod_lib, report):
    ok, problems = dod_lib.validate_report(report, "REFACTOR", None)
    assert ok, f"a freshly generated, prose-completed report must validate: {problems}"


# --------------------------------------------------------------------------- #
# Layout: ONE FOLDER PER DAY                                                   #
# --------------------------------------------------------------------------- #
def test_reports_are_sharded_one_folder_per_day(tmp_path):
    """`reports/<YYYY-MM-DD>/<slug>__<KIND>.md` — the date is the folder, not a filename prefix."""
    from datetime import datetime
    when = datetime(2026, 8, 18, 14, 30)
    path = report_path("DATA", "fix-q4", root=tmp_path, when=when)
    assert path.relative_to(tmp_path).as_posix() == "reports/2026-08-18/fix-q4__DATA.md"


def test_a_days_reports_share_one_folder(tmp_path):
    from datetime import datetime
    when = datetime(2026, 8, 18)
    a = report_path("DATA", "one", root=tmp_path, when=when)
    b = report_path("REFACTOR", "two", root=tmp_path, when=when)
    assert a.parent == b.parent, "same-day reports must land in the same folder"


def test_slug_is_filesystem_safe(tmp_path):
    from datetime import datetime
    path = report_path("DATA", "fix/Q4 tags!", root=tmp_path, when=datetime(2026, 8, 18))
    assert path.name == "fix-Q4-tags__DATA.md"


def test_link_prefix_matches_the_actual_report_depth(tmp_path):
    """Links inside a report are relative; a wrong depth produces dead links silently, so the
    prefix is derived from the path rather than hardcoded."""
    path = report_path("DATA", "x", root=tmp_path)
    prefix = link_prefix(path, tmp_path)
    assert prefix == "../../", f"reports/<date>/file.md is 2 deep, got {prefix!r}"
    resolved = (path.parent / prefix / "docs" / "definition_of_done.md").resolve()
    assert resolved == (tmp_path / "docs" / "definition_of_done.md").resolve()


def test_unfilled_todo_markers_are_rejected(dod_lib, tmp_path):
    """The generator SEEDS the prose sections; leaving its markers in place is not 'done'."""
    (tmp_path / "reports").mkdir()
    path = write_report("REFACTOR", "raw", generator="g@1",
                        gates=[Gate("G1", "x", True)], metrics_md="m", evidence_md="e",
                        payload={}, root=tmp_path)
    ok, problems = dod_lib.validate_report(path, "REFACTOR", None)
    assert not ok
    assert any("TODO" in p for p in problems), problems


@pytest.mark.parametrize("heading", [
    "## 2. Gates", "## 3. Metrics", "## 5. Regressions, gaps and deliberate omissions",
])
def test_a_missing_section_is_rejected(dod_lib, report, heading):
    text = report.read_text(encoding="utf-8").replace(heading, "## Something else")
    report.write_text(text, encoding="utf-8")
    ok, problems = dod_lib.validate_report(report, "REFACTOR", None)
    assert not ok
    assert any(heading in p for p in problems), problems


def test_an_empty_section_5_is_rejected(dod_lib, report):
    text = report.read_text(encoding="utf-8")
    head, _, tail = text.partition(dod_lib.SECTION_REGRESSIONS)
    rest = tail.split("## 6.", 1)[1]
    report.write_text(f"{head}{dod_lib.SECTION_REGRESSIONS}\n\n## 6.{rest}", encoding="utf-8")
    ok, problems = dod_lib.validate_report(report, "REFACTOR", None)
    assert not ok
    assert any("section 5 is empty" in p for p in problems), problems


@pytest.mark.parametrize("bullet,valid", [
    ("- None. Checked: every call site of run_audit plus the two comparison modules.", True),
    ("- None. Checked: looked around.", False),
    ("- None. Checked:", False),
])
def test_the_only_accepted_nothing_needs_real_justification(dod_lib, report, bullet, valid):
    text = report.read_text(encoding="utf-8")
    head, _, tail = text.partition(dod_lib.SECTION_REGRESSIONS)
    rest = tail.split("## 6.", 1)[1]
    report.write_text(f"{head}{dod_lib.SECTION_REGRESSIONS}\n\n{bullet}\n\n## 6.{rest}",
                      encoding="utf-8")
    ok, problems = dod_lib.validate_report(report, "REFACTOR", None)
    assert ok is valid, problems


def test_a_hand_edited_metrics_block_is_rejected(dod_lib, report):
    """Numbers come from the generator. Editing the block breaks the content_hash."""
    text = report.read_text(encoding="utf-8")
    assert '"loc": 10' in text
    report.write_text(text.replace('"loc": 10', '"loc": 5'), encoding="utf-8")
    ok, problems = dod_lib.validate_report(report, "REFACTOR", None)
    assert not ok
    assert any("content_hash" in p for p in problems), problems


def test_a_removed_metrics_block_is_rejected(dod_lib, report):
    text = report.read_text(encoding="utf-8").split(dod_lib.METRICS_FENCE)[0]
    report.write_text(text, encoding="utf-8")
    ok, problems = dod_lib.validate_report(report, "REFACTOR", None)
    assert not ok
    assert any("dod-metrics" in p for p in problems), problems


def test_the_wrong_report_type_is_rejected(dod_lib, report):
    ok, problems = dod_lib.validate_report(report, "MODELLING", None)
    assert not ok
    assert any("classified as MODELLING" in p for p in problems), problems


def test_another_sessions_report_is_rejected(dod_lib, report):
    ok, problems = dod_lib.validate_report(report, "REFACTOR", "sess-2")
    assert not ok
    assert any("belongs to session" in p for p in problems), problems


def test_a_standalone_report_is_accepted_by_any_session(dod_lib, tmp_path):
    """`load_baseline` falls back to 'standalone' when no hook has ever run, and a generator
    must stay useful on its own -- so that value is not treated as a foreign session."""
    (tmp_path / "reports").mkdir()
    path = write_report("DATA", "standalone-ok", generator="g@1",
                        gates=[Gate("D1", "pk", True)], metrics_md="m", evidence_md="e",
                        payload={"session_id": "standalone"}, root=tmp_path)
    path.write_text(_fill_prose(path.read_text(encoding="utf-8")), encoding="utf-8")
    ok, problems = dod_lib.validate_report(path, "DATA", "sess-9")
    assert ok, problems


def test_hash_matches_the_generator_exactly(dod_lib, report):
    """The hook recomputes the hash; both sides must serialise identically."""
    from scripts.dod.report_common import content_hash
    text = report.read_text(encoding="utf-8")
    body = text.split(dod_lib.METRICS_FENCE, 1)[1]
    payload = json.loads(body[body.find("\n"):body.find("\n```")])
    assert dod_lib.recompute_hash(payload) == payload["content_hash"]
    assert content_hash(payload) == payload["content_hash"], \
        "generator and hook disagree on the canonical serialisation"

    print("\n=== SANITY CHECK: definition-of-done report contract ===")
    print("  a generator-produced report validates only once its prose sections are actually")
    print("  written: the seeded TODO markers, a missing section, an empty section 5, and a")
    print("  'None. Checked:' shorter than 30 characters are all rejected.")
    print("  Tampering is caught -- editing one number inside the dod-metrics block, or")
    print("  deleting the block, breaks the content_hash, and the generator and the hook are")
    print("  proven to compute that hash identically (so the check cannot drift).")
    print("  A report of the wrong type, or from another session, is refused; the 'standalone'")
    print("  session id stays usable so the generators still work with no hook installed.")
    print("  Validated.")
