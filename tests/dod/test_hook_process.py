"""
test_hook_process.py  (tests/dod/test_hook_process.py)
------------------------------------------------------
End-to-end tests of the hooks as REAL PROCESSES, launched the way `.claude/settings.json`
launches them: `python -S -E <hook>.py` with the harness JSON on stdin.

`-S -E` is the point of testing it this way. Under `-S` there are no site-packages, so a hook
that accidentally grew a pandas import would pass an in-process unit test and fail in the field.
The escape hatches are also process-level (exit codes), so they can only be checked here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.dod.conftest import HOOKS_DIR, text_line, tool_line, write_transcript

STOP_HOOK = HOOKS_DIR / "dod_stop.py"
START_HOOK = HOOKS_DIR / "dod_session_start.py"


def run_hook(hook: Path, payload: dict, *, env_extra: dict | None = None,
             args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Exactly the settings.json invocation: stdlib-only interpreter, JSON on stdin."""
    env = dict(os.environ)
    env.pop("PEA_DOD", None)
    env.pop("PEA_DOD_MODE", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-S", "-E", str(hook), *(args or [])],
        input=json.dumps(payload), capture_output=True, text=True, timeout=60, env=env)


@pytest.fixture
def session(tmp_path, monkeypatch):
    """An isolated repo root AND an isolated state dir, so tests never touch real state."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "state"))
    root = tmp_path / "repo"
    (root / ".claude").mkdir(parents=True)
    (root / "reports").mkdir()
    return root


def _payload(root: Path, transcript: Path, sid: str = "sess-e2e", **extra) -> dict:
    return {"session_id": sid, "cwd": str(root), "transcript_path": str(transcript),
            "hook_event_name": "Stop", **extra}


def _verdicts(tmp_path, sid: str = "sess-e2e") -> list[dict]:
    base = tmp_path / "state" / "pea-dod"
    files = list(base.rglob("verdicts.jsonl"))
    if not files:
        return []
    return [json.loads(ln) for ln in files[0].read_text(encoding="utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# The hooks run at all, under -S -E                                           #
# --------------------------------------------------------------------------- #
def test_session_start_records_a_baseline_without_running_git(session, tmp_path):
    proc = run_hook(START_HOOK, {"session_id": "sess-e2e", "cwd": str(session),
                                 "source": "startup"})
    assert proc.returncode == 0, proc.stderr
    found = list((tmp_path / "state" / "pea-dod").rglob("baseline.json"))
    assert found, "SessionStart must write baseline.json"
    data = json.loads(found[0].read_text(encoding="utf-8"))
    assert data["session_id"] == "sess-e2e"
    assert "head_sha" in data and "started_ts" in data


def test_head_sha_is_read_from_dot_git_not_from_git(session, tmp_path):
    """No subprocess: `.git` is parsed directly. A fabricated .git proves it never ran git."""
    (session / ".git" / "refs" / "heads").mkdir(parents=True)
    (session / ".git" / "HEAD").write_text("ref: refs/heads/dev\n", encoding="utf-8")
    (session / ".git" / "refs" / "heads" / "dev").write_text("a" * 40 + "\n", encoding="utf-8")
    run_hook(START_HOOK, {"session_id": "sess-e2e", "cwd": str(session)})
    data = json.loads(list((tmp_path / "state" / "pea-dod").rglob("baseline.json"))[0]
                      .read_text(encoding="utf-8"))
    assert data["head_sha"] == "a" * 40


def test_packed_refs_are_resolved(session, tmp_path):
    (session / ".git").mkdir()
    (session / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (session / ".git" / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted \n"
        f"{'b' * 40} refs/heads/main\n", encoding="utf-8")
    run_hook(START_HOOK, {"session_id": "sess-e2e", "cwd": str(session)})
    data = json.loads(list((tmp_path / "state" / "pea-dod").rglob("baseline.json"))[0]
                      .read_text(encoding="utf-8"))
    assert data["head_sha"] == "b" * 40


# --------------------------------------------------------------------------- #
# Warn-only rollout                                                           #
# --------------------------------------------------------------------------- #
def test_warn_only_never_blocks_but_records_the_verdict(session, tmp_path):
    t = write_transcript(tmp_path, [tool_line("Edit", file_path="src/a.py"),
                                    tool_line("Edit", file_path="src/b.py")])
    proc = run_hook(STOP_HOOK, _payload(session, t))
    assert proc.returncode == 0, "warn mode must never block"
    assert "[dod: warn-only]" in proc.stderr
    rec = _verdicts(tmp_path)[-1]
    assert rec["outcome"] == "would_block"
    assert rec["kind"] == "REFACTOR"
    assert rec["mode"] == "warn"


def test_a_question_turn_is_never_blocked(session, tmp_path):
    t = write_transcript(tmp_path, [tool_line("Edit", file_path="src/a.py"),
                                    tool_line("Edit", file_path="src/b.py"),
                                    text_line("Which sleeve should I wire first?")])
    proc = run_hook(STOP_HOOK, _payload(session, t),
                    env_extra={"PEA_DOD_MODE": "enforce"})
    assert proc.returncode == 0, "a turn ending in a question must never be blocked"
    assert _verdicts(tmp_path)[-1]["outcome"] == "not_required"


# --------------------------------------------------------------------------- #
# Enforcing                                                                   #
# --------------------------------------------------------------------------- #
def test_enforce_blocks_with_evidence_then_caps_attempts(session, tmp_path):
    t = write_transcript(tmp_path, [tool_line("Edit", file_path="src/a.py"),
                                    tool_line("Edit", file_path="src/b.py")])
    payload = _payload(session, t)
    env = {"PEA_DOD_MODE": "enforce"}

    first = run_hook(STOP_HOOK, payload, env_extra=env)
    assert first.returncode == 2, first.stderr
    assert "DEFINITION OF DONE not met" in first.stderr
    assert "classified REFACTOR" in first.stderr
    assert "dod-refactor-report" in first.stderr
    assert "Classified from:" in first.stderr           # the evidence line
    assert "Attempt 1 of 2" in first.stderr

    second = run_hook(STOP_HOOK, payload, env_extra=env)
    assert second.returncode == 2
    assert "Attempt 2 of 2" in second.stderr

    third = run_hook(STOP_HOOK, payload, env_extra=env)
    assert third.returncode == 0, "the attempt cap must let the turn through"
    assert _verdicts(tmp_path)[-1]["outcome"] == "attempt_cap_reached"


def test_a_valid_report_satisfies_the_gate(session, tmp_path):
    from scripts.dod.report_common import Gate, write_report
    from tests.dod.test_report_contract import _fill_prose

    run_hook(START_HOOK, {"session_id": "sess-e2e", "cwd": str(session)})
    path = write_report("REFACTOR", "e2e", generator="g@1",
                        gates=[Gate("G1", "tests", True, "ok")], metrics_md="m",
                        evidence_md="e", payload={"session_id": "sess-e2e"}, root=session)
    path.write_text(_fill_prose(path.read_text(encoding="utf-8")), encoding="utf-8")
    os.utime(path, (time.time() + 5, time.time() + 5))   # unambiguously after the baseline

    t = write_transcript(tmp_path, [tool_line("Edit", file_path="src/a.py"),
                                    tool_line("Edit", file_path="src/b.py")])
    proc = run_hook(STOP_HOOK, _payload(session, t), env_extra={"PEA_DOD_MODE": "enforce"})
    assert proc.returncode == 0, proc.stderr
    rec = _verdicts(tmp_path)[-1]
    assert rec["outcome"] == "satisfied"
    assert rec["report"] == path.name


# --------------------------------------------------------------------------- #
# Escape hatches                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,kill_file,env_extra,stop_active", [
    ("kill switch file", True, {"PEA_DOD_MODE": "enforce"}, False),
    ("PEA_DOD=off", False, {"PEA_DOD_MODE": "enforce", "PEA_DOD": "off"}, False),
    ("stop_hook_active", False, {"PEA_DOD_MODE": "enforce"}, True),
])
def test_every_escape_hatch_exits_zero(session, tmp_path, label, kill_file, env_extra,
                                       stop_active):
    if kill_file:
        (session / ".claude" / "dod-disabled").write_text("", encoding="utf-8")
    t = write_transcript(tmp_path, [tool_line("Edit", file_path="src/a.py"),
                                    tool_line("Edit", file_path="src/b.py")])
    payload = _payload(session, t)
    if stop_active:
        payload["stop_hook_active"] = True
    proc = run_hook(STOP_HOOK, payload, env_extra=env_extra)
    assert proc.returncode == 0, f"{label} must exit 0 (got {proc.returncode}): {proc.stderr}"


def test_a_broken_payload_fails_open(session):
    """Garbage on stdin must not block the user."""
    proc = subprocess.run([sys.executable, "-S", "-E", str(STOP_HOOK)],
                          input="not json at all", capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0


def test_dod_skip_records_a_loud_reason(session, tmp_path):
    reason = "one-line docstring fix, no behaviour change"
    proc = run_hook(STOP_HOOK, {}, args=["--skip", reason],
                    env_extra={"CLAUDE_PROJECT_DIR": str(session),
                               "CLAUDE_SESSION_ID": "sess-e2e"})
    assert proc.returncode == 0
    assert reason in proc.stdout

    t = write_transcript(tmp_path, [tool_line("Edit", file_path="src/a.py"),
                                    tool_line("Edit", file_path="src/b.py")])
    after = run_hook(STOP_HOOK, _payload(session, t),
                     env_extra={"PEA_DOD_MODE": "enforce"})
    assert after.returncode == 0, "a recorded skip must stand the gate down"
    assert _verdicts(tmp_path)[-1]["outcome"] == "skipped"


def test_hook_is_fast_enough_on_a_large_transcript(session, tmp_path):
    """One process per turn; the whole thing must be comfortably under a second."""
    lines = [tool_line("Edit", file_path=f"src/m{i % 40}.py") if i % 4
             else tool_line("Bash", command=f"echo {i}") for i in range(5000)]
    t = write_transcript(tmp_path, lines)
    t0 = time.perf_counter()
    proc = run_hook(STOP_HOOK, _payload(session, t))
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0
    assert elapsed < 5.0, f"hook process took {elapsed:.2f}s"

    rec = _verdicts(tmp_path)[-1]
    print("\n=== SANITY CHECK: definition-of-done hooks as real processes ===")
    print("  Both hooks run under `python -S -E` (no site-packages), proving they are")
    print("  stdlib-only, and SessionStart resolves HEAD by parsing .git directly -- loose ref")
    print("  AND packed-refs -- so no turn ever pays for a git subprocess.")
    print("  Warn-only mode records `would_block` and exits 0; enforce mode exits 2 with a")
    print("  refusal naming the classification, the skill to use and the evidence, then stops")
    print("  nagging after 2 attempts. A valid fresh report flips the outcome to `satisfied`.")
    print("  All five escape hatches exit 0: stop_hook_active, .claude/dod-disabled,")
    print("  PEA_DOD=off, the attempt cap, and a garbage payload (fails OPEN).")
    print(f"  Whole hook process on a 5,000-line transcript: {elapsed * 1000:.0f} ms "
          f"({rec['scan_lines']} lines scanned).")
    print("  Validated.")
