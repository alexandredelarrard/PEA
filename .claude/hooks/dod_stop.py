#!/usr/bin/env python
"""
dod_stop.py  (.claude/hooks/dod_stop.py)
----------------------------------------
`Stop` hook: the definition-of-done gate. It asks exactly ONE question --

    does a FRESH, VALID report of the required type exist?

-- and never judges the content. Judgement lives in the skills and the generators; a hook that
tried to be clever would be wrong often enough to get deleted (docs/definition_of_done.md).

    MODE      warn (default)  -> always exit 0, append the verdict to verdicts.jsonl
              enforce         -> exit 2 with a refusal that NAMES its evidence
    flip with `PEA_DOD_MODE=enforce`, or by creating `.claude/dod-enforce`

FIVE INDEPENDENT ESCAPE HATCHES, so you can never be trapped:
    1. `stop_hook_active` -- checked first; the harness is already re-prompting
    2. `.claude/dod-disabled` -- presence = off
    3. `PEA_DOD=off`
    4. a 2-attempt cap per task
    5. a blanket try/except plus a 3 s wall-clock bail -- a broken hook fails OPEN

Also serves `/dod-skip <reason>`:  `dod_stop.py --skip "<reason>"`
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dod_lib as L                                                    # noqa: E402

MAX_ATTEMPTS = 2
SKILL_FOR = {"MODELLING": "dod-modelling-report", "DATA": "dod-data-report",
             "REFACTOR": "dod-refactor-report"}


def _mode(root: Path) -> str:
    if (root / ".claude" / "dod-enforce").exists():
        return "enforce"
    return (os.environ.get("PEA_DOD_MODE") or "warn").strip().lower()


def _disabled(root: Path) -> str | None:
    """Reason the gate is off, or None."""
    if (root / ".claude" / "dod-disabled").exists():
        return "kill switch .claude/dod-disabled is present"
    if (os.environ.get("PEA_DOD") or "").strip().lower() in ("off", "0", "false", "no"):
        return "PEA_DOD=off"
    return None


def _handle_skip(reason: str) -> int:
    """`/dod-skip` writes the marker here, so the skip is recorded, not silent."""
    root = L.repo_root_from(os.environ.get("CLAUDE_PROJECT_DIR"))
    sid = str(os.environ.get("CLAUDE_SESSION_ID") or "unknown")
    sdir = L.session_dir(root, sid)
    state = L.read_json(sdir / "state.json", {}) or {}
    state["skip_reason"] = reason
    state["skipped_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L.write_json(sdir / "state.json", state)
    L.append_jsonl(sdir / "verdicts.jsonl",
                   {"at": state["skipped_at"], "event": "skip", "reason": reason})
    print(f"definition-of-done report SKIPPED for this task. Reason on record: {reason}")
    return 0


def _refusal(kind: str, verdict: dict, problems: list[str], attempt: int,
             expected: str) -> str:
    # ASCII only. This goes to stderr on a cp1252 console, where an em-dash renders as a
    # replacement char -- a refusal that looks corrupted invites being ignored.
    lines = [
        f"DEFINITION OF DONE not met - classified {kind}.",
        f"Missing: {expected}",
        f"  1. Use skill: {SKILL_FOR[kind]}",
        f'  2. "{L.SECTION_REGRESSIONS}" must be non-empty.',
    ]
    if problems:
        lines.append("  Rejected the newest candidate because:")
        lines += [f"    - {p}" for p in problems[:4]]
    lines += [
        f"Classified from: {verdict['evidence']}.",
        f"Rules fired: {'; '.join(verdict['reasons'])}.",
        "Wrong? Say so - do not fight the hook.",
        f"Attempt {attempt} of {MAX_ATTEMPTS}.  Disable: create .claude/dod-disabled",
    ]
    return "\n".join(lines)


def main() -> int:
    started = time.monotonic()

    if len(sys.argv) > 2 and sys.argv[1] == "--skip":
        return _handle_skip(" ".join(sys.argv[2:]).strip())

    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        raw = ""
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError:
        payload = {}

    # ---- hatch 1: the harness is already re-prompting --------------------- #
    if payload.get("stop_hook_active"):
        return 0

    root = L.repo_root_from(payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR"))
    session_id = str(payload.get("session_id") or "unknown")

    # ---- hatches 2 and 3 -------------------------------------------------- #
    off = _disabled(root)
    if off:
        return 0

    sdir = L.session_dir(root, session_id)
    state = L.read_json(sdir / "state.json", {}) or {}
    baseline = L.read_json(sdir / "baseline.json", {}) or {}

    # ---- incremental transcript scan -------------------------------------- #
    tpath = payload.get("transcript_path") or ""
    scan = L.scan_transcript(Path(tpath), int(state.get("cursor") or 0)) if tpath else None
    if scan is not None:
        state = L.merge_scan(state, scan, root)

    verdict = L.classify(state)
    mode = _mode(root)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    record = {
        "at": now, "mode": mode, "session_id": session_id,
        "required": verdict["required"], "kind": verdict["kind"],
        "reasons": verdict["reasons"], "evidence": verdict["evidence"],
        "n_writes": len(state.get("writes") or []),
        "n_commands": len(state.get("commands") or []),
        "truncated_scan": bool(state.get("truncated_scan")),
        "scan_lines": (scan or {}).get("lines", 0),
    }

    # ---- a recorded skip stands down the gate for this task -------------- #
    if state.get("skip_reason"):
        record.update({"outcome": "skipped", "reason": state["skip_reason"]})
        L.append_jsonl(sdir / "verdicts.jsonl", record)
        L.write_json(sdir / "state.json", state)
        return 0

    if not verdict["required"]:
        record["outcome"] = "not_required"
        L.append_jsonl(sdir / "verdicts.jsonl", record)
        L.write_json(sdir / "state.json", state)
        return 0

    if L.budget_exceeded(started):                       # hatch 5a: out of time -> fail open
        record.update({"outcome": "bailed_on_budget",
                       "elapsed_ms": round((time.monotonic() - started) * 1000, 1)})
        L.append_jsonl(sdir / "verdicts.jsonl", record)
        L.write_json(sdir / "state.json", state)
        return 0

    kind = verdict["kind"]
    since = float(state.get("accepted_after_ts")
                  or baseline.get("started_ts") or 0.0)
    found, problems = L.fresh_report(root, kind, session_id, since)

    record["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)

    if found is not None:
        # accepting advances the cursor, so a SECOND task in this session does not get to
        # reuse the first task's report
        state["accepted_after_ts"] = time.time()
        state["writes"] = []
        state["commands"] = []
        record.update({"outcome": "satisfied", "report": found.name})
        L.append_jsonl(sdir / "verdicts.jsonl", record)
        L.write_json(sdir / "state.json", state)
        return 0

    attempt = int(state.get("attempts") or 0) + 1
    state["attempts"] = attempt
    expected = f"reports/{datetime.now():%Y-%m-%d}/<slug>__{kind}.md"
    message = _refusal(kind, verdict, problems, attempt, expected)

    if attempt > MAX_ATTEMPTS:                           # hatch 4: never nag a third time
        record.update({"outcome": "attempt_cap_reached", "attempts": attempt,
                       "problems": problems})
        state["attempts"] = 0
        L.append_jsonl(sdir / "verdicts.jsonl", record)
        L.write_json(sdir / "state.json", state)
        return 0

    record.update({"outcome": "would_block" if mode != "enforce" else "blocked",
                   "attempts": attempt, "problems": problems, "expected": expected})
    L.append_jsonl(sdir / "verdicts.jsonl", record)
    L.write_json(sdir / "state.json", state)

    if mode == "enforce":
        sys.stderr.write(message + "\n")
        return 2

    # warn-only: say it once on stderr for the transcript, but never block
    sys.stderr.write("[dod: warn-only] " + message.splitlines()[0]
                     + f"  (would have blocked; {expected})\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:                                  # noqa: BLE001 - hatch 5b: fail OPEN
        raise SystemExit(0)
