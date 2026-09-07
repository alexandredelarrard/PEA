"""
dod_stop.py  (.claude/hooks/dod_stop.py)
------------------------------------------------------------------
`Stop` hook: the definition-of-done gate. See docs/definition_of_done.md.

It does exactly one thing -- decide whether this turn needed a report and whether a fresh,
valid one exists -- and it is allowed to be wrong, because every path out of it is an escape
hatch. Exit 2 asks the model to write the report; exit 0 lets the turn end.

⚠ IT MUST FAIL OPEN. A gate that crashes and blocks is infinitely worse than a gate that
misses a report: the user cannot end their turn, and the only way out is to edit hooks from a
session that will not finish. Hence the blanket `except` at the bottom, the wall-clock bail,
and the four independent escape hatches.

⚠ ONE PROCESS, STDLIB ONLY, NO GIT. Runs under `python -S -E`. A spawn costs ~450 ms here, so
this hook never shells out -- see `dod_lib` for the standing budget.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dod_lib  # noqa: E402

T0 = time.perf_counter()


def _elapsed() -> float:
    return time.perf_counter() - T0


def _record(root: Path, session_id: str, **fields) -> None:
    fields.setdefault("ts", time.time())
    fields.setdefault("session_id", session_id)
    dod_lib.append_jsonl(dod_lib.session_dir(root, session_id) / "verdicts.jsonl", fields)


def _skip(argv: list[str]) -> int:
    """`/dod-skip <reason>`: stand the gate down for the next turn, loudly.

    ONE-SHOT on purpose. A skip that persisted for the session would be indistinguishable
    from the gate being uninstalled, which is the failure mode the kill switch already covers
    deliberately. The reason is echoed to stdout so it lands in the transcript -- a silent
    skip is not a skip, it is a hole."""
    reason = " ".join(argv).strip() or "(no reason given)"
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or ".").resolve()
    session_id = os.environ.get("CLAUDE_SESSION_ID") or "unknown"

    target = dod_lib.session_dir(root, session_id)
    gate = dod_lib.read_json(target / "gate.json")
    gate["skip"] = reason
    gate["skip_ts"] = time.time()
    dod_lib.write_json(target / "gate.json", gate)

    print(f"[dod: skip recorded] {reason}")
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--skip":
        return _skip(argv[1:])

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        # Garbage on stdin is the harness's problem, not the user's turn's problem.
        return 0
    if not isinstance(payload, dict):
        return 0

    root = Path(payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or ".").resolve()
    session_id = (payload.get("session_id") or os.environ.get("CLAUDE_SESSION_ID")
                  or "unknown")
    mode = (os.environ.get("PEA_DOD_MODE") or "warn").strip().lower()

    # ---- escape hatches, in the precedence docs/definition_of_done.md states ---- #
    if payload.get("stop_hook_active"):
        # Already inside a Stop-hook continuation: blocking again would loop forever.
        return 0
    if (root / ".claude" / "dod-disabled").exists():
        return 0
    if (os.environ.get("PEA_DOD") or "").strip().lower() == "off":
        return 0

    target = dod_lib.session_dir(root, session_id)
    gate = dod_lib.read_json(target / "gate.json")
    baseline = dod_lib.read_json(target / "baseline.json")

    if gate.get("skip"):
        reason = gate.pop("skip")
        gate.pop("skip_ts", None)
        dod_lib.write_json(target / "gate.json", gate)
        _record(root, session_id, outcome="skipped", mode=mode, kind=None,
                scan_lines=0, reason=reason)
        print(f"[dod: skipped] {reason}", file=sys.stderr)
        return 0

    # ---- classify ---- #
    transcript = payload.get("transcript_path")
    scan = (dod_lib.scan_transcript(transcript, int(gate.get("cursor") or 0))
            if transcript else
            {"writes": [], "commands": [], "texts": [], "asked": False,
             "lines": 0, "cursor": 0})
    state = dod_lib.merge_scan(gate, scan, root)
    verdict = dod_lib.classify(state)

    gate.update({"cursor": state.get("cursor", 0), "writes": state.get("writes", []),
                 "commands": state.get("commands", [])})
    dod_lib.write_json(target / "gate.json", gate)

    scan_lines = int(scan.get("lines") or 0)
    if not verdict["required"]:
        _record(root, session_id, outcome="not_required", mode=mode, kind=None,
                scan_lines=scan_lines, reasons=verdict["reasons"])
        return 0

    kind = verdict["kind"]
    skill = dod_lib.SKILL_FOR_KIND.get(kind, "dod-refactor-report")

    # ---- is a fresh, valid report already there? ---- #
    since = float(baseline.get("started_ts") or 0.0)
    report, why_not = dod_lib.find_report(root, kind, session_id, since)
    if report is not None:
        _record(root, session_id, outcome="satisfied", mode=mode, kind=kind,
                scan_lines=scan_lines, report=report.name)
        return 0

    # ---- required, and missing ---- #
    if _elapsed() > dod_lib.TIME_BUDGET_S:
        _record(root, session_id, outcome="timed_out", mode=mode, kind=kind,
                scan_lines=scan_lines, elapsed=_elapsed())
        return 0

    evidence = "; ".join(verdict["reasons"])

    if mode != "enforce":
        _record(root, session_id, outcome="would_block", mode=mode, kind=kind,
                scan_lines=scan_lines, reasons=verdict["reasons"])
        print(f"[dod: warn-only] this turn is classified {kind} and has no fresh report. "
              f"Classified from: {evidence}. Run the {skill} skill, or set "
              f"PEA_DOD_MODE=enforce to make this blocking.", file=sys.stderr)
        return 0

    attempts = int(gate.get("attempts") or 0) + 1
    gate["attempts"] = attempts
    dod_lib.write_json(target / "gate.json", gate)

    if attempts > dod_lib.MAX_ATTEMPTS:
        # The cap is not a formality: two refusals is the whole budget, after which the gate
        # has said its piece and must let the human get on with it.
        _record(root, session_id, outcome="attempt_cap_reached", mode=mode, kind=kind,
                scan_lines=scan_lines, attempts=attempts)
        print(f"[dod: attempt cap reached after {dod_lib.MAX_ATTEMPTS}] letting the turn "
              f"through without a {kind} report.", file=sys.stderr)
        return 0

    _record(root, session_id, outcome="blocked", mode=mode, kind=kind,
            scan_lines=scan_lines, attempts=attempts, reasons=verdict["reasons"])
    print(
        f"DEFINITION OF DONE not met -- this turn is classified {kind}.\n"
        f"  Run the `{skill}` skill and write the report before ending the turn.\n"
        f"  Classified from: {evidence}\n"
        + ("".join(f"  Rejected: {p}\n" for p in why_not[:3]) if why_not else "")
        + f"  Attempt {attempts} of {dod_lib.MAX_ATTEMPTS} -- after that this stops asking.\n"
        f"  If the classification is wrong, say so: `/dod-skip <reason>`, or "
        f"PEA_DOD=off / .claude/dod-disabled to stand it down.",
        file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:                       # noqa: BLE001 -- must fail OPEN
        print(f"[dod: gate failed open] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(0)
