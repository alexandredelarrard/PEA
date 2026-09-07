"""
dod_session_start.py  (.claude/hooks/dod_session_start.py)
------------------------------------------------------------------
`SessionStart` hook: record the session's BASELINE so everything downstream has a "before".

Two consumers need it and neither can reconstruct it later:
  * the Stop gate compares a candidate report's mtime against `started_ts`, so an old report
    of the right type cannot satisfy every future task forever;
  * the generators read `head_sha` to diff what a task changed (`report_common.load_baseline`).

⚠ NO SUBPROCESS, EVER -- `head_sha` parses `.git` directly. See `dod_lib.head_sha`. Runs under
`python -S -E` and imports nothing but the stdlib and its sibling `dod_lib`.

Always exits 0. A SessionStart hook that fails is allowed to lose the baseline (the generators
synthesise one); it is NOT allowed to stop the session from starting.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dod_lib  # noqa: E402


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    root = Path(payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or ".").resolve()
    session_id = (payload.get("session_id") or os.environ.get("CLAUDE_SESSION_ID")
                  or "unknown")

    baseline = {
        "session_id": session_id,
        "head_sha": dod_lib.head_sha(root),
        "started_ts": time.time(),
        # `started_at` is what `report_common.load_baseline` reads; `started_ts` is the epoch
        # float the Stop gate compares mtimes against. Both, because neither reader should
        # have to parse the other's format.
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "source": payload.get("source") or "unknown",
    }

    target = dod_lib.session_dir(root, session_id)
    dod_lib.write_json(target / "baseline.json", baseline)
    # A fresh session starts with a clean gate: no carried-over attempt count and no
    # carried-over `/dod-skip`, so a skip can never silently disable the gate for days.
    dod_lib.write_json(target / "gate.json", {"attempts": 0, "cursor": 0,
                                              "writes": [], "commands": []})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                       # noqa: BLE001 -- must fail OPEN
        print(f"[dod: session-start failed open] {exc}", file=sys.stderr)
        sys.exit(0)
