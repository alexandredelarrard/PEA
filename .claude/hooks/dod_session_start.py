#!/usr/bin/env python
"""
dod_session_start.py  (.claude/hooks/dod_session_start.py)
----------------------------------------------------------
`SessionStart` hook: record the session's BASELINE so the report generators and the Stop gate
agree on "before". One process, stdlib only, no subprocess -- the head sha is read out of `.git`
(see dod_lib.head_sha_no_subprocess).

Always exits 0. A baseline is a convenience; failing to record one must never stop a session.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dod_lib as L                                                    # noqa: E402


def main() -> int:
    started = time.monotonic()
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        raw = ""
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError:
        payload = {}

    root = L.repo_root_from(payload.get("cwd") or None)
    session_id = str(payload.get("session_id") or "unknown")
    sdir = L.session_dir(root, session_id)

    baseline_path = sdir / "baseline.json"
    existing = L.read_json(baseline_path, None)
    if isinstance(existing, dict) and existing.get("head_sha"):
        # a resumed / compacted session keeps its ORIGINAL baseline: re-anchoring here would
        # silently forgive every change already made in this session
        return 0

    L.write_json(baseline_path, {
        "session_id": session_id,
        "head_sha": L.head_sha_no_subprocess(root),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_ts": time.time(),
        "source": payload.get("source"),
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:                                  # noqa: BLE001 - must never block
        raise SystemExit(0)
