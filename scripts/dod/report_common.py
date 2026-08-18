"""
report_common.py  (scripts/dod/report_common.py)
------------------------------------------------
Shared plumbing for the three Definition-of-Done report generators
(`modelling_report.py`, `data_profile.py`, `refactor_metrics.py`).

Implements the contract in docs/definition_of_done.md:

    reports/<YYYY-MM-DD>/<slug>__<TYPE>.md          one folder per day

        ---  YAML front matter: type, session_id, generated_at, baseline, generator  ---
        ## 1. Scope        (agent writes)
        ## 2. Gates        PASS/FAIL table            <- generator
        ## 3. Metrics      observed numbers, no verdict column  <- generator
        ## 4. Evidence     artifact paths             <- generator
        ## 5. Regressions, gaps and deliberate omissions   (agent writes, must be non-empty)
        ## 6. Next actions (agent writes)
        ```json dod-metrics ... ```                   <- generator, hash-checked

Design notes
  * TWO WRITERS, ONE FORMAT. The generator fills §2/§3/§4 and the metrics block; the agent
    fills §1/§5/§6. `write_report` therefore seeds the prose sections with an explicit
    TODO marker rather than leaving them blank -- a blank §5 is indistinguishable from a
    forgotten §5, and the hook rejects both, so the marker tells the agent what to do.
  * THE HASH COVERS ONLY THE BLOCK. `content_hash` is sha256 over the metrics payload with
    `content_hash` itself removed, `sort_keys=True`, separators `(",", ":")`. Prose is
    deliberately NOT hashed: the agent must be free to write §1/§5/§6 after generation.
  * STATE LIVES OUTSIDE THE REPO. `%LOCALAPPDATA%\\pea-dod\\<repo-hash>\\<session_id>\\`.
    The working tree is OneDrive-synced; per-turn writes into it cause sync churn and file
    locks. The identical formula is duplicated in `.claude/hooks/dod_lib.py` -- the hook is
    stdlib-only and must not import repo code that a refactor could break. The duplication
    is pinned by `tests/dod/test_state_dir_agrees.py`.
  * GIT IS ALLOWED HERE. A generator runs ONCE per task, so a `git` subprocess is affordable.
    A hook runs every turn and must never shell out (docs/definition_of_done.md).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Contract constants -- the hook validates against exactly these              #
# --------------------------------------------------------------------------- #
KINDS: tuple[str, ...] = ("MODELLING", "DATA", "REFACTOR")

SECTION_SCOPE = "## 1. Scope"
SECTION_GATES = "## 2. Gates"
SECTION_METRICS = "## 3. Metrics"
SECTION_EVIDENCE = "## 4. Evidence"
SECTION_REGRESSIONS = "## 5. Regressions, gaps and deliberate omissions"
SECTION_NEXT = "## 6. Next actions"

SECTIONS: tuple[str, ...] = (
    SECTION_SCOPE, SECTION_GATES, SECTION_METRICS,
    SECTION_EVIDENCE, SECTION_REGRESSIONS, SECTION_NEXT,
)

METRICS_FENCE = "```json dod-metrics"
#: The only accepted way for §5 to say "nothing regressed" (>= 30 chars of justification).
EMPTY_SECTION_5_PREFIX = "- None. Checked:"
MIN_CHECKED_CHARS = 30

#: Seeded into the prose sections so a forgotten section is visibly forgotten.
TODO_MARKER = "<!-- TODO(agent): fill this in, in your own words. -->"

APP_DIR_NAME = "pea-dod"


# --------------------------------------------------------------------------- #
# Paths and session state                                                     #
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    """The repository root (this file is `<root>/scripts/dod/report_common.py`)."""
    return Path(__file__).resolve().parents[2]


def _state_root_for(root: Path) -> Path:
    """`<LOCALAPPDATA>/pea-dod/<10-hex-of-sha256(lowercased repo path)>`.

    Lower-cased because Windows paths are case-insensitive: the hook may see
    `C:\\Users\\...` where a shell hands us `c:\\users\\...`, and those must hash alike."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TMPDIR") or "/tmp"
    digest = hashlib.sha256(str(root).replace("\\", "/").lower().encode("utf-8")).hexdigest()[:10]
    return Path(base) / APP_DIR_NAME / digest


def state_dir(session_id: str | None = None, *, root: Path | None = None) -> Path:
    """Per-session scratch dir, created on demand. Never inside the repo."""
    base = _state_root_for(root or repo_root())
    return base / session_id if session_id else base


def head_sha(root: Path | None = None) -> str:
    """Current HEAD sha, or `"unknown"`. Uses `git` -- affordable in a generator."""
    root = root or repo_root()
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=15, check=False)
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def changed_files(baseline_sha: str | None = None, root: Path | None = None) -> list[str]:
    """Repo-relative paths touched since `baseline_sha` (default: the working tree vs HEAD).

    Union of committed-since-baseline, staged, unstaged and untracked, because a task's work
    may be in any of those states when the report is written."""
    root = root or repo_root()
    paths: set[str] = set()

    def _run(args: list[str]) -> list[str]:
        try:
            out = subprocess.run(args, cwd=root, capture_output=True, text=True,
                                 timeout=30, check=False)
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        except (OSError, subprocess.SubprocessError):
            return []

    if baseline_sha and baseline_sha != "unknown":
        paths.update(_run(["git", "diff", "--name-only", f"{baseline_sha}..HEAD"]))
    paths.update(_run(["git", "diff", "--name-only", "HEAD"]))
    paths.update(_run(["git", "diff", "--name-only", "--cached"]))
    paths.update(_run(["git", "ls-files", "--others", "--exclude-standard"]))
    return sorted(p for p in paths if p)


def load_baseline(session_id: str | None = None, *, root: Path | None = None) -> dict:
    """The session's baseline (`{session_id, head_sha, started_at}`).

    Resolution order: explicit `session_id` -> `$CLAUDE_SESSION_ID` -> the most recently
    written `baseline.json` under this repo's state root -> a synthesised one from HEAD. The
    fallback matters: a generator is useful standalone, before any hook has ever run.
    """
    root = root or repo_root()
    sid = session_id or os.environ.get("CLAUDE_SESSION_ID") or None

    candidates: list[Path] = []
    if sid:
        candidates.append(state_dir(sid, root=root) / "baseline.json")
    else:
        base = _state_root_for(root)
        if base.is_dir():
            found = sorted(base.glob("*/baseline.json"), key=lambda p: p.stat().st_mtime,
                           reverse=True)
            candidates.extend(found[:1])

    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data.setdefault("session_id", path.parent.name)
        data.setdefault("head_sha", "unknown")
        return data

    return {"session_id": sid or "standalone", "head_sha": head_sha(root),
            "started_at": None, "synthesised": True}


# --------------------------------------------------------------------------- #
# Gates and metrics                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Gate:
    """One binary check. `passed=None` means "not applicable to this task".

    N/A is a first-class outcome, not a silent pass: elasticnet has no SHAP, and a task that
    touched no `data_store/` code cannot run the boundary test. Both must still be STATED."""

    id: str
    name: str
    passed: bool | None
    detail: str = ""

    @property
    def verdict(self) -> str:
        return "N/A" if self.passed is None else ("PASS" if self.passed else "FAIL")


def gate_table(gates: list[Gate]) -> str:
    """The §2 markdown table. Carries the PASS/FAIL column that §3 must never have."""
    lines = ["| Gate | Check | Verdict | Detail |", "|---|---|---|---|"]
    for g in gates:
        detail = (g.detail or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {g.id} | {g.name} | **{g.verdict}** | {detail} |")
    failed = [g.id for g in gates if g.passed is False]
    lines.append("")
    lines.append(f"**{len(failed)} FAIL** — {', '.join(failed)}. The work is **NOT done**."
                 if failed else "**All gates pass** (N/A gates are stated above, not skipped).")
    return "\n".join(lines)


def content_hash(payload: dict) -> str:
    """`sha256:<hex>` over `payload` minus `content_hash`, sorted keys, tight separators."""
    body = {k: v for k, v in payload.items() if k != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def metrics_block(payload: dict) -> str:
    """The fenced `json dod-metrics` block, `content_hash` filled in. Never edit by hand."""
    stamped = {k: v for k, v in payload.items() if k != "content_hash"}
    stamped["content_hash"] = content_hash(stamped)
    return f"{METRICS_FENCE}\n{json.dumps(stamped, indent=2, sort_keys=True, default=str)}\n```"


def metrics_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """A §3 table: observed numbers, **no verdict column** (the validator checks for that)."""
    if not rows:
        return "_No metrics collected._"
    cols = columns or sorted({k for r in rows for k in r})
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:                                    # NaN
            return "NaN"
        return f"{v:,.6g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v).replace("|", "\\|")


# --------------------------------------------------------------------------- #
# Writing                                                                     #
# --------------------------------------------------------------------------- #
def report_dir(root: Path | None = None, when: datetime | None = None) -> Path:
    """`reports/<YYYY-MM-DD>/` — ONE FOLDER PER DAY.

    A day's work usually produces several reports (and, for MODELLING, a pile of copied plots);
    grouping them by date keeps everything from one session's work in one place to read, review
    or prune together."""
    now = when or datetime.now()
    return (root or repo_root()) / "reports" / f"{now:%Y-%m-%d}"


def report_path(kind: str, slug: str, *, root: Path | None = None,
                when: datetime | None = None) -> Path:
    """`reports/<YYYY-MM-DD>/<slug>__<KIND>.md`.

    The date is the FOLDER, so it is not repeated in the filename."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in slug).strip("-") or "task"
    return report_dir(root, when) / f"{safe}__{kind}.md"


def link_prefix(path: Path, root: Path) -> str:
    """`'../' * depth`, so a link written INSIDE a report reaches the repo root.

    Computed rather than hardcoded: the report path's depth has already changed once (it used to
    be `reports/YYYY/MM/`), and a stale `../../../` silently produces dead links in every report
    rather than an error anyone would notice."""
    try:
        depth = len(path.relative_to(root).parts) - 1
    except ValueError:
        return ""
    return "../" * max(0, depth)


def write_report(kind: str, slug: str, *, generator: str, gates: list[Gate],
                 metrics_md: str, evidence_md: str, payload: dict,
                 scope_md: str | None = None, root: Path | None = None,
                 session_id: str | None = None) -> Path:
    """Write the report and return its path. Overwrites a same-day report of the same slug.

    `scope_md` is the generator's *machine-known* part of §1 (files written, commands run,
    sample scope). The agent still has to say what was ASKED -- that is seeded as a TODO."""
    root = root or repo_root()
    baseline = load_baseline(session_id, root=root)
    path = report_path(kind, slug, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    full_payload = dict(payload)
    full_payload.setdefault("type", kind)
    full_payload.setdefault("generator", generator)
    full_payload.setdefault("session_id", baseline.get("session_id"))
    full_payload.setdefault("baseline_head_sha", baseline.get("head_sha"))
    full_payload["gates"] = {g.id: g.verdict for g in gates}

    front = "\n".join([
        "---",
        f"type: {kind}",
        f"session_id: {baseline.get('session_id')}",
        f"generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"baseline: {{head_sha: {baseline.get('head_sha')}}}",
        f"generator: {generator}",
        "---",
    ])

    body = "\n\n".join([
        front,
        SECTION_SCOPE,
        (scope_md + "\n\n" if scope_md else "") + f"**What was asked:** {TODO_MARKER}",
        SECTION_GATES, gate_table(gates),
        SECTION_METRICS, metrics_md,
        SECTION_EVIDENCE, evidence_md,
        SECTION_REGRESSIONS,
        f"{TODO_MARKER}\n"
        f"- \n"
        f"<!-- At least one bullet. If genuinely nothing: "
        f"`{EMPTY_SECTION_5_PREFIX} <{MIN_CHECKED_CHARS}+ chars>` -->",
        SECTION_NEXT, f"{TODO_MARKER}\n- ",
        metrics_block(full_payload),
        "",
    ])
    path.write_text(body, encoding="utf-8")
    return path


def announce(path: Path, gates: list[Gate]) -> None:
    """Print where the report landed and what still needs a human sentence. `scripts/` may
    print -- the no-`print` rule in AGENTS.md scopes to `src/`."""
    root = repo_root()
    try:
        shown = path.relative_to(root)
    except ValueError:
        shown = path
    failed = [g.id for g in gates if g.passed is False]
    # ASCII only: this console is cp1252, where a printed "§" raises/garbles. The report FILE
    # is written as UTF-8 and does use the section sign.
    print(f"\nDoD report -> {shown}")
    print(f"  gates: {sum(g.passed is True for g in gates)} pass, "
          f"{len(failed)} fail, {sum(g.passed is None for g in gates)} n/a")
    if failed:
        print(f"  FAILING: {', '.join(failed)} -- the work is NOT done until these pass "
              f"or section 5 explains why they stand.")
    print("  NOW EDIT section 1 (what was asked), section 5 (regressions/gaps) and "
          "section 6 (next actions). Do not touch the dod-metrics block.")
