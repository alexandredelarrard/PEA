"""
dod_lib.py  (.claude/hooks/dod_lib.py)
--------------------------------------
Everything the definition-of-done hooks need: session state, an incremental transcript scan,
classification, and report validation. See docs/definition_of_done.md for the contract.

HARD CONSTRAINTS (docs/definition_of_done.md, "Two standing budgets")
  * STDLIB ONLY, and run under `python -S -E`. No pandas, no repo imports -- a hook that can be
    broken by a refactor of `src/` is a hook that will be deleted.
  * NEVER SHELL OUT. A process spawn costs ~450 ms here (Git Bash + Defender + a OneDrive-synced
    tree), so one `git`/`grep`/`wc` pipeline would tax every turn by seconds. The head sha is
    read straight out of `.git/`; the "what did this session do" question is answered from the
    TRANSCRIPT, not from a diff.
  * ONE PROCESS PER TURN. This module is imported by exactly one script per hook event.

Why the transcript rather than git:
  The Stop hook is handed `transcript_path`, a JSONL file containing every `Edit`/`Write`
  `file_path` and every `Bash` `command` of the session -- which is precisely "what this session
  did". A git diff would instead report whatever was already dirty when the session started,
  and would fire on a pre-existing mess the agent never touched. A byte cursor makes each turn's
  rescan incremental, so a long session does not re-parse megabytes every turn.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

APP_DIR_NAME = "pea-dod"
KINDS = ("MODELLING", "DATA", "REFACTOR")

#: Cold-parse cap. A resumed session can have a huge transcript; only the tail matters for
#: "what happened recently", and the flag tells the reader the scan was partial.
TAIL_CAP_BYTES = 2 * 1024 * 1024
#: A broken hook must never block the user. Checked between phases.
WALL_CLOCK_BUDGET_S = 3.0

WRITE_TOOLS = frozenset({"Edit", "Write", "NotebookEdit", "MultiEdit"})
CODE_ROOTS = ("src/", "tests/", "configs/", "sql/", "scripts/")

#: docs/coding_standard.md "Risk zones". One write here is enough to require a report.
RISK_ZONES = (
    "src/context.py",
    "src/utils/step.py",
    "src/constants/",
    "src/data_store/",
    "sql/schema.sql",
    "configs/",
    "tests/data_aggregate/aggregate_fingerprint_baseline.json",
)

#: R3 -- "ran a pipeline, changed no code", which path inference alone cannot see.
PIPELINE_RE = re.compile(
    r"-m\s+src\s+(data_extract|data_aggregate|data_peers|modelling|portfolio)\b|(?<![\w/])main\.py\b")

MODELLING_PREFIXES = ("src/modelling/", "src/strategies/", "src/portfolio/",
                      "configs/models", "configs/modellling.yml", "configs/strategy")
DATA_PREFIXES = ("src/data_extract/", "src/data_aggregate/", "src/data_peers/",
                 "src/data_store/", "src/validate/", "sql/",
                 "configs/build_cube.yml", "configs/peers.yml")
MODELLING_CMD_RE = re.compile(r"-m\s+src\s+(modelling|portfolio)\b")
DATA_CMD_RE = re.compile(r"-m\s+src\s+(data_extract|data_aggregate|data_peers)\b")

# Report contract (mirrors scripts/dod/report_common.py; pinned by tests/dod/)
SECTIONS = (
    "## 1. Scope", "## 2. Gates", "## 3. Metrics", "## 4. Evidence",
    "## 5. Regressions, gaps and deliberate omissions", "## 6. Next actions",
)
SECTION_REGRESSIONS = SECTIONS[4]
METRICS_FENCE = "```json dod-metrics"
EMPTY_SECTION_5_PREFIX = "- None. Checked:"
MIN_CHECKED_CHARS = 30
TODO_MARKER = "TODO(agent)"


# --------------------------------------------------------------------------- #
# State                                                                       #
# --------------------------------------------------------------------------- #
def repo_root_from(project_dir: str | None) -> Path:
    """`$CLAUDE_PROJECT_DIR` when the harness supplies it, else this file's grandparent."""
    if project_dir:
        return Path(project_dir).resolve()
    return Path(__file__).resolve().parents[2]


def state_root(root: Path) -> Path:
    """`<LOCALAPPDATA>/pea-dod/<10-hex>`. NOT inside the repo: the tree is OneDrive-synced and
    per-turn writes there cause sync churn and file locks. Duplicated in
    `scripts/dod/report_common.py::_state_root_for`; pinned by tests/dod/."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TMPDIR") or "/tmp"
    digest = hashlib.sha256(str(root).replace("\\", "/").lower().encode("utf-8")).hexdigest()[:10]
    return Path(base) / APP_DIR_NAME / digest


def session_dir(root: Path, session_id: str) -> Path:
    d = state_root(root) / (session_id or "unknown")
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path: Path, default: object = None) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data: object) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass                                  # state is a convenience, never a hard dependency


def append_jsonl(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def head_sha_no_subprocess(root: Path) -> str:
    """HEAD's sha by READING `.git`, never by running git.

    Handles the three shapes: a detached sha in `.git/HEAD`, a `ref:` into `.git/refs/`, and a
    ref that only exists in `.git/packed-refs`. Also follows a `.git` FILE (worktree)."""
    try:
        gitdir = root / ".git"
        if gitdir.is_file():                                # worktree: "gitdir: <path>"
            txt = gitdir.read_text(encoding="utf-8").strip()
            if txt.startswith("gitdir:"):
                gitdir = Path(txt.split(":", 1)[1].strip())
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        loose = gitdir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
        packed = gitdir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except (OSError, ValueError, IndexError):
        pass
    return "unknown"


# --------------------------------------------------------------------------- #
# Transcript scan                                                             #
# --------------------------------------------------------------------------- #
def _walk_tool_uses(obj: object, out: list[dict]) -> None:
    """Collect every `{"name": ..., "input": {...}}` block, at any depth.

    Deliberately schema-agnostic: the transcript's envelope has changed shape across releases,
    but a tool call is always a dict carrying a tool name and its input."""
    if isinstance(obj, dict):
        name = obj.get("name")
        inp = obj.get("input")
        if isinstance(name, str) and isinstance(inp, dict):
            out.append({"name": name, "input": inp})
        for v in obj.values():
            _walk_tool_uses(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_tool_uses(v, out)


def _walk_text(obj: object, out: list[str]) -> None:
    if isinstance(obj, dict):
        if obj.get("type") == "text" and isinstance(obj.get("text"), str):
            out.append(obj["text"])
        for v in obj.values():
            _walk_text(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_text(v, out)


def scan_transcript(path: Path, cursor: int = 0) -> dict:
    """Read `path` from byte `cursor` and summarise what the session did.

    Returns `{writes, commands, tools, last_text, cursor, truncated, lines}`. `writes` are
    repo-relative-ish POSIX paths; `commands` are raw Bash command strings.
    """
    result = {"writes": [], "commands": [], "tools": [], "last_text": "",
              "cursor": cursor, "truncated": False, "lines": 0}
    try:
        size = path.stat().st_size
    except OSError:
        return result

    start = cursor if 0 <= cursor <= size else 0
    if start == 0 and size > TAIL_CAP_BYTES:
        start = size - TAIL_CAP_BYTES
        result["truncated"] = True

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(start)
            if result["truncated"]:
                fh.readline()                              # drop the partial first line
            data = fh.read()
    except OSError:
        return result
    result["cursor"] = size

    tool_uses: list[dict] = []
    texts: list[str] = []
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        result["lines"] += 1
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        _walk_tool_uses(obj, tool_uses)
        _walk_text(obj, texts)

    for tu in tool_uses:
        name, inp = tu["name"], tu["input"]
        result["tools"].append(name)
        if name in WRITE_TOOLS:
            fp = inp.get("file_path") or inp.get("notebook_path") or ""
            if isinstance(fp, str) and fp:
                result["writes"].append(fp)
        elif name == "Bash":
            cmd = inp.get("command")
            if isinstance(cmd, str) and cmd:
                result["commands"].append(cmd)
    if texts:
        result["last_text"] = texts[-1]
    return result


def relativise(paths: list[str], root: Path) -> list[str]:
    """Absolute or backslashed paths -> repo-relative POSIX. Unknown paths pass through."""
    rr = str(root).replace("\\", "/").rstrip("/").lower()
    out = []
    for p in paths:
        q = str(p).replace("\\", "/")
        if q.lower().startswith(rr + "/"):
            q = q[len(rr) + 1:]
        out.append(q.lstrip("./"))
    return out


def merge_scan(state: dict, scan: dict, root: Path) -> dict:
    """Accumulate this turn's scan into the session's running totals (deduped, order kept)."""
    writes = list(dict.fromkeys((state.get("writes") or [])
                               + relativise(scan["writes"], root)))
    commands = (state.get("commands") or []) + scan["commands"]
    state.update({
        "writes": writes,
        "commands": commands[-400:],                        # bounded; only patterns matter
        "cursor": scan["cursor"],
        "truncated_scan": bool(state.get("truncated_scan")) or scan["truncated"],
        "last_text": scan["last_text"] or state.get("last_text", ""),
        "tools_seen": list(dict.fromkeys((state.get("tools_seen") or []) + scan["tools"])),
    })
    return state


# --------------------------------------------------------------------------- #
# Classification                                                              #
# --------------------------------------------------------------------------- #
def _is_code(path: str) -> bool:
    return path.startswith(CODE_ROOTS)


def _is_risk(path: str) -> bool:
    return path.startswith(RISK_ZONES)


def _is_docs_only(path: str) -> bool:
    return (path.startswith(("docs/", "specs/")) or path.endswith(".md")
            or path.startswith(".claude/"))


def ends_in_question(text: str) -> bool:
    """True when the agent's last message ends by asking something.

    Never block a turn where the agent is asking the user a question -- that is the single most
    annoying failure this gate could have. Trailing markdown/list punctuation is stripped first.
    """
    if not text:
        return False
    tail = text.rstrip()
    tail = re.sub(r"[\s*_`>\-\)\]]+$", "", tail)
    return tail.endswith("?")


def classify(state: dict) -> dict:
    """Decide `required` and `kind` from the accumulated session facts.

    Returns `{required, kind, reasons, evidence}`; `kind` is None when not required.
    """
    writes: list[str] = list(state.get("writes") or [])
    commands: list[str] = list(state.get("commands") or [])

    code_writes = [w for w in writes if _is_code(w)]
    risk_writes = [w for w in writes if _is_risk(w)]
    report_writes = [w for w in writes if w.startswith("reports/")]
    pipeline_cmds = [c for c in commands if PIPELINE_RE.search(c)]
    non_report_writes = [w for w in writes if not w.startswith("reports/")]

    reasons: list[str] = []
    # -- exemptions, checked first ----------------------------------------- #
    if ends_in_question(state.get("last_text", "")) or "AskUserQuestion" in (
            state.get("tools_seen") or []):
        return {"required": False, "kind": None,
                "reasons": ["N-question: the turn ends in a question to the user"],
                "evidence": _evidence(code_writes, pipeline_cmds, writes)}
    # N-reports before N-idle: both mean "not required", but when reports/** was written the
    # SPECIFIC reason is the useful one in verdicts.jsonl while thresholds are being tuned.
    if report_writes and not non_report_writes:
        return {"required": False, "kind": None,
                "reasons": ["N-reports: only reports/** was written"],
                "evidence": _evidence(code_writes, pipeline_cmds, writes)}
    if not non_report_writes and not pipeline_cmds:
        return {"required": False, "kind": None,
                "reasons": ["N-idle: no code write and no pipeline command"],
                "evidence": _evidence(code_writes, pipeline_cmds, writes)}
    if (not pipeline_cmds and not code_writes
            and all(_is_docs_only(w) for w in non_report_writes)
            and len(non_report_writes) < 2):
        return {"required": False, "kind": None,
                "reasons": ["N-docs: a single docs/markdown file and nothing else"],
                "evidence": _evidence(code_writes, pipeline_cmds, writes)}

    # -- requirement rules -------------------------------------------------- #
    if len(set(code_writes)) >= 2:
        reasons.append(f"R1: {len(set(code_writes))} distinct code writes")
    if risk_writes:
        reasons.append(f"R2: risk zone written ({', '.join(sorted(set(risk_writes))[:3])})")
    if pipeline_cmds:
        reasons.append(f"R3: ran a pipeline command ({pipeline_cmds[-1][:70]})")
    if len(set(writes)) >= 8:
        reasons.append(f"R4: {len(set(writes))} total writes")

    if not reasons:
        return {"required": False, "kind": None,
                "reasons": ["N-small: below every requirement threshold"],
                "evidence": _evidence(code_writes, pipeline_cmds, writes)}

    return {"required": True, "kind": resolve_kind(writes, commands), "reasons": reasons,
            "evidence": _evidence(code_writes, pipeline_cmds, writes)}


def resolve_kind(writes: list[str], commands: list[str]) -> str:
    """WRITES outrank COMMANDS when writes exist; commands decide when they do not.

    A session that edits `src/modelling/` and also happens to run a data command is modelling
    work -- the edit is the change, the command was probably just input for it."""
    relevant = [w for w in writes if not w.startswith("reports/")]
    if relevant:
        if any(w.startswith(MODELLING_PREFIXES) for w in relevant):
            return "MODELLING"
        if any(w.startswith(DATA_PREFIXES) for w in relevant):
            return "DATA"
        return "REFACTOR"
    for c in reversed(commands):
        if MODELLING_CMD_RE.search(c):
            return "MODELLING"
        if DATA_CMD_RE.search(c):
            return "DATA"
    return "REFACTOR"


def _evidence(code_writes: list[str], pipeline_cmds: list[str], writes: list[str]) -> str:
    """One human sentence naming what the classification was based on, for the refusal text."""
    bits = []
    if pipeline_cmds:
        bits.append(f"ran `{pipeline_cmds[-1][:60]}`")
    shown = code_writes or writes
    if shown:
        head = ", ".join(Path(p).name for p in list(dict.fromkeys(shown))[:3])
        extra = len(set(shown)) - 3
        bits.append(f"wrote {head}" + (f" (+{extra})" if extra > 0 else ""))
    return "; ".join(bits) or "no writes, no commands"


# --------------------------------------------------------------------------- #
# Report validation                                                           #
# --------------------------------------------------------------------------- #
def _front_matter(text: str) -> dict:
    out: dict = {}
    if not text.startswith("---"):
        return out
    end = text.find("\n---", 3)
    if end == -1:
        return out
    for line in text[3:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _metrics_payload(text: str) -> tuple[dict | None, str]:
    i = text.find(METRICS_FENCE)
    if i == -1:
        return None, "no ```json dod-metrics block"
    body_start = text.find("\n", i)
    end = text.find("\n```", body_start)
    if body_start == -1 or end == -1:
        return None, "the dod-metrics block is not closed"
    try:
        payload = json.loads(text[body_start:end])
    except ValueError as e:
        return None, f"the dod-metrics block is not valid JSON ({e.__class__.__name__})"
    return (payload, "") if isinstance(payload, dict) else (None, "dod-metrics is not an object")


def recompute_hash(payload: dict) -> str:
    """Must match `scripts/dod/report_common.content_hash` exactly."""
    body = {k: v for k, v in payload.items() if k != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _section_body(text: str, heading: str) -> str:
    i = text.find(heading)
    if i == -1:
        return ""
    start = i + len(heading)
    nxt = text.find("\n## ", start)
    fence = text.find("\n" + METRICS_FENCE, start)
    ends = [e for e in (nxt, fence) if e != -1]
    return text[start:min(ends)] if ends else text[start:]


def _section_5_ok(body: str) -> tuple[bool, str]:
    """>=1 real bullet, or the explicit `- None. Checked: <30+ chars>` form."""
    lines = [ln.strip() for ln in body.splitlines()]
    kept = [ln for ln in lines
            if ln and not ln.startswith("<!--") and TODO_MARKER not in ln]
    bullets = [ln for ln in kept if ln.startswith(("-", "*")) and len(ln.lstrip("-* ")) > 0]
    if not bullets:
        return False, ("section 5 is empty. Give at least one bullet, or "
                       f"`{EMPTY_SECTION_5_PREFIX} <{MIN_CHECKED_CHARS}+ chars>`")
    for b in bullets:
        if b.lower().startswith(EMPTY_SECTION_5_PREFIX.lower()[:15]):
            checked = b.split(":", 1)[1].strip() if ":" in b else ""
            if len(checked) < MIN_CHECKED_CHARS:
                return False, (f"'None. Checked:' needs >= {MIN_CHECKED_CHARS} characters "
                               f"describing what you looked at (got {len(checked)})")
    return True, ""


def validate_report(path: Path, expected_kind: str | None = None,
                    session_id: str | None = None) -> tuple[bool, list[str]]:
    """Structural validation only -- never a judgement on the prose."""
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, [f"cannot read {path.name}: {e.__class__.__name__}"]

    front = _front_matter(text)
    kind = (front.get("type") or "").strip()
    if kind not in KINDS:
        problems.append(f"front matter `type` must be one of {'/'.join(KINDS)}, got {kind!r}")
    elif expected_kind and kind != expected_kind:
        problems.append(f"report type is {kind}, but this task classified as {expected_kind}")

    for heading in SECTIONS:
        if heading not in text:
            problems.append(f"missing section: '{heading}'")

    body5 = _section_body(text, SECTION_REGRESSIONS)
    ok5, why5 = _section_5_ok(body5)
    if not ok5:
        problems.append(why5)

    if TODO_MARKER in text:
        problems.append("the generator's TODO markers are still in the file -- sections 1, 5 "
                        "and 6 have not been written")

    payload, why = _metrics_payload(text)
    if payload is None:
        problems.append(why)
    else:
        claimed = payload.get("content_hash")
        actual = recompute_hash(payload)
        if not claimed:
            problems.append("the dod-metrics block has no content_hash")
        elif claimed != actual:
            problems.append("the dod-metrics block was edited by hand (content_hash mismatch) "
                            "-- re-run the generator instead")
        if session_id and payload.get("session_id") not in (None, "standalone", session_id):
            problems.append(f"the report belongs to session {payload.get('session_id')}, "
                            f"not this one -- re-run the generator")
    return (not problems), problems


def find_reports(root: Path, kind: str | None = None) -> list[Path]:
    """Every report under `reports/`, newest mtime first."""
    base = root / "reports"
    if not base.is_dir():
        return []
    pat = f"*__{kind}.md" if kind else "*__*.md"
    try:
        found = [p for p in base.rglob(pat) if p.is_file()]
    except OSError:
        return []
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def fresh_report(root: Path, kind: str, session_id: str, since_ts: float) -> tuple[Path | None, list[str]]:
    """The newest VALID report of `kind` written after `since_ts`, plus why others were rejected."""
    problems: list[str] = []
    for path in find_reports(root, kind):
        try:
            if path.stat().st_mtime < since_ts:
                continue
        except OSError:
            continue
        ok, why = validate_report(path, kind, session_id)
        if ok:
            return path, []
        problems = [f"{path.name}: {w}" for w in why]
        break                                    # only the newest candidate is worth reporting
    return None, problems


def budget_exceeded(started: float) -> bool:
    return (time.monotonic() - started) > WALL_CLOCK_BUDGET_S
