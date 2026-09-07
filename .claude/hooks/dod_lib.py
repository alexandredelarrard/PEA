"""
dod_lib.py  (.claude/hooks/dod_lib.py)
------------------------------------------------------------------
Shared library for the definition-of-done hooks. See docs/definition_of_done.md.

⚠ STDLIB ONLY, AND DELIBERATELY NOT AN IMPORTABLE PACKAGE. The hooks run under
`python -S -E` (no site-packages, no PYTHON* env vars), so this file must never import repo
code: a refactor in `src/` could otherwise break the hook that runs on every single turn, and
under `-S` a stray `import pandas` fails at load time rather than in a test. The tests load it
by path (`tests/dod/conftest.py`) exactly the way the entry points do.

⚠ THE CONTRACT CONSTANTS ARE DUPLICATED FROM `scripts/dod/report_common.py` on purpose, for
the same reason. The duplication is fine; a DIVERGENCE is not, and
`tests/dod/test_agents_md_budget.py::test_the_gate_and_the_generator_agree_on_the_contract`
fails the moment the two drift.

⚠ NEVER SHELL OUT. A hook is exactly one process. A spawn costs ~450 ms on this machine
(Git Bash + Defender + a OneDrive-synced tree), so `git rev-parse` in a Stop hook would cost
seconds of every turn. `head_sha` therefore PARSES `.git` directly -- loose ref and
packed-refs -- and `docs/definition_of_done.md` records that as a standing budget.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Contract constants -- must equal scripts/dod/report_common.py               #
# --------------------------------------------------------------------------- #
KINDS = ("MODELLING", "DATA", "REFACTOR")

SECTION_SCOPE = "## 1. Scope"
SECTION_GATES = "## 2. Gates"
SECTION_METRICS = "## 3. Metrics"
SECTION_EVIDENCE = "## 4. Evidence"
SECTION_REGRESSIONS = "## 5. Regressions, gaps and deliberate omissions"
SECTION_NEXT = "## 6. Next actions"

SECTIONS = (SECTION_SCOPE, SECTION_GATES, SECTION_METRICS,
            SECTION_EVIDENCE, SECTION_REGRESSIONS, SECTION_NEXT)

METRICS_FENCE = "```json dod-metrics"
EMPTY_SECTION_5_PREFIX = "- None. Checked:"
MIN_CHECKED_CHARS = 30
TODO_MARKER = "TODO(agent)"

APP_DIR_NAME = "pea-dod"

#: Which skill the refusal names, per classification.
SKILL_FOR_KIND = {"MODELLING": "dod-modelling-report",
                  "DATA": "dod-data-report",
                  "REFACTOR": "dod-refactor-report"}

#: How many times one session may be blocked before the gate stands down. The gate must never
#: trap the user (docs/definition_of_done.md); two refusals is enough to be heard.
MAX_ATTEMPTS = 2

#: Wall-clock budget for the whole hook. Past this it gives up and fails OPEN.
TIME_BUDGET_S = 3.0

# --------------------------------------------------------------------------- #
# Classification zones                                                        #
# --------------------------------------------------------------------------- #
#: A write here is "code" for R1. Mirrors the trigger list in docs/definition_of_done.md.
CODE_PREFIXES = ("src/", "tests/", "configs/", "sql/", "scripts/")

#: A SINGLE write here requires a report (R2): these are the places where one line changes
#: everything downstream. `data_store/` owns every read; `context.py` is constructed by every
#: step; `configs/` decides what the pipeline computes; the fingerprint baseline is the file
#: that makes a numeric regression invisible once it is blessed.
RISK_ZONES = ("src/context.py", "src/data_store/", "configs/", "sql/",
              "tests/data_aggregate/aggregate_fingerprint_baseline.json")

#: Writes here never TYPE a task and never trigger one on their own -- a report is not work.
REPORT_PREFIXES = ("reports/",)

DATA_ZONES = ("src/data_extract/", "src/data_aggregate/", "src/data_store/", "src/data_peers/",
              "src/validate/", "sql/")
MODELLING_ZONES = ("src/modelling/", "src/strategies/", "src/portfolio/")

DATA_COMMANDS = ("data_extract", "data_aggregate", "data_peers", "validate")
MODELLING_COMMANDS = ("modelling", "portfolio", "strategies")

WRITE_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
QUESTION_TOOLS = ("AskUserQuestion",)

#: `-m src <group>` or a bare `main.py` -- the two ways this repo starts a pipeline.
_PIPELINE_RE = re.compile(r"-m\s+src\b|(?<![\w/])main\.py\b")
_MODULE_RE = re.compile(r"-m\s+src\s+([a-z_]+)")

#: Trailing decoration a model puts after a question mark: bold/italic/code markers, a closing
#: bracket, a stray quote. Stripped before the `?` test so "Should I proceed?**" still counts.
_TRAILING_DECOR = "*_`)]}\"' \t\r\n>"


# --------------------------------------------------------------------------- #
# Paths and state                                                             #
# --------------------------------------------------------------------------- #
def state_root(root: Path) -> Path:
    """`<LOCALAPPDATA>/pea-dod/<10-hex-of-sha256(lowercased repo path)>`.

    ⚠ Byte-for-byte the formula in `report_common._state_root_for`. The hook WRITES the
    baseline the generators READ; if the two formulas disagree every report silently falls
    back to a synthesised baseline and nobody notices. Pinned by
    `test_the_state_dir_formula_agrees`.

    Lower-cased because Windows paths are case-insensitive: the hook may see `C:\\Users\\...`
    where a shell hands us `c:\\users\\...`, and those must hash alike."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TMPDIR") or "/tmp"
    digest = hashlib.sha256(str(root).replace("\\", "/").lower().encode("utf-8")).hexdigest()[:10]
    return Path(base) / APP_DIR_NAME / digest


def session_dir(root: Path, session_id: str) -> Path:
    return state_root(root) / (session_id or "unknown")


def read_json(path: Path) -> dict:
    """`{}` on anything unreadable -- a hook must never die on its own state file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass


def append_jsonl(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def head_sha(root: Path) -> str:
    """HEAD's sha by PARSING `.git`, never by running git. `"unknown"` when it cannot be read.

    Both ref shapes matter: a fresh clone keeps `refs/heads/<branch>` as a loose file, but
    `git gc` moves it into `packed-refs` and then the loose file is simply absent -- a reader
    that only knows the first shape reports "unknown" on any long-lived repo."""
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"

    if not head.startswith("ref:"):
        return head.split()[0] if head else "unknown"       # detached HEAD holds the sha

    ref = head.split(":", 1)[1].strip()
    try:
        return (root / ".git" / ref).read_text(encoding="utf-8").strip()
    except OSError:
        pass
    try:
        for line in (root / ".git" / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")) or not line.strip():
                continue
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha.strip()
    except OSError:
        pass
    return "unknown"


# --------------------------------------------------------------------------- #
# Transcript scanning                                                         #
# --------------------------------------------------------------------------- #
def _tool_uses(entry: dict):
    """Yield `(name, input)` for every tool_use block in one transcript entry."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            yield str(block.get("name") or ""), block.get("input") or {}


def _texts(entry: dict):
    message = entry.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            yield str(block.get("text") or "")


def scan_transcript(path, cursor: int = 0) -> dict:
    """Read the transcript from byte `cursor` and extract only what the classifier needs.

    A BYTE CURSOR, not a line count, so a long session does not re-parse its whole history on
    every turn -- the hook runs once per turn and the transcript only grows.

    ⚠ A cursor past EOF means the file was REPLACED, not that there is nothing new: a
    `/compact` rewrites the transcript shorter. Restarting from 0 in that case is the
    difference between classifying the turn and silently classifying nothing.
    """
    path = Path(path)
    writes: list[str] = []
    commands: list[str] = []
    texts: list[str] = []
    asked = False
    lines = 0

    try:
        size = path.stat().st_size
    except OSError:
        return {"writes": [], "commands": [], "texts": [], "asked": False,
                "lines": 0, "cursor": 0}
    if cursor > size:
        cursor = 0

    try:
        with path.open("rb") as fh:
            fh.seek(cursor)
            raw = fh.read()
            end = cursor + len(raw)
    except OSError:
        return {"writes": [], "commands": [], "texts": [], "asked": False,
                "lines": 0, "cursor": cursor}

    for line in raw.splitlines():
        if not line.strip():
            continue
        lines += 1
        try:
            entry = json.loads(line.decode("utf-8", "replace"))
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        for name, inp in _tool_uses(entry):
            if name in WRITE_TOOLS:
                target = inp.get("file_path") or inp.get("notebook_path") or ""
                if target:
                    writes.append(str(target))
            elif name == "Bash":
                commands.append(str(inp.get("command") or ""))
            elif name in QUESTION_TOOLS:
                asked = True
        texts.extend(_texts(entry))

    return {"writes": writes, "commands": commands, "texts": texts, "asked": asked,
            "lines": lines, "cursor": end}


def normalise(target: str, repo_root: Path) -> str:
    """A write target as a repo-relative POSIX path. Absolute paths outside the repo are kept
    as-is (they still count toward R4, they just cannot match a zone)."""
    text = str(target).replace("\\", "/")
    root = str(repo_root).replace("\\", "/").rstrip("/")
    if root and text.lower().startswith(root.lower() + "/"):
        return text[len(root) + 1:]
    return text.lstrip("./")


def merge_scan(state: dict, scan: dict, repo_root: Path) -> dict:
    """Fold one scan into the session's accumulated state.

    ACCUMULATED, not per-turn: the cursor means each turn sees only its own new lines, so the
    verdict has to be taken over everything the session has done. Without this, a turn that
    edits one file after an earlier turn edited another would classify as N-small twice and
    the pair of edits would never be reported.
    """
    merged = dict(state or {})
    writes = list(merged.get("writes") or [])
    commands = list(merged.get("commands") or [])

    for target in scan.get("writes") or []:
        writes.append(normalise(target, repo_root))
    commands.extend(scan.get("commands") or [])

    merged["writes"] = writes
    merged["commands"] = commands
    merged["cursor"] = scan.get("cursor", merged.get("cursor", 0))
    merged["lines"] = int(merged.get("lines") or 0) + int(scan.get("lines") or 0)
    # The QUESTION signals are about how THIS turn ended, so they replace rather than accumulate.
    merged["asked"] = bool(scan.get("asked"))
    texts = [t for t in (scan.get("texts") or []) if t.strip()]
    merged["last_text"] = texts[-1] if texts else ""
    return merged


# --------------------------------------------------------------------------- #
# Classification                                                              #
# --------------------------------------------------------------------------- #
def ends_in_question(text: str) -> bool:
    """True when the assistant's last words ask the user something.

    Trailing markdown is stripped first: a model writes `Should I proceed?**` far more often
    than a bare `?`, and treating that as a statement blocks a turn that was waiting on the
    user -- the one case the gate must never touch."""
    return bool(str(text or "").rstrip(_TRAILING_DECOR).endswith("?"))


def _is_code(path: str) -> bool:
    return path.startswith(CODE_PREFIXES)


def _is_risk(path: str) -> bool:
    return path.startswith(RISK_ZONES)


def _is_report(path: str) -> bool:
    return path.startswith(REPORT_PREFIXES)


def _is_pipeline(command: str) -> bool:
    return bool(_PIPELINE_RE.search(command or ""))


def resolve_kind(writes, commands) -> str:
    """Which of the three report types this task is.

    WRITES OUTRANK COMMANDS, and only when writes exist: what a task CHANGED describes it
    better than what it happened to run, since almost every task runs a pipeline command to
    check its own work. When nothing was written the command is the only evidence there is.

    `reports/**` is excluded from typing -- writing the report is not the work it describes,
    and letting it type the task would make every task REFACTOR."""
    real = [w for w in (writes or []) if not _is_report(w)]

    if real:
        # MODELLING before DATA on a mixed task: a model/strategy change needs the modelling
        # report's gates (IC, turnover, SHAP), which the data report does not carry.
        if any(w.startswith(MODELLING_ZONES) for w in real):
            return "MODELLING"
        if any(w.startswith(DATA_ZONES) for w in real):
            return "DATA"
        return "REFACTOR"

    for command in commands or []:
        module = _MODULE_RE.search(command or "")
        if not module:
            continue
        group = module.group(1)
        if group in MODELLING_COMMANDS:
            return "MODELLING"
        if group in DATA_COMMANDS:
            return "DATA"
    return "REFACTOR"


def classify(state: dict) -> dict:
    """`{required, kind, reasons}` for the session so far.

    The rules are ordered by PRECEDENCE, not by number, and the order is the whole design:

      * N-question first, ahead of every R rule. A turn that ends by asking the user is not
        finished work, and blocking it would deadlock the conversation -- the user cannot
        answer a question the harness refused to deliver.
      * R2 (a single risk-zone write) before N-small, or one line in `data_store/` or
        `configs/` would be exempted as "just one file", which is exactly the change most
        able to break everything downstream.
      * N-small last, so it only catches what no R rule wanted.
    """
    writes = list(state.get("writes") or [])
    commands = list(state.get("commands") or [])
    reasons: list[str] = []

    if state.get("asked"):
        reasons.append("N-question: the turn used AskUserQuestion")
        return {"required": False, "kind": None, "reasons": reasons}
    if ends_in_question(state.get("last_text") or ""):
        reasons.append("N-question: the turn ends by asking the user")
        return {"required": False, "kind": None, "reasons": reasons}

    pipeline = [c for c in commands if _is_pipeline(c)]

    if not writes and not commands:
        reasons.append("N-idle: no writes and no commands -- reads and prose only")
        return {"required": False, "kind": None, "reasons": reasons}

    if writes and all(_is_report(w) for w in writes) and not pipeline:
        reasons.append("N-reports: the only writes were reports/**")
        return {"required": False, "kind": None, "reasons": reasons}

    risk = [w for w in writes if _is_risk(w)]
    if risk:
        reasons.append(f"R2: wrote a risk zone ({', '.join(sorted(set(risk))[:4])})")
        return {"required": True, "kind": resolve_kind(writes, commands), "reasons": reasons}

    if len(writes) >= 8:
        reasons.append(f"R4: {len(writes)} writes in one session")
        return {"required": True, "kind": resolve_kind(writes, commands), "reasons": reasons}

    code = [w for w in writes if _is_code(w)]
    if len(code) >= 2:
        reasons.append(f"R1: {len(code)} code writes ({', '.join(sorted(set(code))[:4])})")
        return {"required": True, "kind": resolve_kind(writes, commands), "reasons": reasons}

    if pipeline and not writes:
        reasons.append(f"R3: ran a pipeline command with no edits ({pipeline[0][:60]})")
        return {"required": True, "kind": resolve_kind(writes, commands), "reasons": reasons}

    if len(writes) == 1 and writes[0].lower().endswith((".md", ".txt", ".rst")):
        reasons.append(f"N-docs: one prose file ({writes[0]})")
        return {"required": False, "kind": None, "reasons": reasons}

    reasons.append(f"N-small: {len(code)} code write(s), no pipeline run")
    return {"required": False, "kind": None, "reasons": reasons}


# --------------------------------------------------------------------------- #
# Report validation                                                           #
# --------------------------------------------------------------------------- #
def recompute_hash(payload: dict) -> str:
    """`sha256:<hex>` over `payload` minus `content_hash`, sorted keys, tight separators.

    ⚠ Must serialise IDENTICALLY to `report_common.content_hash`, including `default=str`.
    `test_hash_matches_the_generator_exactly` runs both over one payload; a drift here would
    reject every genuine report instead of only tampered ones."""
    body = {k: v for k, v in payload.items() if k != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _front_matter(text: str) -> dict:
    """The `key: value` pairs of the leading `---` block. Not a YAML parser -- the front
    matter this reads is written by `report_common.write_report`, one flat pair per line."""
    if not text.startswith("---"):
        return {}
    _, _, rest = text.partition("\n")
    block, _, _ = rest.partition("\n---")
    out = {}
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _section(text: str, heading: str, nxt: str | None) -> str:
    """The body between `heading` and the next section (or the metrics fence)."""
    _, sep, tail = text.partition(heading)
    if not sep:
        return ""
    for stop in [s for s in (nxt, METRICS_FENCE) if s]:
        tail = tail.split(stop, 1)[0]
    return tail.strip()


def validate_report(path, kind: str, session_id: str | None) -> tuple[bool, list[str]]:
    """`(ok, problems)` for one candidate report.

    Every check exists because of a specific way a report can look finished and not be:
    a seeded TODO marker left in place, a section deleted, §5 emptied, a number edited inside
    the metrics block, a report from another task or another session reused.
    """
    problems: list[str] = []
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"cannot read {path.name}: {exc}"]

    front = _front_matter(text)

    declared = front.get("type", "")
    if declared != kind:
        problems.append(f"report is type {declared or '<none>'} but the task is "
                        f"classified as {kind}")

    # "standalone" is the value `load_baseline` synthesises when no hook has ever run, and a
    # generator must stay useful on its own -- so it is not a foreign session.
    owner = front.get("session_id", "")
    if session_id and owner and owner not in (session_id, "standalone", "None"):
        problems.append(f"report belongs to session {owner}, not {session_id}")

    for heading in SECTIONS:
        if heading not in text:
            problems.append(f"missing section: {heading}")

    if TODO_MARKER in text:
        problems.append(f"{TODO_MARKER} markers are still in place -- §1/§5/§6 are unwritten")

    if SECTION_REGRESSIONS in text:
        body = _section(text, SECTION_REGRESSIONS, SECTION_NEXT)
        stripped = "\n".join(ln for ln in body.splitlines()
                             if ln.strip() and not ln.strip().startswith("<!--")
                             and ln.strip() != "-").strip()
        if not stripped:
            problems.append("section 5 is empty -- at least one bullet is mandatory")
        else:
            for line in stripped.splitlines():
                if line.strip().startswith(EMPTY_SECTION_5_PREFIX):
                    detail = line.strip()[len(EMPTY_SECTION_5_PREFIX):].strip()
                    if len(detail) < MIN_CHECKED_CHARS:
                        problems.append(
                            f"'{EMPTY_SECTION_5_PREFIX}' needs >= {MIN_CHECKED_CHARS} "
                            f"characters describing what was checked (got {len(detail)})")

    if METRICS_FENCE not in text:
        problems.append("no ```json dod-metrics block -- numbers must come from a generator")
    else:
        body = text.split(METRICS_FENCE, 1)[1]
        raw = body[body.find("\n"):body.find("\n```")]
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            problems.append(f"the dod-metrics block is not valid JSON: {exc}")
        else:
            stored = payload.get("content_hash")
            if not stored:
                problems.append("the dod-metrics block has no content_hash")
            elif recompute_hash(payload) != stored:
                problems.append("content_hash does not match the block -- it was edited by "
                                "hand; regenerate instead")

    return (not problems), problems


def find_report(root: Path, kind: str, session_id: str | None,
                since_ts: float = 0.0) -> tuple[Path | None, list[str]]:
    """The newest report under `reports/` that VALIDATES for this task, plus why the others
    did not. `since_ts` rejects a report written before the session started -- an old report
    of the right type would otherwise satisfy every future task forever."""
    reports = root / "reports"
    if not reports.is_dir():
        return None, ["no reports/ directory"]

    problems: list[str] = []
    try:
        candidates = sorted((p for p in reports.rglob(f"*__{kind}.md") if p.is_file()),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None, ["reports/ is unreadable"]

    for path in candidates:
        try:
            if path.stat().st_mtime < since_ts:
                continue
        except OSError:
            continue
        ok, why = validate_report(path, kind, session_id)
        if ok:
            return path, []
        problems.append(f"{path.name}: " + "; ".join(why[:3]))
    return None, problems or [f"no fresh *__{kind}.md report since the session started"]
