"""
check_register.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
`configs/fundamentals/fundamentals_check.json`: the SETTLED findings, with their evidence.

This is the half of decision 42 that makes the loop converge. `fundamentals_check` (the table)
is append-only and grows every night; the register is what makes the QUEUE shrink. A finding
investigated once and settled is subtracted from every later run by its `finding_id`, so
nothing is re-investigated and the queue only contains work.

## Why a git-tracked JSON and not a `state` column on the table

Two reasons, both learned rather than assumed:

  * a mutable column is lost on any table rebuild -- and this project rebuilds these tables
    routinely, by design (`fundamentals --rebuild`);
  * "why was this accepted?" would then live in free text outside version control. The
    evidence for an acceptance is the most valuable artifact the loop produces. It belongs
    where a diff can be reviewed.

## THE REGISTER IS NOT A SUPPRESSION LIST

That is the failure mode, and every rule below exists to prevent it:

  * `accepted` requires filing-level `evidence` -- prose naming what was read, and where;
  * `fixed` requires BOTH a `commit` and a named `regression_test` (decision 48). The
    precedent is section 3c.8: four defects were *created by* the 3c.1-3c.5 fixes and were
    visible only on a full re-sweep. A fix with no test is a defect waiting to come back;
  * `fixed` also carries `regression_swept` (decision 64), false until a batched full-roster
    sweep has seen it. A per-finding close is provisional;
  * `wontfix` requires a QUANTIFIED cost in its evidence -- a number, not "small". NEE's
    $5.2bn understatement is a `wontfix`, and it is only defensible because the number is
    written down;
  * `fix_kind: config_proposed` does NOT close a finding (decision 65). `configs/` is the one
    artifact where a wrong entry is invisible forever, so an agent proposes the diff and the
    finding stays OPEN until a human approves it. The withdrawn "UNH has no premiums" edit was
    one approval away from being written;
  * STALE ENTRIES ARE REPORTED. A settled finding whose check no longer fires is surfaced
    every run, so the register decays visibly rather than accumulating suppressions for
    defects that were fixed elsewhere years ago.

## Hand-formatted

Like every other fundamentals config. A `json.dumps` round-trip reformats the entire file and
turns a one-entry change into a whole-file diff, which destroys the reviewability the register
exists for. `render_entry` emits ONE entry in the house style; new entries are spliced in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

#: `configs/` layout. The validator reads this file; only a human writes it.
REGISTER_FILENAME = "fundamentals/fundamentals_check.json"

ACCEPTED, FIXED, WONTFIX = "accepted", "fixed", "wontfix"
#: The closed outcome vocabulary (decision 48). An outcome outside this set is a typo, and a
#: typo here silently suppresses a finding -- so it raises.
OUTCOMES: frozenset[str] = frozenset({ACCEPTED, FIXED, WONTFIX})

CODE, CONFIG_PROPOSED = "code", "config_proposed"
#: Where a fix LANDED (decision 65). Different terminal states, not a label: `code` closes a
#: finding, `config_proposed` does not.
FIX_KINDS: frozenset[str] = frozenset({CODE, CONFIG_PROPOSED})

#: Keys every entry must carry. Absent ones are a schema error, not a default.
REQUIRED_KEYS: tuple[str, ...] = ("finding_id", "check", "ticker", "field", "period_key",
                                  "outcome", "evidence", "decided_on", "decided_by")


class RegisterError(ValueError):
    """A malformed register entry. Raised rather than warned: a register that cannot be parsed
    would silently subtract nothing, and the queue would quietly fill with settled work."""


@dataclass(frozen=True, slots=True)
class SettledFinding:
    """One settled finding, validated on construction."""

    finding_id: str
    check: str
    ticker: str
    field: str
    period_key: str
    outcome: str
    evidence: str
    decided_on: str
    decided_by: str
    fix_kind: str | None = None
    commit: str | None = None
    regression_test: str | None = None
    regression_swept: bool = False

    @property
    def closes(self) -> bool:
        """Does this entry actually take the finding OUT of the queue?

        Everything does EXCEPT a `config_proposed` fix, which is a proposal awaiting approval
        (decision 65). Until the diff is approved the data is still wrong, so the finding is
        still work -- marking it closed would hide an open defect behind a resolved label.
        """
        return not (self.outcome == FIXED and self.fix_kind == CONFIG_PROPOSED)


def parse_entry(raw: dict[str, Any]) -> SettledFinding:
    """One validated `SettledFinding`, or `RegisterError` naming exactly what is wrong."""
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise RegisterError(f"entry {raw.get('finding_id', '?')}: missing key(s) {missing}")

    outcome = raw["outcome"]
    if outcome not in OUTCOMES:
        raise RegisterError(f"entry {raw['finding_id']}: outcome {outcome!r} not in "
                            f"{sorted(OUTCOMES)}")
    fix_kind = raw.get("fix_kind")
    if fix_kind is not None and fix_kind not in FIX_KINDS:
        raise RegisterError(f"entry {raw['finding_id']}: fix_kind {fix_kind!r} not in "
                            f"{sorted(FIX_KINDS)}")
    if not str(raw.get("evidence", "")).strip():
        raise RegisterError(f"entry {raw['finding_id']}: every outcome needs evidence")

    if outcome == FIXED:
        # Decision 48. Both, and non-empty -- a fix without a regression test is the 3c.8
        # failure mode with a resolved label on it.
        for key in ("commit", "regression_test"):
            if not str(raw.get(key) or "").strip():
                raise RegisterError(f"entry {raw['finding_id']}: outcome 'fixed' requires "
                                    f"a non-empty {key!r}")
        if fix_kind is None:
            raise RegisterError(f"entry {raw['finding_id']}: outcome 'fixed' requires "
                                "fix_kind ('code' or 'config_proposed')")
    if outcome == WONTFIX and not _has_number(raw["evidence"]):
        raise RegisterError(f"entry {raw['finding_id']}: outcome 'wontfix' requires a "
                            "QUANTIFIED cost in evidence -- a number, not an adjective")

    return SettledFinding(
        finding_id=str(raw["finding_id"]), check=str(raw["check"]),
        ticker=str(raw["ticker"]), field=str(raw["field"]),
        period_key=str(raw["period_key"]), outcome=outcome,
        evidence=str(raw["evidence"]), decided_on=str(raw["decided_on"]),
        decided_by=str(raw["decided_by"]), fix_kind=fix_kind,
        commit=raw.get("commit"), regression_test=raw.get("regression_test"),
        regression_swept=bool(raw.get("regression_swept", False)))


def _has_number(text: str) -> bool:
    """Does this evidence string contain a digit? The `wontfix` cost test.

    Deliberately crude. The rule being enforced is "somebody measured it", and any real
    measurement carries a numeral; a stricter parser would reject legitimate phrasings and
    teach people to write around it.
    """
    return any(ch.isdigit() for ch in str(text))


class CheckRegister:
    """The parsed register, plus the two questions the validator asks of it."""

    def __init__(self, entries: Iterable[SettledFinding]) -> None:
        self._by_id: dict[str, SettledFinding] = {e.finding_id: e for e in entries}

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, finding_id: str) -> bool:
        return finding_id in self._by_id

    def get(self, finding_id: str) -> SettledFinding | None:
        return self._by_id.get(finding_id)

    def is_settled(self, finding_id: str) -> bool:
        """Should this finding be subtracted from the queue?

        A `config_proposed` fix is present in the register and still NOT settled -- see
        `SettledFinding.closes`.
        """
        entry = self._by_id.get(finding_id)
        return entry is not None and entry.closes

    def open_proposals(self) -> list[SettledFinding]:
        """Entries awaiting a `configs/` approval. Reported every run so a proposal cannot be
        forgotten in a JSON file while the data stays wrong."""
        return [e for e in self._by_id.values() if not e.closes]

    def unswept_fixes(self) -> list[SettledFinding]:
        """`fixed` entries no full-roster sweep has confirmed yet (decision 64).

        A phase must not close while this is non-empty: the per-finding close was scoped to
        the affected tickers, and section 3c.8's four fix-induced defects were visible only on
        a full re-sweep.
        """
        return [e for e in self._by_id.values()
                if e.outcome == FIXED and not e.regression_swept]

    def stale(self, fired_ids: Iterable[str]) -> list[SettledFinding]:
        """Settled entries whose check did NOT fire this run.

        Not an error -- a fixed defect SHOULD stop firing, and an `accepted` one whose
        underlying data was re-fetched may legitimately go quiet. It is reported so the
        register decays visibly instead of accumulating suppressions nobody can justify.
        """
        fired = set(fired_ids)
        return [e for e in self._by_id.values() if e.finding_id not in fired]


def load_register(config_dir: str | Path = "./configs") -> CheckRegister:
    """Read and validate the register. A missing file is an EMPTY register, not an error --
    that is the state on day one and before the first finding is ever settled."""
    path = Path(config_dir) / REGISTER_FILENAME
    if not path.exists():
        return CheckRegister([])
    blob = json.loads(path.read_text(encoding="utf-8"))
    entries = blob.get("findings", []) if isinstance(blob, dict) else blob
    parsed = [parse_entry(raw) for raw in entries]
    duplicates = sorted({e.finding_id for e in parsed
                         if sum(1 for o in parsed if o.finding_id == e.finding_id) > 1})
    if duplicates:
        raise RegisterError(f"duplicate finding_id(s) in the register: {duplicates}; "
                            "two entries settling one finding can disagree")
    return CheckRegister(parsed)


#: The house style for one entry, so an addition is a small diff rather than a reformat of the
#: whole file. Keys in a fixed order; `evidence` last of the prose keys because it is the long
#: one and a reviewer reads it as a paragraph.
_ENTRY_TEMPLATE = """    {{
      "finding_id": {finding_id}, "check": {check},
      "ticker": {ticker}, "field": {field}, "period_key": {period_key},
      "outcome": {outcome}, "fix_kind": {fix_kind},
      "commit": {commit}, "regression_test": {regression_test},
      "regression_swept": {regression_swept},
      "decided_on": {decided_on}, "decided_by": {decided_by},
      "evidence": {evidence}
    }}"""


def render_entry(entry: dict[str, Any]) -> str:
    """One register entry in the file's hand-written style, validated first.

    Validated on the way out, not just on the way in: an entry rendered into the file without
    passing `parse_entry` would be a schema violation that only the NEXT run discovers, by
    which time the diff has been approved and merged.
    """
    parse_entry(entry)
    return _ENTRY_TEMPLATE.format(**{
        key: json.dumps(entry.get(key)) for key in
        ("finding_id", "check", "ticker", "field", "period_key", "outcome", "fix_kind",
         "commit", "regression_test", "regression_swept", "decided_on", "decided_by",
         "evidence")})
