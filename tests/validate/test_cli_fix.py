"""
test_cli_fix.py  (tests/validate/)
--------------------------------------------------------------------------------------------
WHAT `validate fix record` REFUSES, and what it writes when it does not.

The command exists because cluster `1c9a517eaa47` was fixed on 2026-08-25 and left no
machine-readable trace anywhere -- not in the status table, not in the settled set, not in the
report. Its only record was a commit sha.

Every assertion here is about a REFUSAL or about atomicity, because those are the properties
that decide whether the table is evidence or decoration. A fix row that cites an unresolvable
commit, a missing test, two incomparable runs or a check that is not firing reads as proof and
is not proof, and nobody discovers that until they try to reproduce it -- by which time the
run it was measured against may be gone.

The single exception is the NO-IMPROVEMENT path, which warns and writes. Correcting a
wrong-but-plausible value where no check was firing is a real fix; it simply cannot settle
anything. Permissive to record, strict to settle.

Synthetic frames throughout. This is refusal logic, not economics -- AGENTS.md's real-data
rule applies to feature and economic tests, and the two runs here need shapes no live ledger
happens to contain.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_store.schema import Tables
from src.validate.cli import fix_record, fix_show

CLUSTER = "1c9a517eaa47"

#: The two runs the MCD fix was measured between, at ONE scope hash. That equality is the
#: whole reason a before/after count is a comparison rather than two unrelated numbers.
BEFORE, AFTER, SCOPE = "3df52ae9af75", "725bae7bf8ed", "scopeAAA"


def _findings(run_id: str, run_date: str, specs) -> pd.DataFrame:
    """`(check_name, severity, period)` tuples -> one run's slice of `fundamentals_check`."""
    return pd.DataFrame([{
        "run_date": pd.Timestamp(run_date), "run_id": run_id, "cluster_id": CLUSTER,
        "check_name": check, "ticker": "MCD", "field": "capex", "period_key": period,
        "finding_id": f"{check}{period}", "tier": 2, "severity": severity,
        "substrate": "facts", "edgar_url": "https://sec.gov/MCD", "detail": "{}",
    } for check, severity, period in specs])


#: 6 queue findings before, 2 after -- the shape a settlement is judged on.
_BEFORE_ROWS = _findings(BEFORE, "2026-08-24", [
    ("cross_identity", "high", f"2019-0{i}-30") for i in range(1, 5)]
    + [("peer_ratio", "medium", "2019-06-30"), ("peer_ratio", "medium", "2020-06-30")])
_AFTER_ROWS = _findings(AFTER, "2026-08-25", [
    ("peer_ratio", "medium", "2019-06-30"), ("peer_ratio", "medium", "2020-06-30"),
    ("catalogue_exclusion_cost", "info", "")])


def _runs() -> pd.DataFrame:
    return pd.DataFrame([{
        "run_id": run_id, "run_date": pd.Timestamp(day), "check_name": "peer_ratio",
        "scope_hash": scope, "scope_roster": "in_sample", "scope_tickers": 54,
        "scope_ticker_list": '["MCD"]', "scope_fields": "[]", "scope_tiers": "[1, 2, 3]",
        "tier": 2, "substrate": "facts", "examined": 100, "queued": 5, "info": 0,
        "ceiling": 0.06, "abstained": False, "over_ceiling": False,
    } for run_id, day, scope in [(BEFORE, "2026-08-24", SCOPE), (AFTER, "2026-08-25", SCOPE),
                                 ("otherscope1", "2026-08-23", "scopeBBB")]])


class _Store:
    """An in-memory stand-in for `DataStore`, recording every write so a rollback is visible.

    A fake rather than a live DB: the point of these tests is which calls the command makes
    and in what order, and a real Postgres would make the atomicity assertion depend on
    transaction semantics that `store.save` does not actually provide.
    """

    def __init__(self) -> None:
        self.tables: dict[str, pd.DataFrame] = {
            str(Tables.fundamentals_check): pd.concat([_BEFORE_ROWS, _AFTER_ROWS],
                                                      ignore_index=True),
            str(Tables.fundamentals_check_run): _runs(),
            str(Tables.fundamentals_check_status): pd.DataFrame(),
            str(Tables.fundamentals_check_fix): pd.DataFrame(),
        }
        self.fail_on: str | None = None

    def load(self, table, columns=None, where=None, optional=False):
        frame = self.tables.get(str(table), pd.DataFrame())
        return frame.copy()

    def save(self, table, df, pk=None) -> int:
        name = str(table)
        if self.fail_on == name:
            raise RuntimeError(f"simulated write failure on {name}")
        existing = self.tables.get(name, pd.DataFrame())
        self.tables[name] = (pd.concat([existing, df], ignore_index=True)
                             if not existing.empty else df.copy())
        return len(df)

    def delete(self, table, where) -> int:
        name = str(table)
        frame = self.tables.get(name, pd.DataFrame())
        if frame.empty:
            return 0
        mask = pd.Series(True, index=frame.index)
        for column, value in where.items():
            values = value if isinstance(value, (list, tuple, set)) else [value]
            mask &= frame[column].isin(list(values))
        self.tables[name] = frame[~mask].reset_index(drop=True)
        return int(mask.sum())


class _Log:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _record(self, message, *args):
        self.messages.append(str(message) % args if args else str(message))

    info = warning = error = _record


@pytest.fixture
def store(monkeypatch) -> _Store:
    """Patch `_ctx` so the command talks to the fake store instead of Postgres."""
    fake = _Store()
    log = _Log()
    context = type("Ctx", (), {"store": fake, "log": log})()
    monkeypatch.setattr("src.validate.cli._ctx", lambda _p: (None, context))
    fake.log = log
    return fake


#: A well-formed invocation. Individual tests override one field to trip one refusal.
def _args(**overrides) -> list[str]:
    base = {
        "--layer": "extraction",
        "--root-cause": "route 1 took a total the filer declares beside its own leg",
        "--evidence": json.dumps({"accessions": ["0000063908-18-000010"]}),
        "--commit": "HEAD",
        "--test": "tests/validate/test_cli_fix.py",
    }
    base.update(overrides)
    args = [CLUSTER]
    for flag, value in base.items():
        if value is not None:
            args += [flag, value]
    return args


def _run(store: _Store, *extra, **overrides):
    return CliRunner().invoke(fix_record, _args(**overrides) + list(extra))


def _message(result) -> str:
    """Everything the invocation said, wherever it said it.

    A `ClickException` exits via `SystemExit`, so `result.exception` is the bare code `1` and
    the sentence a human reads went to stderr. Asserting on the exception alone would pass on
    any non-zero exit -- including a crash -- which is the opposite of testing a refusal.
    """
    parts = [result.output or ""]
    try:
        parts.append(result.stderr or "")
    except ValueError:                       # stderr not captured separately on this click
        pass
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        parts.append(repr(result.exception))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# the refusals -- each its own assertion, each naming the rule it enforces      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("case,overrides,extra,expect", [
    ("unknown layer", {"--layer": "vibes"}, (), "layer"),
    ("evidence is prose", {"--evidence": "we fixed the capex route"}, (),
     "parseable JSON"),
    ("evidence missing its layer's keys", {"--evidence": '{"note": "x"}'}, (),
     "missing ['accessions']"),
    ("unresolvable commit", {"--commit": "deadbeefdeadbeef"}, (), "does not resolve"),
    ("missing test file", {"--test": "tests/does/not/exist.py"}, (), "does not exist"),
    ("incomparable runs", {}, ("--before", "otherscope1"), "different scope hashes"),
    ("waiving a check that is not firing", {}, ("--waive", "trend_break:3 findings"),
     "no finding on cluster"),
    ("an unquantified waiver note", {}, ("--waive", "peer_ratio:looks fine to me"),
     "QUANTIFIED"),
])
def test_fix_record_refuses_what_it_cannot_verify(store, case, overrides, extra, expect):
    """Eight ways to record a fix that reads as proof and is not proof."""
    result = _run(store, *extra, **overrides)
    message = _message(result)

    print(f"\n{case:<40} -> {message.strip().splitlines()[0][:110]}")
    assert result.exit_code != 0, f"{case} was ACCEPTED"
    assert expect in message, f"{case}: message did not name the rule ({message[:200]})"
    assert store.tables[str(Tables.fundamentals_check_fix)].empty
    assert store.tables[str(Tables.fundamentals_check_status)].empty


def test_an_unknown_cluster_is_refused_before_anything_is_measured(store):
    """A fix on a cluster nobody measured has no before/after and can prove nothing."""
    result = CliRunner().invoke(fix_record, ["nosuchcluster"] + _args()[1:])
    message = _message(result)

    print(f"\nunknown cluster -> {message.strip().splitlines()[0][:120]}")
    print("  SANITY: refused at derivation, before any count is taken.")
    assert result.exit_code != 0 and "not in fundamentals_check" in message


# --------------------------------------------------------------------------- #
# what it writes                                                               #
# --------------------------------------------------------------------------- #

def test_a_well_formed_record_derives_every_count_from_the_ledger(store):
    """Decision 7: only `cluster_id` is required. Everything else is looked up.

    Asking a human to type two 12-hex run ids and four counts they would have to query is how
    a wrong number lands on an evidence row -- and an evidence row nobody can check is not
    evidence.
    """
    result = _run(store, "--waive", "peer_ratio:2 findings, 8.3% capex/revenue vs 3.5% median")
    assert result.exit_code == 0, result.exception
    row = store.tables[str(Tables.fundamentals_check_fix)].iloc[0]
    waiver = store.tables[str(Tables.fundamentals_check_status)].iloc[0]

    print(f"\nderived runs   : {row['run_id_before']} -> {row['run_id_after']} "
          f"(scope {row['scope_hash']})")
    print(f"derived identity: {row['ticker']} {row['field']}")
    print(f"derived counts : findings {row['findings_before']} -> {row['findings_after']}, "
          f"queue {row['queued_before']} -> {row['queued_after']}")
    print(f"waiver         : {waiver['check_name']} at "
          f"{waiver['findings_at_decision']} finding(s)")
    print("  SANITY: nothing above was typed on the command line except the cluster id. "
          "The `info` finding is counted in findings_after but NOT in queued_after.")
    assert (row["run_id_before"], row["run_id_after"]) == (BEFORE, AFTER)
    assert (row["ticker"], row["field"]) == ("MCD", "capex")
    assert row["findings_before"] == 6 and row["findings_after"] == 3
    assert row["queued_before"] == 6 and row["queued_after"] == 2
    assert waiver["check_name"] == "peer_ratio" and waiver["findings_at_decision"] == 2


def test_evidence_is_stored_as_canonical_json_not_as_typed(store):
    """JSON, never prose, and normalised so two identical records compare equal."""
    result = _run(store, **{"--evidence":
                            '{"concepts": {"took": "A"}, "accessions": ["0000063908-18-000010"]}'})
    assert result.exit_code == 0, result.exception
    stored = store.tables[str(Tables.fundamentals_check_fix)].iloc[0]["evidence"]

    print(f"\nstored evidence: {stored}")
    print("  SANITY: key-sorted JSON. Prose lives in root_cause; this field is what a later "
          "reader QUERIES.")
    assert json.loads(stored)["accessions"] == ["0000063908-18-000010"]
    assert stored == json.dumps(json.loads(stored), sort_keys=True)


def test_a_no_improvement_fix_WARNS_and_still_writes(store):
    """Decision 8, and the one case that is permissive.

    A wrong-but-plausible value corrected where no check was firing closed no queue finding
    and is still a real fix. So it is recordable, it is loud, and `Ledger.qualifying_fix`
    refuses to let it settle anything.
    """
    store.tables[str(Tables.fundamentals_check)] = pd.concat(
        [_findings(BEFORE, "2026-08-24",
                   [("peer_ratio", "medium", "2019-06-30"),
                    ("peer_ratio", "medium", "2020-06-30")]),
         _AFTER_ROWS], ignore_index=True)
    result = _run(store)
    assert result.exit_code == 0, result.exception
    row = store.tables[str(Tables.fundamentals_check_fix)].iloc[0]
    warned = [m for m in store.log.messages if "closed NO queue findings" in m]

    print(f"\nqueue {row['queued_before']} -> {row['queued_after']} (no improvement)")
    print(f"warning: {warned[0][:150] if warned else 'NONE'}")
    print("  SANITY: written AND warned, and the warning says it cannot settle the cluster. "
          "Refusing it would lose a legitimate fix; settling on it would be a lie.")
    assert row["queued_before"] == 2 and row["queued_after"] == 2
    assert warned, "a no-improvement fix must warn"


def test_a_failed_waiver_write_ROLLS_BACK_the_fix_row(store):
    """Atomicity (decision 6). The failure mode must be "nothing recorded".

    A fix row whose cluster still reads OPEN because its waivers never landed is the exact
    half-recorded state this command was written to remove.
    """
    store.fail_on = str(Tables.fundamentals_check_status)
    result = _run(store, "--waive", "peer_ratio:2 findings, 8.3% vs 3.5% peer median")

    print(f"\nexit code: {result.exit_code} ({type(result.exception).__name__})")
    print(f"fix rows written   : {len(store.tables[str(Tables.fundamentals_check_fix)])}")
    print(f"waiver rows written: {len(store.tables[str(Tables.fundamentals_check_status)])}")
    print("  SANITY: neither table advanced. Recording a fix and tolerating its residue is "
          "ONE decision, so it lands whole or not at all.")
    assert result.exit_code != 0
    assert store.tables[str(Tables.fundamentals_check_fix)].empty
    assert store.tables[str(Tables.fundamentals_check_status)].empty


def test_a_failed_fix_write_removes_the_waivers_it_already_wrote(store):
    """The other order. The waivers land first, so this is the rollback that has real work."""
    store.fail_on = str(Tables.fundamentals_check_fix)
    result = _run(store, "--waive", "peer_ratio:2 findings, 8.3% vs 3.5% peer median")

    print(f"\nexit code: {result.exit_code}")
    print(f"waiver rows left after rollback: "
          f"{len(store.tables[str(Tables.fundamentals_check_status)])}")
    print(f"rollback logged: "
          f"{[m for m in store.log.messages if 'rolled back' in m][:1]}")
    print("  SANITY: the waiver was written and then removed. A waiver with no fix behind it "
          "would leave the cluster reading `wontfix` for a fix that was never recorded.")
    assert result.exit_code != 0
    assert store.tables[str(Tables.fundamentals_check_status)].empty


def test_the_ledger_is_never_touched_by_recording_a_fix(store):
    """The invariant the whole design rests on, asserted at the CLI boundary too."""
    before = len(store.tables[str(Tables.fundamentals_check)])
    _run(store, "--waive", "peer_ratio:2 findings, 8.3% vs 3.5% peer median")
    after = len(store.tables[str(Tables.fundamentals_check)])

    print(f"\nfundamentals_check rows: {before} before, {after} after")
    print("  SANITY: identical. A fix row records what was done; it never subtracts one, "
          "which is what keeps a row-count drop usable as proof.")
    assert before == after


def test_fix_show_answers_reason_what_before_after_and_pending(store):
    """The read-back. One query, five questions -- that is the point of the command."""
    _run(store, "--waive", "peer_ratio:2 findings, 8.3% capex/revenue vs 3.5% peer median")
    result = CliRunner().invoke(fix_show, [CLUSTER])
    printed = "\n".join(store.log.messages)

    print(f"\n{printed[printed.find('cluster '):][:900]}")
    print("  SANITY: reason, layer, before -> after, evidence, test, waivers and what is "
          "still pending -- without a second query or a source dive.")
    assert result.exit_code == 0, result.exception
    assert "route 1 took a total" in printed          # reason
    assert "layer=extraction" in printed              # what
    assert "queue 6 -> 2" in printed                  # before / after
    assert "0000063908-18-000010" in printed          # evidence
    assert "nothing unwaived at queue severity remains" in printed   # pending
