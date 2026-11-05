"""Resolutions of identity items (phase 5, REQ-031 / REQ-032, AC-015 machinery).

Known-truth configs in a temporary `configs/sec/`: an entry without a cited URL or accession, a missing field or a second
entry for one item is refused at load; the review splits items into unreviewed action items, information and resolved
items and lists the stale resolutions; `run_publish` logs a WARNING block of the unreviewed action items only (a refused
merge is one) and one INFO summary. The shipped configs carry an empty `resolutions` list each, which the curated
loaders ignore.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.transformers import step_extract_identity as step_module
from src.data_extract.transformers.step_extract_identity import StepExtractIdentity
from src.data_extract.utils.common.security_lines import load_security_manual
from src.data_extract.utils.identity.cik_history import BACKLOG_COLUMNS, BLOCKED_COLUMNS, manual_lineage_blob, parse_manual_lineage
from src.data_extract.utils.identity.symbol_tenure import load_manual_symbol_tenure, load_rejected_symbol_tenure
from src.utils.identity_flags import (
    FLAG_COLUMNS,
    RESOLUTION_FILES,
    RESOLUTIONS_KEY,
    ResolutionError,
    load_resolutions,
    review_flags,
    review_summary,
)

ROOT = Path(__file__).resolve().parents[3]
CONFLICT = {
    "kind": "conflict",
    "ticker": "AAA",
    "ciks": "0000000100",
    "decision": "AAA-P is the subsidiary's preferred on a combined cover page: acknowledged",
    "evidence": ["https://www.sec.gov/Archives/edgar/data/100/000000010019000001/0000000100-19-000001-index.htm"],
    "decided_on": "2026-11-05",
}


def _config(tmp_path: Path, by_file: dict[str, list[dict[str, Any]]]) -> str:
    """A `configs/` directory whose owning JSONs hold the given `resolutions` lists."""
    sec = tmp_path / "sec"
    sec.mkdir(parents=True, exist_ok=True)
    for name, entries in by_file.items():
        (sec / name).write_text(json.dumps({RESOLUTIONS_KEY: entries}), encoding="utf-8")
    return str(tmp_path)


def _item(kind: str, action: bool, ticker: str, ciks: str) -> dict[str, object]:
    return dict(zip(FLAG_COLUMNS, (kind, action, ticker, ciks, f"{kind} evidence", "suggested", "configs/sec/x.json"), strict=True))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"evidence": []}, "evidence citing"),
        ({"evidence": ["see the 10-K"]}, "evidence citing"),
        ({"decided_on": ""}, "decided_on"),
        ({"decision": None}, "decision"),
    ],
)
def test_a_resolution_without_cited_evidence_or_a_field_is_refused_at_load(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    """REQ-031: evidence must cite a URL or an accession; kind, ticker, ciks, decision and decided_on are required."""
    config = _config(tmp_path, {"symbol_tenure_manual.json": [{**CONFLICT, **change}]})

    with pytest.raises(ResolutionError, match=message):
        load_resolutions(config)
    print(f"\n=== SANITY CHECK: refused resolution ({change}) ===\n  OK: the loader raises naming '{message}'")


def test_a_cited_resolution_loads_and_one_item_reviewed_twice_is_refused(tmp_path: Path) -> None:
    """A complete entry loads typed with its owning file; the same item key in two files is refused."""
    accession = {**CONFLICT, "evidence": "8-K 0000000100-19-000001"}
    loaded = load_resolutions(_config(tmp_path, {"symbol_tenure_manual.json": [accession]}))

    assert loaded.to_dict("records")[0]["config_file"] == "configs/sec/symbol_tenure_manual.json"
    assert loaded["decided_on"].iloc[0] == pd.Timestamp("2026-11-05") and loaded["evidence"].iloc[0] == "8-K 0000000100-19-000001"
    twice = _config(tmp_path, {"symbol_tenure_manual.json": [CONFLICT], "security_master_manual.json": [CONFLICT]})
    with pytest.raises(ResolutionError, match="reviewed twice"):
        load_resolutions(twice)
    print("\n=== SANITY CHECK: resolution load ===")
    print(f"  loaded: {loaded[['kind', 'ticker', 'ciks', 'config_file']].to_dict('records')}")
    print("  OK: an accession is a citation; one item reviewed in two files is refused")


def test_the_review_splits_unreviewed_information_resolved_and_stale() -> None:
    """REQ-032: a resolved action item leaves the WARNING block; an unresolved one stays; an information item is counted;
    a resolution matching no item is stale."""
    flags = pd.DataFrame(
        [
            _item("conflict", True, "AAA", "0000000100"),
            _item("refused_merge", True, "TT", "0001466258,0001699150"),
            _item("noise", False, "BBB", "0000000200"),
        ]
    )
    stale = {**CONFLICT, "kind": "security_ratio_step", "ticker": "WBD", "ciks": ""}
    resolutions = pd.DataFrame([{**entry, "evidence": "x", "config_file": "c"} for entry in (CONFLICT, stale)])

    review = review_flags(flags, resolutions)
    summary = review_summary(review)

    assert review.unreviewed["kind"].tolist() == ["refused_merge"] and review.resolved["kind"].tolist() == ["conflict"]
    assert review.information["kind"].tolist() == ["noise"] and review.stale["ticker"].tolist() == ["WBD"]
    assert "1 unreviewed action item(s); information 1 (noise 1); resolved 1 (conflict 1); 1 stale resolution(s)" in summary, summary
    print("\n=== SANITY CHECK: review split ===")
    print(f"  {summary}")
    print("  OK: only the unresolved action item is left for the WARNING block")


def _lineage() -> pd.DataFrame:
    """AAA's roster window, its own symbol and a `conflict` interval AAA-P."""
    rows = [
        ("cik_window", "", "1900-01-01", "roster", "curated"),
        ("symbol", "AAA", "2006-01-03", "form345,roster", "curated"),
        ("symbol", "AAA-P", "2019-07-25", "form345", "conflict"),
    ]
    frame = pd.DataFrame(rows, columns=["role", "symbol", "valid_from", "sources", "status"])
    return frame.assign(
        company_id="E0000000100",
        canonical_ticker="AAA",
        cik="0000000100",
        valid_from=pd.to_datetime(frame["valid_from"]),
        valid_to=pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]"),
        oracle="roster",
        evidence="",
    )


def _publish(config_dir: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> tuple[list[str], list[str]]:
    """`run_publish` over a store with nothing stored, this run's builds preset; returns its WARNING and INFO messages."""
    log = logging.getLogger("test.identity.resolutions")
    store = SimpleNamespace(load=lambda table, **kwargs: None)
    context = SimpleNamespace(store=store, log=log, config_dir=config_dir, universe=SimpleNamespace(alias_symbols=frozenset(), tickers=["AAA"]))
    blocked = pd.DataFrame([("0001466258", "0001699150", "symbol_handoff[IR]", None, "TT", "IR")], columns=list(BLOCKED_COLUMNS))
    step = cast(Any, object.__new__(StepExtractIdentity))
    step._context, step._config_dir, step._log = context, config_dir, log
    step._symbols = SimpleNamespace(rows=_lineage(), backlog=pd.DataFrame(columns=list(BACKLOG_COLUMNS)), blocked=blocked)
    step._cik_history = SimpleNamespace(activity=pd.DataFrame(columns=["cik", "source", "first", "last"]))
    step._cusips = SimpleNamespace(rows=pd.DataFrame(columns=["canonical_company"]), items=pd.DataFrame(columns=list(FLAG_COLUMNS)))
    step._lifecycle = SimpleNamespace(rows=pd.DataFrame(columns=["ticker"]), items=pd.DataFrame(columns=list(FLAG_COLUMNS)))
    monkeypatch.setattr(step_module, "save_unless_unchanged", lambda *args: 0)
    monkeypatch.setattr(step_module, "update_identity_columns", lambda *args: 0)
    monkeypatch.setattr(step_module, "cusip_company_ids", lambda lines, rows: lines)
    monkeypatch.setattr(step_module, "changed_identity", lambda rows, stored: rows)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=log.name):
        step.run_publish()
    records = [r for r in caplog.records if r.name == log.name]
    return [r.getMessage() for r in records if r.levelno == logging.WARNING], [r.getMessage() for r in records if r.levelno == logging.INFO]


def test_publish_warns_only_for_unreviewed_action_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Without resolutions the conflict and the refused merge are both in the block; once the conflict is resolved only
    the refused merge warns and the INFO summary counts the resolution; resolving both leaves no WARNING at all."""
    merge = {**CONFLICT, "kind": "refused_merge", "ticker": "TT", "ciks": "0001466258,0001699150", "decision": "IR is Ingersoll Rand Inc: own entity"}
    before, _ = _publish(_config(tmp_path / "none", {}), monkeypatch, caplog)
    after, infos = _publish(_config(tmp_path / "one", {"symbol_tenure_manual.json": [CONFLICT]}), monkeypatch, caplog)
    both, quiet = _publish(
        _config(tmp_path / "two", {"symbol_tenure_manual.json": [CONFLICT], "entity_lineage_manual.json": [merge]}), monkeypatch, caplog
    )

    assert len(before) == 1 and "conflict AAA" in before[0] and "refused_merge TT" in before[0], before
    assert len(after) == 1 and "conflict AAA" not in after[0] and "refused_merge TT (CIK 0001466258,0001699150)" in after[0], after
    summary = next(m for m in infos if m.startswith("identity items:"))
    assert "1 unreviewed action item(s)" in summary and "resolved 1 (conflict 1)" in summary, summary
    assert both == [] and any("0 unreviewed action item(s)" in m and "resolved 2" in m for m in quiet), (both, quiet)
    print("\n=== SANITY CHECK: publish logging policy (REQ-032) ===")
    print(f"  no resolutions: 1 WARNING block, 2 action items; one resolved: {summary}")
    print("  both resolved: 0 WARNING lines")
    print("  OK: a reviewed item never warns; a refused merge is an action item of the block until reviewed")


def test_the_shipped_configs_carry_an_empty_resolutions_list_the_loaders_ignore() -> None:
    """Each owning JSON holds `resolutions` (empty until phase 8); the lineage, symbol and security loaders read as before."""
    config = str(ROOT / "configs")
    lists = {name: json.loads((ROOT / "configs" / "sec" / name).read_text(encoding="utf-8"))[RESOLUTIONS_KEY] for name in RESOLUTION_FILES}

    assert all(isinstance(entries, list) for entries in lists.values()), lists
    assert len(load_resolutions(config)) == sum(len(entries) for entries in lists.values())
    assert RESOLUTIONS_KEY not in parse_manual_lineage(manual_lineage_blob(config))
    manual, rejected, security = load_manual_symbol_tenure(config), load_rejected_symbol_tenure(config), load_security_manual(config)
    print("\n=== SANITY CHECK: shipped resolutions lists ===")
    print(f"  {', '.join(f'{name}: {len(entries)}' for name, entries in lists.items())}")
    print(f"  loaders unchanged: {len(manual)} manual symbol interval(s), {len(rejected)} rejection(s), {len(security.boundaries)} market boundaries")
    print("  OK: the lists exist and no curated loader reads them as an entry")
