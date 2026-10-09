"""
`entity_lineage` -- the id minting, the oracle priority, the refusals and the dated verdict rows,
on synthetic known-truth inputs, plus a real-data check that the live register and curated files
produce the verdicts the plan specifies.

The refusals are the point of this file:
  * a merge joining two UNIVERSE tickers is rejected -- the only failure in this design that
    corrupts rows rather than dropping them;
  * an owner-overlap score in the grey band, a reuse conflict, an uncorroborated extra CIK and an
    older-CIK rekey are EXCLUDED and backlogged -- never guessed, and never a stopped build;
  * a D19 roster disagreement with no allow-list entry still stops the build.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import entity_lineage as lineage_module
from src.data_extract.utils.common.entity_lineage import (
    OVERLAP_JACCARD_SAME,
    OVERLAP_SHARED_SAME,
    ManualTenureEntityError,
    _Union,
    candidate_ciks,
    classify_overlap,
    derive_entity_lineage,
    detect_older_cik_rekeys,
    entity_id_for,
    load_d19_allowlist,
    load_manual_lineage,
    score_overlap,
    validate_manual_tenure_entities,
)
from src.data_extract.utils.common.registrant import load_registrants
from src.data_extract.utils.common.symbol_tenure import scan_form345_cache
from src.data_store.schema import Tables
from src.utils.identity_flags import KIND_ORDER, cik_activity, identity_flags

CONFIG_DIR = "./configs"
CACHE = Path("data/sec_insider_transactions")
L: Any = lineage_module  # P3 names are read off the module so a missing one fails its test, not collection


def _tenure(rows: list[tuple[str, str, str, str | None, int]], source: str = "form345") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": s,
                "issuer_cik": c,
                "valid_from": pd.Timestamp(f),
                "valid_to": pd.Timestamp(t) if t else pd.NaT,
                "n_filings": n,
                "source": source,
                "evidence": f"{source} {c}",
            }
            for s, c, f, t, n in rows
        ]
    )


def _roster(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "cik": c} for t, c in rows])


def _owner_pairs(owners: dict[str, set[str]]) -> pd.DataFrame:
    """The (issuer_cik, owner_cik_raw) pair frame a Form 345 cache scan yields."""
    return pd.DataFrame([(issuer, owner) for issuer, members in owners.items() for owner in sorted(members)], columns=["issuer_cik", "owner_cik_raw"])


def _config(tmp_path: Path, register: dict | None = None, manual: dict | None = None) -> str:
    """A config directory holding only the given register / manual lineage JSON."""
    sec = tmp_path / "configs" / "sec"
    sec.mkdir(parents=True, exist_ok=True)
    if register is not None:
        (sec / "registrant_cutover.json").write_text(json.dumps(register), encoding="utf-8")
    if manual is not None:
        (sec / "entity_lineage_manual.json").write_text(json.dumps(manual), encoding="utf-8")
    return str(tmp_path / "configs")


def _rows(build: Any, **match: object) -> pd.DataFrame:
    """The rows of a build matching every `column=value`."""
    rows = build.rows
    for column, value in match.items():
        rows = rows[rows[column].eq(value)]
    return rows


def _handoff_evidence(*, dei_successor_first: str = "2015-07-10", dei_predecessor_last: str = "2015-06-16") -> tuple[pd.DataFrame, pd.DataFrame]:
    """`ABC` passes from CIK 100 to the roster CIK 200 on 2015-07-01 on Forms 3/4/5; `dei` places the switch as given."""
    tenure = _tenure([("ABC", "0000000100", "2006-01-05", "2015-07-01", 300), ("ABC", "0000000200", "2015-07-01", None, 100)])
    dei = _tenure(
        [("ABC", "0000000100", "2010-03-01", dei_predecessor_last, 20), ("ABC", "0000000200", dei_successor_first, "2024-11-02", 30)],
        source="dei",
    )
    return tenure, dei


# --------------------------------------------------------------------------- #
# membership                                                                    #
# --------------------------------------------------------------------------- #
def test_entity_id_is_the_oldest_cik_in_the_group():
    """No allocation state, so two independent derivations agree by construction."""
    group = {"0000030554", "0001666700", "0000029915"}
    assert entity_id_for(group) == "E0000029915"
    assert entity_id_for({"0001666700"}) == "E0001666700"

    print("\n=== SANITY CHECK: entity_id minting ===")
    print(f"  {sorted(group)} -> {entity_id_for(group)}")
    print("  OK: the numerically smallest (oldest) CIK names the group")


def test_merge_order_does_not_change_the_id():
    """The union-find visits pairs in oracle order; a different order must not rename an entity."""
    ids = []
    for pairs in ([("c", "a"), ("b", "c")], [("b", "c"), ("c", "a")], [("a", "b"), ("b", "c")]):
        union = _Union(roster_ciks=frozenset())
        for left, right in pairs:
            union.union(left, right, source="test")
        ids.append({entity_id_for(members) for members in union.groups().values()})
    assert ids[0] == ids[1] == ids[2] == {"Ea"}

    print("\n=== SANITY CHECK: id stability under merge order ===")
    print(f"  three different merge orders -> {ids[0]}")
    print("  OK: the oldest CIK roots the group however the merges arrive")


def test_a_merge_joining_two_universe_tickers_is_refused():
    """The one failure that corrupts rather than drops. `ITW`/`NTRS` is the live instance."""
    union = _Union(roster_ciks=frozenset({"0000049826", "0000073124"}))
    assert union.union("0000049826", "0000073124", source="owner_overlap[NTRS]", confidence=0.0397) is False
    assert union.find("0000049826") != union.find("0000073124")
    assert len(union.blocked) == 1 and union.blocked[0][2] == "owner_overlap[NTRS]"
    assert union.union("0000073124", "0001063537", source="owner_overlap[NTRS]") is True

    print("\n=== SANITY CHECK: the two-universe-tickers refusal ===")
    print(f"  blocked: {union.blocked[0]}")
    print("  OK: two roster CIKs stay in two entities; a non-roster CIK still merges")


def test_score_overlap_is_symmetric_and_empty_safe():
    assert score_overlap(set(), {"a"}) == (0, 0.0)
    assert score_overlap({"a", "b"}, {"b", "c"}) == score_overlap({"b", "c"}, {"a", "b"})
    assert score_overlap({"a", "b"}, {"b", "c"}) == (1, 1 / 3)

    print("\n=== SANITY CHECK: overlap scoring ===")
    print(f"  {{a,b}} vs {{b,c}} -> {score_overlap({'a', 'b'}, {'b', 'c'})}")
    print("  OK: symmetric, and a CIK with no Form 345 owner scores 0 rather than raising")


def test_the_grey_band_is_excluded_with_a_warning_and_a_backlog_row(tmp_path, caplog):
    """`shared > 0` below both thresholds, with no curated row: the CIK stays out, the build goes on."""
    assert classify_overlap(0, 0.0) == "unrelated"
    assert classify_overlap(3, 0.037) == "grey"
    assert classify_overlap(3, OVERLAP_JACCARD_SAME) == "same"
    assert classify_overlap(OVERLAP_SHARED_SAME, 0.001) == "same"

    # COHR's real shape in miniature: 3 owners shared out of 31 and 53, jaccard 0.037
    tenure = _tenure([("AAA", "0000000111", "2006-01-05", "2012-01-01", 700), ("AAA", "0000000222", "2013-01-05", None, 700)])
    owners = {"0000000111": {f"{i:010d}" for i in range(31)}, "0000000222": {f"{i:010d}" for i in range(28, 81)}}
    caplog.set_level(logging.WARNING, logger=lineage_module.__name__)
    build = derive_entity_lineage(tenure, _roster([("AAA", "0000000222")]), _owner_pairs(owners), _config(tmp_path))

    assert "0000000111" not in set(build.rows["cik"])
    grey = build.backlog[build.backlog["kind"].eq("grey_band")]
    assert list(grey["cik"]) == ["0000000111"] and grey["canonical_ticker"].tolist() == ["AAA"]
    assert "grey_band" in caplog.text and "0000000111" in caplog.text
    current = _rows(build, role="symbol", symbol="AAA")
    assert current["cik"].tolist() == ["0000000222"] and current["valid_to"].isna().all()

    print("\n=== SANITY CHECK: grey band ===")
    print(f"  classify(3, 0.037) = {classify_overlap(3, 0.037)}  (COHR's real score)")
    print(f"  backlog: {grey[['kind', 'canonical_ticker', 'cik', 'detail']].to_dict('records')}")
    print("  OK: the grey CIK is excluded and listed with a WARNING; AAA keeps one current row")


def test_candidate_set_is_universe_symbols_plus_roster_ciks():
    tenure = _tenure([("AAA", "0000000111", "2006-01-05", "2012-01-01", 5), ("ZZZ", "0000000999", "2006-01-05", None, 5)])
    roster = _roster([("AAA", "0000000222"), ("BBB", "0000000333")])
    candidates, by_ticker, roster_cik = candidate_ciks(tenure, roster)

    assert candidates == frozenset({"0000000111", "0000000222", "0000000333"})
    assert by_ticker["AAA"] == {"0000000111", "0000000222"}
    assert by_ticker["BBB"] == {"0000000333"}
    assert roster_cik == {"AAA": "0000000222", "BBB": "0000000333"}

    print("\n=== SANITY CHECK: candidate set ===")
    print(f"  candidates={sorted(candidates)}")
    print("  OK: scoped to universe symbols + roster CIKs; every other EDGAR CIK is a singleton by default")


def test_manual_config_rejects_an_undocumented_verdict(tmp_path):
    """Same discipline as the register: an identity decision with no evidence is a guess."""
    (tmp_path / "sec").mkdir()
    target = tmp_path / "sec" / "entity_lineage_manual.json"

    target.write_text(json.dumps({"X": {"same_entity": ["1", "2"], "evidence": "  "}}))
    with pytest.raises(ValueError, match="empty `evidence`"):
        load_manual_lineage(str(tmp_path))
    target.write_text(json.dumps({"X": {"same_entity": ["1"], "evidence": "why"}}))
    with pytest.raises(ValueError, match=">= 2 CIKs"):
        load_manual_lineage(str(tmp_path))
    target.write_text(json.dumps({"X": {"evidence": "why"}}))
    with pytest.raises(ValueError, match="decides nothing"):
        load_manual_lineage(str(tmp_path))
    target.write_text(json.dumps({"X": {"own_entity": ["1004440"], "evidence": "why"}}))
    assert load_manual_lineage(str(tmp_path))["X"]["own_entity"] == ["0001004440"]

    print("\n=== SANITY CHECK: curated-file validation ===")
    print("  empty evidence / 1-CIK same_entity / an entry asserting nothing all RAISE; CIKs zero-padded on load")


def test_manual_tenure_cik_must_resolve_to_its_canonical_entity():
    lineage = pd.DataFrame([{"cik": "0000000001", "entity_id": "E_HOME"}, {"cik": "0000000002", "entity_id": "E_OTHER"}])
    roster = _roster([("AAA", "0000000001")])
    manual = pd.DataFrame([{"canonical_ticker": "AAA", "symbol": "OLD", "issuer_cik": "0000000002"}])
    with pytest.raises(ManualTenureEntityError, match="AAA/OLD/0000000002"):
        validate_manual_tenure_entities(manual, lineage, roster)
    manual.loc[0, "issuer_cik"] = "0000000001"
    validate_manual_tenure_entities(manual, lineage, roster)
    print("\n=== SANITY CHECK: manual CIK/entity cross-check ===")
    print("  unrelated OLD CIK raises; the roster entity CIK passes")


def test_new_older_cik_rekey_is_detected():
    existing = pd.DataFrame([{"cik": "0000000200", "entity_id": "E0000000200"}, {"cik": "0000000300", "entity_id": "E0000000200"}])
    candidate = pd.DataFrame(
        [
            {"cik": "0000000100", "entity_id": "E0000000100"},
            {"cik": "0000000200", "entity_id": "E0000000100"},
            {"cik": "0000000300", "entity_id": "E0000000100"},
        ]
    )
    impacts = detect_older_cik_rekeys(existing, candidate)
    assert impacts == [
        {
            "old_entity_id": "E0000000200",
            "new_entity_id": "E0000000100",
            "existing_ciks": ["0000000200", "0000000300"],
            "new_older_ciks": ["0000000100"],
            "candidate_ciks": ["0000000100", "0000000200", "0000000300"],
        }
    ]
    assert detect_older_cik_rekeys(existing, existing.copy()) == []
    print("\n=== SANITY CHECK: older-CIK rekey detection ===")
    print("  adding 0000000100 would rename E0000000200 -> E0000000100; named as an impact")


# --------------------------------------------------------------------------- #
# P3: the dated verdict                                                         #
# --------------------------------------------------------------------------- #
def test_an_older_cik_rekey_is_excluded_backlogged_and_writes_no_file(tmp_path, caplog):
    """P7/P9: a rekey never stops the build and writes no impact file; an exact approval applies it."""
    tenure = _tenure([("AAA", "0000000100", "2006-01-05", "2012-01-01", 400), ("AAA", "0000000200", "2012-06-01", None, 400)])
    owners = {"0000000100": {f"{i:010d}" for i in range(20)}, "0000000200": {f"{i:010d}" for i in range(20)}}
    existing = pd.DataFrame([{"cik": "0000000200", "entity_id": "E0000000200"}])
    config = _config(tmp_path)
    caplog.set_level(logging.WARNING, logger=lineage_module.__name__)

    build = derive_entity_lineage(tenure, _roster([("AAA", "0000000200")]), _owner_pairs(owners), config, existing=existing)
    assert set(build.rows["entity_id"]) == {"E0000000200"} and "0000000100" not in set(build.rows["cik"])
    rekey = build.backlog[build.backlog["kind"].eq("rekey")]
    assert rekey["cik"].tolist() == ["0000000100"] and "E0000000200:E0000000100" in rekey["detail"].iloc[0]
    assert "excluded" in caplog.text
    assert not (tmp_path / "reports").exists() and not list(tmp_path.rglob("identity-rekey-impact.json"))

    approved = derive_entity_lineage(
        tenure,
        _roster([("AAA", "0000000200")]),
        _owner_pairs(owners),
        config,
        existing=existing,
        approved_rekeys=frozenset({("E0000000200", "E0000000100")}),
    )
    assert set(approved.rows["entity_id"]) == {"E0000000100"} and approved.backlog[approved.backlog["kind"].eq("rekey")].empty

    print("\n=== SANITY CHECK: older-CIK rekey exclusion ===")
    print(f"  unapproved: entity stays E0000000200, backlog {rekey[['kind', 'cik']].to_dict('records')}, no file written")
    print("  approved E0000000200:E0000000100: 0000000100 joins and the entity is renamed")
    print("  OK: the stop became an exclusion; the CLI approval path still applies a rekey")


def test_open_starts_are_the_sentinel_and_register_segments_become_windows(tmp_path):
    register = {
        "AAA": {
            "kind": "reorganisation",
            "segments": [
                {"cik": "100", "valid_to": "2015-01-01", "evidence": "old registrant"},
                {"cik": "200", "valid_from": "2015-01-01", "evidence": "new holding company"},
            ],
        }
    }
    tenure = _tenure(
        [("AAA", "0000000100", "2006-01-05", None, 300), ("AAA", "0000000200", "2015-01-02", None, 90), ("BBB", "0000000300", "2006-01-05", None, 9)]
    )
    build = derive_entity_lineage(
        tenure, _roster([("AAA", "0000000200"), ("BBB", "0000000300")]), _owner_pairs({}), _config(tmp_path, register=register)
    )

    windows = _rows(build, role="cik_window").set_index("cik")
    assert windows.loc["0000000100", "valid_from"] == L.SENTINEL_START == pd.Timestamp("1900-01-01")
    assert windows.loc["0000000100", "valid_to"] == pd.Timestamp("2015-01-01")
    assert windows.loc["0000000200", "valid_from"] == pd.Timestamp("2015-01-01") and pd.isna(windows.loc["0000000200", "valid_to"])
    assert windows.loc["0000000300", "valid_from"] == L.SENTINEL_START and pd.isna(windows.loc["0000000300", "valid_to"])
    assert windows.loc["0000000100", "status"] == "curated" and windows.loc["0000000300", "sources"] == "roster"
    assert build.rows["valid_from"].notna().all() and build.rows["evidence"].str.len().gt(0).all()
    # an open symbol row on a CIK whose window closed is not current
    old_symbol = _rows(build, role="symbol", cik="0000000100")
    assert old_symbol["valid_to"].notna().all()

    print("\n=== SANITY CHECK: sentinel and register windows ===")
    print(build.rows[["canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources"]].to_string(index=False))
    print("  OK: open starts are 1900-01-01 (PK-safe), open ends NULL, every row carries evidence")


def test_a_cik_in_two_entities_raises():
    frame = pd.DataFrame({"cik": ["0000000100", "0000000100", "0000000200"], "entity_id": ["E0000000100", "E0000000050", "E0000000100"]})
    with pytest.raises(L.CikInTwoEntitiesError, match="0000000100"):
        L.check_one_entity_per_cik(frame)
    L.check_one_entity_per_cik(frame.iloc[[0, 2]])
    print("\n=== SANITY CHECK: one entity per CIK ===")
    print("  CIK 0000000100 under E0000000100 and E0000000050 raises CikInTwoEntitiesError; one entity passes")


def test_a_symbol_handoff_seen_in_both_sources_joins_the_predecessor(tmp_path):
    tenure, dei = _handoff_evidence()
    config = _config(tmp_path)
    build = derive_entity_lineage(tenure, _roster([("ABC", "0000000200")]), _owner_pairs({}), config, dei=dei)
    predecessor = _rows(build, cik="0000000100")
    assert set(predecessor["entity_id"]) == {"E0000000100"} and set(predecessor["oracle"]) == {"symbol_handoff"}
    assert set(_rows(build, cik="0000000200")["entity_id"]) == {"E0000000100"}

    one_source = derive_entity_lineage(
        tenure, _roster([("ABC", "0000000200")]), _owner_pairs({}), config, dei=dei[dei["issuer_cik"].eq("0000000200")]
    )
    assert "0000000100" not in set(one_source.rows["cik"])

    print("\n=== SANITY CHECK: symbol handoff ===")
    print(f"  ABC 100 -> 200 on Forms 3/4/5 and dei: joined via {predecessor['oracle'].iloc[0]}")
    print("  without the predecessor's own dei rows: not joined (one source is not a handoff)")
    print("  OK: the handoff needs both sources on both CIKs")


def test_automatic_windows_when_both_sources_agree(tmp_path):
    tenure, dei = _handoff_evidence()
    config = _config(tmp_path)
    auto = derive_entity_lineage(tenure, _roster([("ABC", "0000000200")]), _owner_pairs({}), config, dei=dei, auto_windows=True)
    windows = _rows(auto, role="cik_window").set_index("cik")
    assert windows.loc["0000000100", "valid_from"] == L.SENTINEL_START and windows.loc["0000000100", "valid_to"] == pd.Timestamp("2015-07-01")
    assert windows.loc["0000000200", "valid_from"] == pd.Timestamp("2015-07-01") and pd.isna(windows.loc["0000000200", "valid_to"])
    assert set(windows["status"]) == {"corroborated"} and set(windows["sources"]) == {"dei,form345"}
    assert auto.backlog[auto.backlog["kind"].eq("multi_cik_no_window")].empty

    off = derive_entity_lineage(tenure, _roster([("ABC", "0000000200")]), _owner_pairs({}), config, dei=dei, auto_windows=False)
    assert _rows(off, role="cik_window")["cik"].tolist() == ["0000000200"]
    assert _rows(off, role="cik_event")["cik"].tolist() == ["0000000100"]
    pending = off.backlog[off.backlog["kind"].eq("multi_cik_no_window")]
    assert len(pending) == 1 and "suggested chain" in pending["detail"].iloc[0]

    print("\n=== SANITY CHECK: D3 automatic windows ===")
    print(windows[["valid_from", "valid_to", "status", "sources"]].to_string())
    print(f"  disabled: roster window only, predecessor event-only, backlog: {pending['detail'].iloc[0]}")
    print("  OK: agreeing sources date the seam; with D3 off the P4 behaviour applies")


def test_automatic_windows_are_live_and_a_register_entry_overrides_them(tmp_path):
    """P36: agreeing sources date a chain by default; a register entry for the ticker wins over the evidence."""
    tenure, dei = _handoff_evidence()
    roster = _roster([("ABC", "0000000200")])
    live = derive_entity_lineage(tenure, roster, _owner_pairs({}), _config(tmp_path), dei=dei)
    windows = _rows(live, role="cik_window").set_index("cik")
    assert sorted(windows.index) == ["0000000100", "0000000200"], "automatic windows are on without an explicit flag"
    assert windows.loc["0000000200", "valid_from"] == pd.Timestamp("2015-07-01")
    assert windows["evidence"].str.contains("automatic window").all()

    segments = [{"cik": "100", "valid_to": "2016-01-04", "evidence": "old"}, {"cik": "200", "valid_from": "2016-01-04", "evidence": "new"}]
    register = {"ABC": {"kind": "reorganisation", "segments": segments}}
    curated = derive_entity_lineage(tenure, roster, _owner_pairs({}), _config(tmp_path / "register", register=register), dei=dei)
    register_windows = _rows(curated, role="cik_window").set_index("cik")
    assert set(register_windows["sources"]) == {"register"}
    assert register_windows.loc["0000000200", "valid_from"] == pd.Timestamp("2016-01-04")
    assert not register_windows["evidence"].str.contains("automatic window").any()

    print("\n=== SANITY CHECK: P36 automatic windows live, register first ===")
    print(f"  evidence only: 100 -> 200 on {windows.loc['0000000200', 'valid_from'].date()} (dei + Forms 3/4/5)")
    print(f"  with a register entry dated 2016-01-04: windows {register_windows['sources'].unique().tolist()}, seam 2016-01-04")
    print("  OK: the automatic rule fills only what the register leaves undeclared")


def test_every_automatic_window_is_listed_for_review(tmp_path):
    """P36: an automatic chain is an information item (`automatic_cik_window`) carrying the switch evidence."""
    tenure, dei = _handoff_evidence()
    build = derive_entity_lineage(tenure, _roster([("ABC", "0000000200")]), _owner_pairs({}), _config(tmp_path), dei=dei, auto_windows=True)
    flags = identity_flags(build.rows, cik_activity(pd.concat([tenure, dei], ignore_index=True)))
    auto = flags[flags["kind"].eq("automatic_cik_window")]
    assert "automatic_cik_window" in KIND_ORDER
    assert len(auto) == 1 and not bool(auto["action"].iloc[0]) and auto["ticker"].iloc[0] == "ABC"
    assert auto["ciks"].iloc[0] == "0000000100,0000000200"
    text = auto["evidence"].iloc[0]
    assert "0000000200 from 2015-07-01" in text and "dei" in text and "form345" in text
    assert "registrant_cutover.json" in auto["config_file"].iloc[0]

    off = derive_entity_lineage(tenure, _roster([("ABC", "0000000200")]), _owner_pairs({}), _config(tmp_path), dei=dei, auto_windows=False)
    assert identity_flags(off.rows, cik_activity(pd.concat([tenure, dei], ignore_index=True)))["kind"].ne("automatic_cik_window").all()

    print("\n=== SANITY CHECK: automatic windows listed for review ===")
    print(f"  [info] {auto['kind'].iloc[0]} {auto['ticker'].iloc[0]} ({auto['ciks'].iloc[0]}): {text[:160]}")
    print("  OK: each automatic chain is one info item; none when the rule is off")


def test_disagreeing_sources_abstain_and_keep_the_roster_window(tmp_path):
    """AC-019b: Forms 3/4/5 switch in 2015, cover pages in 2018 -> no window, P4 behaviour, backlog."""
    tenure, dei = _handoff_evidence(dei_successor_first="2018-03-01", dei_predecessor_last="2018-02-21")
    owners = {"0000000100": {f"{i:010d}" for i in range(10)}, "0000000200": {f"{i:010d}" for i in range(10)}}
    build = derive_entity_lineage(tenure, _roster([("ABC", "0000000200")]), _owner_pairs(owners), _config(tmp_path), dei=dei, auto_windows=True)
    assert set(_rows(build, cik="0000000100")["oracle"]) == {"owner_overlap"}
    assert _rows(build, role="cik_window")["cik"].tolist() == ["0000000200"]
    assert _rows(build, role="cik_event")["cik"].tolist() == ["0000000100"]
    pending = build.backlog[build.backlog["kind"].eq("multi_cik_no_window")]
    assert len(pending) == 1 and "sources disagree" in pending["detail"].iloc[0]

    print("\n=== SANITY CHECK: abstention ===")
    print(f"  backlog: {pending['detail'].iloc[0][:160]}")
    print("  OK: disagreeing sources produce no window; the roster CIK alone lists consolidating forms")


def test_manual_tenure_outranks_derived_evidence(tmp_path):
    """IR/TT: the curated `IR` interval on TT's CIK clips today's IR holder instead of making a conflict."""
    tenure = pd.concat(
        [
            _tenure([("IR", "0000000300", "2009-07-09", "2020-03-02", 0)], source="manual"),
            _tenure(
                [
                    ("IR", "0000000300", "2009-07-09", "2020-03-06", 500),
                    ("TT", "0000000300", "2020-03-02", None, 100),
                    ("IR", "0000000400", "2020-02-25", None, 200),
                ]
            ),
        ],
        ignore_index=True,
    )
    build = derive_entity_lineage(tenure, _roster([("TT", "0000000300"), ("IR", "0000000400")]), _owner_pairs({}), _config(tmp_path))
    curated = _rows(build, role="symbol", symbol="IR", cik="0000000300")
    assert len(curated) == 1 and curated["status"].iloc[0] == "curated" and curated["sources"].iloc[0] == "manual"
    assert curated["valid_to"].iloc[0] == pd.Timestamp("2020-03-02")
    today = _rows(build, role="symbol", symbol="IR", cik="0000000400")
    assert today["valid_from"].iloc[0] == pd.Timestamp("2020-03-02") and pd.isna(today["valid_to"].iloc[0])
    assert today["status"].iloc[0] == "single_source" and build.backlog[build.backlog["kind"].eq("conflict")].empty

    print("\n=== SANITY CHECK: manual over derived ===")
    print(_rows(build, role="symbol", symbol="IR")[["canonical_ticker", "cik", "valid_from", "valid_to", "status", "sources"]].to_string(index=False))
    print("  OK: the manual interval replaces its derived twin and clips the other holder; no conflict")


def test_reuse_within_ninety_days_is_a_conflict_and_later_reuse_resolves_by_date(tmp_path):
    tenure = _tenure(
        [
            ("AAA", "0000000100", "2006-01-05", None, 100),
            ("XYZ", "0000000100", "2010-01-01", "2015-01-01", 50),
            ("XYZ", "0000000900", "2015-02-01", None, 40),
            ("QQQ", "0000000100", "2010-01-01", "2015-01-01", 50),
            ("QQQ", "0000000901", "2016-01-01", None, 40),
        ]
    )
    build = derive_entity_lineage(tenure, _roster([("AAA", "0000000100")]), _owner_pairs({}), _config(tmp_path))
    assert _rows(build, role="symbol", symbol="XYZ")["status"].tolist() == ["conflict"]
    assert _rows(build, role="symbol", symbol="QQQ")["status"].tolist() == ["single_source"]
    conflicts = build.backlog[build.backlog["kind"].eq("conflict")]
    assert conflicts["symbol"].tolist() == ["XYZ"] and conflicts["canonical_ticker"].tolist() == ["AAA"]
    other = build.holders[build.holders["cik"].eq("0000000900")]
    assert other["status"].tolist() == ["conflict"] and other["canonical_ticker"].isna().all()

    print("\n=== SANITY CHECK: reuse conflicts ===")
    print("  XYZ reused 31 days later -> both intervals `conflict`, backlogged; QQQ reused a year later -> resolved by date")


def test_a_one_source_typo_next_to_another_holder_is_noise(tmp_path):
    """ALB/AB: four Form 4s typed `AB` under Albemarle while AllianceBernstein held it."""
    tenure = _tenure(
        [
            ("ALB", "0000915913", "2006-01-05", None, 500),
            ("AB", "0000915913", "2009-02-11", "2010-05-20", 4),
            ("AB", "0000825313", "2006-01-05", None, 900),
        ]
    )
    dei = _tenure([("AB", "0000825313", "2009-08-01", "2026-08-01", 70)], source="dei")
    build = derive_entity_lineage(tenure, _roster([("ALB", "0000915913")]), _owner_pairs({}), _config(tmp_path), dei=dei)
    assert _rows(build, role="symbol", symbol="AB")["status"].tolist() == ["noise"]
    assert build.backlog[build.backlog["kind"].eq("conflict")].empty
    assert build.holders.loc[build.holders["cik"].eq("0000825313"), "status"].tolist() == ["corroborated"]
    print("\n=== SANITY CHECK: noise ===")
    print("  ALB/AB (4 filings, Forms 3/4/5 only, next to AllianceBernstein) -> noise; no conflict raised")


def test_a_typo_matching_a_symbol_another_company_held_years_apart_is_noise(tmp_path):
    """AC-016 ALGN/ALGM: one Form 4 typed `ALGM` in 2012; Allegro MicroSystems has held `ALGM` since 2020."""
    tenure = _tenure(
        [
            ("ALGN", "0001097149", "2006-01-05", None, 500),
            ("ALGM", "0001097149", "2012-07-24", "2012-07-25", 1),
            ("ALGM", "0000866291", "2020-10-29", None, 80),
        ]
    )
    dei = _tenure([("ALGM", "0000866291", "2020-11-10", "2026-08-01", 20)], source="dei")
    build = derive_entity_lineage(tenure, _roster([("ALGN", "0001097149")]), _owner_pairs({}), _config(tmp_path), dei=dei)
    assert _rows(build, role="symbol", symbol="ALGM")["status"].tolist() == ["noise"]
    print("\n=== SANITY CHECK: noise across years ===")
    print("  ALGN/ALGM (1 filing in 2012, Allegro holds ALGM from 2020 on two sources) -> noise")


def test_a_subsidiary_cover_symbol_does_not_put_the_roster_ticker_in_conflict(tmp_path):
    """A co-registrant subsidiary (Ford Credit) types the parent's `F` on its own cover pages; the roster row anchors `F`."""
    tenure = _tenure([("F", "0000037996", "2006-01-03", None, 2300)])
    dei = _tenure([("F", "0000037996", "2010-11-08", "2026-07-30", 230), ("F", "0000038009", "2019-07-25", "2026-08-14", 150)], source="dei")
    build = derive_entity_lineage(tenure, _roster([("F", "0000037996")]), _owner_pairs({}), _config(tmp_path), dei=dei)
    current = _rows(build, role="symbol", symbol="F")
    assert current["status"].tolist() == ["corroborated"] and current["valid_to"].isna().all()
    assert build.holders.loc[build.holders["cik"].eq("0000038009"), "status"].tolist() == ["noise"]
    assert build.backlog[build.backlog["kind"].eq("conflict")].empty
    print("\n=== SANITY CHECK: roster anchor ===")
    print("  Ford Credit's dei `F` (150 filings) lies inside Ford's roster interval -> superseded (noise); F stays corroborated")


def test_every_roster_ticker_gets_one_current_row_even_with_no_filing_spelling(tmp_path):
    """AC-012 on the BF-B shape: filers type `BFA`/`BFB`, the roster says `BF-B`."""
    tenure = _tenure([("BFA", "0000014693", "2006-01-05", None, 300), ("BFB", "0000014693", "2006-01-05", None, 300)])
    config = _config(tmp_path, manual={"_d19_allowlist": {"BF-B": "filers type BFA/BFB"}})
    build = derive_entity_lineage(tenure, _roster([("BF-B", "0000014693")]), _owner_pairs({}), config)
    current = _rows(build, role="symbol", symbol="BF-B")
    assert len(current) == 1 and current["cik"].iloc[0] == "0000014693" and pd.isna(current["valid_to"].iloc[0])
    assert current["sources"].iloc[0] == "roster" and current["valid_from"].iloc[0] == pd.Timestamp("2006-01-05")
    print("\n=== SANITY CHECK: roster row ===")
    print(current[["symbol", "cik", "valid_from", "valid_to", "status", "sources"]].to_string(index=False))
    print("  OK: one current BF-B row on the roster CIK, dated from the CIK's first filing")


def test_a_d19_disagreement_still_stops_the_build(tmp_path):
    """AC-007: the roster CIK and the dominant filer of the ticker name different entities."""
    tenure = _tenure([("AAA", "0000000900", "2006-01-05", None, 800), ("AAA", "0000000100", "2006-01-05", "2008-01-01", 3)])
    with pytest.raises(L.UniverseEntityDisagreementError, match="AAA"):
        derive_entity_lineage(tenure, _roster([("AAA", "0000000100")]), _owner_pairs({}), _config(tmp_path))
    allowed = _config(tmp_path / "allowed", manual={"_d19_allowlist": {"AAA": "evidenced"}})
    derive_entity_lineage(tenure, _roster([("AAA", "0000000100")]), _owner_pairs({}), allowed)
    print("\n=== SANITY CHECK: D19 ===")
    print("  unlisted disagreement raises UniverseEntityDisagreementError; an allow-list entry clears it")


def test_two_builds_with_a_pinned_timestamp_are_byte_identical(tmp_path):
    """AC-015: same inputs, same stored table, pinned build time -> identical bytes."""
    tenure, dei = _handoff_evidence()
    config = _config(tmp_path)
    pinned = pd.Timestamp("2026-10-03 12:00:00")
    args = (tenure, _roster([("ABC", "0000000200")]), _owner_pairs({}), config)
    first = derive_entity_lineage(*args, dei=dei, built_at=pinned).rows
    second = derive_entity_lineage(*args, dei=dei, built_at=pinned).rows
    assert first.to_csv(index=False).encode() == second.to_csv(index=False).encode()
    again = derive_entity_lineage(*args, dei=dei, existing=first, built_at=pd.Timestamp("2026-10-04")).rows
    assert again.to_csv(index=False).encode() == first.to_csv(index=False).encode()
    print("\n=== SANITY CHECK: determinism ===")
    print(f"  {len(first)} rows; two pinned builds byte-identical; a later build over the stored rows keeps scope_changed_at")


def test_scope_changed_at_moves_only_for_tickers_whose_scope_changed(tmp_path):
    config = _config(tmp_path)
    roster = _roster([("AAA", "0000000100"), ("BBB", "0000000200")])
    tenure = _tenure([("AAA", "0000000100", "2006-01-05", None, 300), ("BBB", "0000000200", "2006-01-05", None, 300)])
    t1, t2, t3 = (pd.Timestamp(f"2026-10-0{d} 01:00:00") for d in (1, 2, 3))
    first = derive_entity_lineage(tenure, roster, _owner_pairs({}), config, built_at=t1).rows
    assert set(first["scope_changed_at"]) == {t1}
    second = derive_entity_lineage(tenure, roster, _owner_pairs({}), config, existing=first, built_at=t2).rows
    assert set(second["scope_changed_at"]) == {t1}

    grown = pd.concat([tenure, _tenure([("AAA", "0000000150", "2006-01-05", "2006-06-01", 40)])], ignore_index=True)
    owners = {"0000000150": {f"{i:010d}" for i in range(10)}, "0000000100": {f"{i:010d}" for i in range(10)}}
    third = derive_entity_lineage(grown, roster, _owner_pairs(owners), config, existing=first, built_at=t3).rows
    stamps = third.groupby("canonical_ticker")["scope_changed_at"].agg(set).to_dict()
    assert stamps == {"AAA": {t3}, "BBB": {t1}}
    print("\n=== SANITY CHECK: scope_changed_at ===")
    print(f"  unchanged rebuild keeps {t1}; AAA gains CIK 0000000150 -> {t3}; BBB stays {t1}")


def test_register_reproduction_reports_reproduced_abstained_and_contradicted(tmp_path):
    """AC-018 on known truth: the automatic rule against three register seams."""
    tenure = _tenure(
        [
            ("AAA", "0000000100", "2006-01-05", "2015-07-01", 300),
            ("AAA", "0000000200", "2015-07-01", None, 100),
            ("BBB", "0000000300", "2006-01-05", "2015-07-01", 300),
            ("BBB", "0000000400", "2015-07-01", None, 100),
            ("CCC", "0000000500", "2006-01-05", "2015-07-01", 300),
            ("CCC", "0000000600", "2015-07-01", None, 100),
        ]
    )
    dei = _tenure(
        [
            ("AAA", "0000000100", "2010-03-01", "2015-06-16", 20),
            ("AAA", "0000000200", "2015-07-10", "2024-11-02", 30),
            ("BBB", "0000000300", "2010-03-01", "2015-06-16", 20),
            ("BBB", "0000000400", "2015-07-10", "2024-11-02", 30),
        ],
        source="dei",
    )

    def chain(old: str, new: str, boundary: str) -> dict:
        return {
            "kind": "reorganisation",
            "segments": [{"cik": old, "valid_to": boundary, "evidence": "old"}, {"cik": new, "valid_from": boundary, "evidence": "new"}],
        }

    register = {"AAA": chain("100", "200", "2015-07-01"), "BBB": chain("300", "400", "2013-01-01"), "CCC": chain("500", "600", "2015-07-01")}
    roster = _roster([("AAA", "0000000200"), ("BBB", "0000000400"), ("CCC", "0000000600")])
    build = derive_entity_lineage(tenure, roster, _owner_pairs({}), _config(tmp_path, register=register), dei=dei)
    verdicts = dict(zip(build.reproduction["ticker"], build.reproduction["verdict"], strict=True))
    assert verdicts == {"AAA": "reproduced", "BBB": "contradicted", "CCC": "abstained"}
    print("\n=== SANITY CHECK: register reproduction ===")
    print(build.reproduction[["ticker", "register_boundary", "auto_boundary", "verdict", "detail"]].to_string(index=False))
    print("  OK: agreeing sources reproduce, a seam outside the evidence window contradicts, one source abstains")


def test_build_writes_through_the_store_and_skips_an_unchanged_rebuild(tmp_path, monkeypatch, sqlite_store, caplog):
    """End to end on SQLite: `dei` read from `symbol_tenure`, rows written, a second build is a no-op."""
    tenure, dei = _handoff_evidence()
    sqlite_store.save(Tables.sp500_tickers, _roster([("ABC", "0000000200")]))
    sqlite_store.save(Tables.symbol_tenure, dei.assign(evidence_period="2025q1"))
    monkeypatch.setattr(
        lineage_module, "load_manual_symbol_tenure", lambda *a, **k: pd.DataFrame(columns=["canonical_ticker", "symbol", "issuer_cik"])
    )
    context: Any = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.entity_lineage.store"), config_dir=_config(tmp_path))
    caplog.set_level(logging.INFO)

    written = lineage_module.build_entity_lineage(context, tenure, _owner_pairs({}), context.config_dir, built_at=pd.Timestamp("2026-10-03 12:00:00"))
    stored = sqlite_store.load(Tables.entity_lineage, project=True)
    assert len(stored) == len(written) and set(stored["oracle"]) >= {"symbol_handoff"}
    lineage_module.build_entity_lineage(context, tenure, _owner_pairs({}), context.config_dir, built_at=pd.Timestamp("2026-10-04 12:00:00"))
    assert f"entity_lineage: unchanged ({len(written)} row(s)); replace skipped" in caplog.text
    print("\n=== SANITY CHECK: store round trip ===")
    print(f"  {len(written)} rows written to SQLite; the rebuild one day later skipped the replace")


# --------------------------------------------------------------------------- #
# build: logging and replace skip                                               #
# --------------------------------------------------------------------------- #
def _empty_build(rows: pd.DataFrame) -> Any:
    return L.LineageBuild(
        rows=rows, blocked=pd.DataFrame(), backlog=pd.DataFrame(columns=list(L.BACKLOG_COLUMNS)), holders=pd.DataFrame(), reproduction=pd.DataFrame()
    )


def test_entity_lineage_build_logs_changed_ciks_and_affected_tickers(monkeypatch, caplog):
    columns = ["cik", "entity_id", "role", "symbol", "valid_from", "valid_to"]
    old = pd.DataFrame([("0000000001", "E0000000001", "cik_window", "", pd.Timestamp("1900-01-01"), pd.NaT)], columns=columns)
    new = pd.DataFrame([("0000000001", "E0000000002", "cik_window", "", pd.Timestamp("1900-01-01"), pd.NaT)], columns=columns)
    roster = _roster([("AAA", "0000000001")])
    tenure = _tenure([("AAA", "0000000001", "2020-01-01", None, 1)])

    class Store:
        def load(self, table, **kwargs):
            if table in (Tables.symbol_tenure, Tables.security_master):  # optional tables not stored yet
                return None
            return {Tables.sp500_tickers: roster, Tables.entity_lineage: old}[table]

        def replace(self, table, frame):
            assert table is Tables.entity_lineage and frame.equals(new)
            return len(frame)

    monkeypatch.setattr(lineage_module, "derive_entity_lineage", lambda *args, **kwargs: _empty_build(new))
    monkeypatch.setattr(lineage_module, "validate_manual_tenure_entities", lambda *args, **kwargs: None)
    monkeypatch.setattr(lineage_module, "load_manual_symbol_tenure", lambda *a, **k: pd.DataFrame({"canonical_ticker": ["AAA"]}))
    context: Any = SimpleNamespace(store=Store(), log=logging.getLogger("test.entity_lineage"))
    caplog.set_level(logging.INFO, logger="test.entity_lineage")

    lineage_module.build_entity_lineage(context, tenure, _owner_pairs({}), CONFIG_DIR)
    assert "1 changed CIK assignment(s): 0000000001" in caplog.text
    assert "affected current ticker(s): AAA" in caplog.text
    print("\n=== SANITY CHECK: entity-lineage refresh visibility ===")
    print("  changed CIK 0000000001 and affected current ticker AAA are named")


def test_the_build_logs_the_manual_decision_block(monkeypatch, caplog):
    """P15: a sequential uncurated pair and a grey-band backlog item reach one WARNING block from the build."""
    columns = ["entity_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources", "oracle"]
    rows = pd.DataFrame(
        [
            ("E1", "AAA", "0000000002", "cik_window", "", pd.Timestamp("1900-01-01"), pd.NaT, "single_source", "roster", "roster"),
            ("E1", "AAA", "0000000001", "cik_event", "", pd.Timestamp("1900-01-01"), pd.NaT, "single_source", "form345", "owner_overlap"),
        ],
        columns=columns,
    )
    backlog = pd.DataFrame([("grey_band", "AAA", "E1", "0000000009", "AAA", "owner overlap with 0000000002")], columns=list(L.BACKLOG_COLUMNS))
    build = L.LineageBuild(rows=rows, blocked=pd.DataFrame(), backlog=backlog, holders=pd.DataFrame(), reproduction=pd.DataFrame())
    tenure = _tenure([("AAA", "0000000001", "2006-01-03", "2012-06-30", 9), ("AAA", "0000000002", "2014-01-02", None, 9)])

    class Store:
        def load(self, table, **kwargs):
            return _roster([("AAA", "0000000002")]) if table is Tables.sp500_tickers else None

        def replace(self, table, frame):
            return len(frame)

    monkeypatch.setattr(lineage_module, "derive_entity_lineage", lambda *args, **kwargs: build)
    monkeypatch.setattr(lineage_module, "validate_manual_tenure_entities", lambda *args, **kwargs: None)
    monkeypatch.setattr(lineage_module, "load_manual_symbol_tenure", lambda *a, **k: pd.DataFrame({"canonical_ticker": ["AAA"]}))
    context: Any = SimpleNamespace(store=Store(), log=logging.getLogger("test.entity_lineage.flags"))
    caplog.set_level(logging.INFO, logger="test.entity_lineage.flags")

    lineage_module.build_entity_lineage(context, tenure, _owner_pairs({}), CONFIG_DIR)
    blocks = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "IDENTITY ITEMS NEEDING A MANUAL DECISION" in r.getMessage()]
    assert len(blocks) == 1 and "missing_cutover AAA" in blocks[0] and "grey_band AAA" in blocks[0], blocks
    print("\n=== SANITY CHECK: build flag block ===")
    print(blocks[0])
    print("  OK: the build emits the same block the validator does, with the build-only grey band")


# --------------------------------------------------------------------------- #
# Real data: the live register + curated files against the live tables         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def live_build() -> Any:
    if not CACHE.exists() or len(list(CACHE.glob("*.zip"))) < 40:
        pytest.skip(f"no cached Form 345 quarters under {CACHE}")
    from src.context import get_config_context

    try:
        _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
        tenure = context.store.load(Tables.symbol_tenure, project=True, where={"source": ["form345", "manual"]}, optional=True)
        roster = context.store.load(Tables.sp500_tickers, optional=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable ({type(exc).__name__})")
    if tenure is None or roster is None or tenure.empty or roster.empty:
        pytest.skip("symbol_tenure / sp500_tickers empty -- run `identity-tables` first")
    assert tenure is not None and roster is not None
    return derive_entity_lineage(tenure, roster, scan_form345_cache(CACHE).owner_pairs, CONFIG_DIR)


@pytest.fixture(scope="module")
def live_lineage(live_build: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    return live_build.rows, live_build.blocked


def test_no_entity_holds_two_universe_tickers(live_lineage):
    """The invariant the whole design rests on, asserted on the live tables."""
    lineage, blocked = live_lineage
    from src.context import get_config_context

    _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    roster = context.store.load(Tables.sp500_tickers)
    assert roster is not None
    entity = dict(zip(lineage["cik"].astype(str), lineage["entity_id"].astype(str), strict=False))
    per_entity: dict[str, list[str]] = {}
    for ticker, cik in zip(roster["ticker"].astype(str), roster["cik"].astype("string").str.zfill(10), strict=False):
        per_entity.setdefault(entity.get(cik, "E" + str(cik)), []).append(ticker)
    collisions = {e: t for e, t in per_entity.items() if len(t) > 1}
    assert not collisions, f"entities holding two universe tickers: {collisions}"
    assert len(per_entity) == len(roster)
    print("\n=== SANITY CHECK: one entity per universe ticker ===")
    print(f"  {len(roster)} tickers -> {len(per_entity)} entities, 0 collisions; merges refused: {len(blocked)}")


def test_the_worked_cases_resolve_as_the_plan_specifies(live_lineage):
    """Predecessors join their successor; reuses stay separate; `IR` falls out of `TT`."""
    lineage, _ = live_lineage
    entity = dict(zip(lineage["cik"].astype(str), lineage["entity_id"].astype(str), strict=False))
    oracle = dict(zip(lineage["cik"].astype(str), lineage["oracle"].astype(str), strict=False))

    def func_e(c):
        return entity.get(c, "E" + c)

    same = {
        "DD/DuPont E I": ("0000030554", "0001666700"),
        "CB/Chubb Corp": ("0000020171", "0000896159"),
        "JCI/Johnson Controls Inc": ("0000053669", "0000833444"),
        "ACN/Accenture Ltd": ("0001134538", "0001467373"),
        "AVGO/Avago": ("0001441634", "0001730168"),
        "TT/Trane Inc": ("0000836102", "0001466258"),
        "COHR/Coherent Inc": ("0000021510", "0000820318"),
    }
    apart = {
        "WTW/Weight Watchers": ("0000105319", "0001140536"),
        "COR/CoreSite": ("0001490892", "0001140859"),
        "ECHO/Echo Global": ("0001426945", "0001415404"),
        "MNST/Monster Worldwide": ("0001020416", "0000865752"),
        "AVGO/Avicena": ("0001317092", "0001730168"),
        "CEG/old Constellation": ("0001004440", "0001868275"),
    }
    for label, (pred, succ) in same.items():
        assert func_e(pred) == func_e(succ), f"{label}: {func_e(pred)} != {func_e(succ)}"
    for label, (other, home) in apart.items():
        assert func_e(other) != func_e(home), f"{label}: both resolved to {func_e(other)}"
    assert func_e("0001160497") == func_e("0001466258") == func_e("0000836102")  # TT's entity
    assert func_e("0001160497") != func_e("0001699150")  # not today's IR
    assert oracle.get("0001004440") == "manual"
    print("\n=== SANITY CHECK: the worked identity cases ===")
    print("  OK: every verdict matches the plan, and there is no IR-specific rule anywhere")


#: AC-002, the traded view: ticker -> ({window CIK: (valid_from, valid_to)}, {event-only CIKs}).
TRADED_VIEW = {
    "PLD": ({"0001045609": ("1900-01-01", None)}, {"0000899881"}),
    "JCI": ({"0000833444": ("1900-01-01", None)}, {"0000053669"}),
    "DD": ({"0000029915": ("1900-01-01", "2017-08-31"), "0001666700": ("2017-08-31", None)}, {"0000030554"}),
    "DOW": ({"0001751788": ("1900-01-01", None)}, set()),
}


def test_the_four_realigned_entities_follow_the_traded_security(live_build):
    """AC-002 on real inputs: the security whose prices each ticker carries owns its windows, the acquired targets are
    event-only CIKs declared by the manual (no `multi_cik_no_window`, no `manual_review`), and TDCC is DD's, not DOW's."""
    rows = live_build.rows
    cik_rows = rows[rows["role"].isin(["cik_window", "cik_event"])]
    print("\n=== SANITY CHECK: PLD / JCI / DD / DOW entities on real inputs ===")
    for ticker, (windows, events) in TRADED_VIEW.items():
        mine = cik_rows[cik_rows["canonical_ticker"].eq(ticker)]
        got_windows = {
            r.cik: (str(pd.Timestamp(r.valid_from).date()), None if pd.isna(r.valid_to) else str(pd.Timestamp(r.valid_to).date()))
            for r in mine[mine["role"].eq("cik_window")].itertuples(index=False)
        }
        got_events = set(mine.loc[mine["role"].eq("cik_event"), "cik"])
        print(f"  {ticker:4s} entity {sorted(set(mine['entity_id']))} windows {got_windows} events {sorted(got_events)}")
        assert got_windows == windows, ticker
        assert got_events == events, ticker
        assert set(mine.loc[mine["cik"].isin(events), "oracle"]) <= {"manual"}, ticker
    backlog = live_build.backlog
    stuck = backlog[backlog["kind"].eq("multi_cik_no_window") & backlog["canonical_ticker"].isin(list(TRADED_VIEW))]
    no_activity = pd.DataFrame(columns=["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence"])
    flags = identity_flags(rows, cik_activity(no_activity))
    mine = flags[flags["ticker"].isin(list(TRADED_VIEW))]
    # extra-CIK items; the concurrent-symbol `manual_review` items (symbol_tenure_manual.json) are another question
    review = mine[
        mine["kind"].isin(["manual_review", "missing_cutover", "co_registrant"]) & mine["config_file"].ne("configs/sec/symbol_tenure_manual.json")
    ]
    symbols = mine[mine["config_file"].eq("configs/sec/symbol_tenure_manual.json")]
    print(f"  multi_cik_no_window for the four: {len(stuck)}; extra-CIK flags for the four: {len(review)}")
    print(
        f"  concurrent-symbol items (security_master decides per security): {sorted(symbols['evidence'].str.split(' resolves').str[0] + '->' + symbols['ticker'])}"
    )
    assert stuck.empty and review.empty
    print("  OK: AMB, Tyco, TDCC and Dow Inc own the pre-seam history; old ProLogis, old JCI and old DuPont are cited event-only CIKs.")


def test_the_register_beats_owner_overlap(live_lineage):
    """XOM's successor shell shares zero reporting owners with the predecessor; the register still wins."""
    lineage, _ = live_lineage
    entity = dict(zip(lineage["cik"].astype(str), lineage["entity_id"].astype(str), strict=False))
    oracle = dict(zip(lineage["cik"].astype(str), lineage["oracle"].astype(str), strict=False))
    assert entity["0000034088"] == entity["0002115436"]
    assert oracle["0000034088"] == oracle["0002115436"] == "register"
    register_ciks = {c for entry in load_registrants(CONFIG_DIR).values() for c in entry.all_ciks()}
    assert {c for c, s in oracle.items() if s == "register"} == register_ciks & set(oracle)
    print("\n=== SANITY CHECK: oracle priority ===")
    print("  OK: the curated layer is never overruled by the automatic oracle")


def test_governance_cutover_ciks_are_register_sourced_without_owner_inference():
    """The accepted pairs must resolve through the register even with no owner evidence; JCI's old CIK through the manual."""
    current = {"EVRG": "0001711269", "JCI": "0000833444", "PSKY": "0002041610"}
    pairs = {"EVRG": ("0000054507", current["EVRG"]), "PSKY": ("0000813828", current["PSKY"])}
    build = derive_entity_lineage(
        _tenure([(ticker, cik, "2000-01-01", None, 1) for ticker, cik in current.items()]),
        _roster(list(current.items())),
        _owner_pairs({}),
        CONFIG_DIR,
    )
    lineage = build.rows
    entity = dict(zip(lineage["cik"], lineage["entity_id"], strict=False))
    oracle = dict(zip(lineage["cik"], lineage["oracle"], strict=False))
    for predecessor, successor in pairs.values():
        assert entity[predecessor] == entity[successor]
        assert oracle[predecessor] == oracle[successor] == "register"
    assert entity["0000053669"] == entity["0000833444"] and oracle["0000053669"] == "manual"
    print("\n=== SANITY CHECK: governance lineage comes from the curated files ===")
    for ticker, (predecessor, successor) in pairs.items():
        print(f"  {ticker}: {predecessor} == {successor} via register")
    print(f"  JCI: 0000053669 == 0000833444 via {oracle['0000053669']} (acquired target, traded-security view)")
    print("  OK: every CIK assignment is curated without owner overlap.")


#: ticker: (predecessor, successor, seam) -- each dated from the successor's own 8-K12B / 8-K12G3.
SUCCESSOR_FILINGS = {
    "ACN": ("0001134538", "0001467373", "2009-09-01"),
    "DD": ("0000029915", "0001666700", "2017-08-31"),
    "DUK": ("0000030371", "0001326160", "2006-04-03"),
    "FERG": ("0001832433", "0002011641", "2024-08-01"),
    "MRVL": ("0001058057", "0001835632", "2021-04-20"),
    "ORCL": ("0000777676", "0001341439", "2006-01-31"),
    "PSA": ("0000318380", "0001393311", "2007-06-01"),
    "VMC": ("0000103973", "0001396009", "2007-11-16"),
}


def test_a_register_entry_turns_a_backlogged_multi_cik_entity_into_dated_windows(tmp_path):
    """The live register dates the seams that owner overlap alone leaves as `multi_cik_no_window` backlog."""
    rows, owners = [], {}
    for i, (ticker, (predecessor, successor, seam)) in enumerate(SUCCESSOR_FILINGS.items()):
        rows += [(ticker, predecessor, "2000-01-03", seam, 50), (ticker, successor, seam, None, 50)]
        owners[predecessor] = owners[successor] = {f"{i:04d}{k:06d}" for k in range(10)}
    tenure, roster = _tenure(rows), _roster([(ticker, successor) for ticker, (_, successor, _) in SUCCESSOR_FILINGS.items()])

    before = derive_entity_lineage(tenure, roster, _owner_pairs(owners), _config(tmp_path))
    pending = before.backlog[before.backlog["kind"].eq("multi_cik_no_window")]
    assert sorted(pending["canonical_ticker"]) == sorted(SUCCESSOR_FILINGS)

    after = derive_entity_lineage(tenure, roster, _owner_pairs(owners), CONFIG_DIR)
    windows = _rows(after, role="cik_window").set_index("cik")
    for ticker, (predecessor, successor, seam) in SUCCESSOR_FILINGS.items():
        assert windows.loc[predecessor, "valid_from"] == L.SENTINEL_START and windows.loc[predecessor, "valid_to"] == pd.Timestamp(seam), ticker
        assert windows.loc[successor, "valid_from"] == pd.Timestamp(seam) and pd.isna(windows.loc[successor, "valid_to"]), ticker
        assert windows.loc[predecessor, "oracle"] == windows.loc[successor, "oracle"] == "register", ticker
    assert after.backlog[after.backlog["kind"].eq("multi_cik_no_window") & after.backlog["canonical_ticker"].isin(SUCCESSOR_FILINGS)].empty
    blk = {segment.cik: segment for segment in load_registrants(CONFIG_DIR)["BLK"].segments}
    assert blk["0001364742"].valid_to == blk["0002012383"].valid_from == pd.Timestamp("2024-10-01")

    print("\n=== SANITY CHECK: register entries close the multi-CIK backlog ===")
    print(f"  without the register: {len(pending)} multi_cik_no_window items {sorted(pending['canonical_ticker'])}")
    for ticker, (predecessor, successor, seam) in SUCCESSOR_FILINGS.items():
        print(f"  {ticker:5s} {predecessor} [1900-01-01, {seam}) -> {successor} [{seam}, open)")
    print("  BLK 0001364742 -> 0002012383 at 2024-10-01 (the 8-K12B closing date)")
    print("  OK: every successor-filing entry yields dated register windows and leaves the backlog")


#: P18 register decisions kept by the traded-security realignment: (predecessor, successor, seam, the successor's own
#: pre-seam symbol or None). PLD and DOW left the register (2026-10-07).
REVISION_4_SEAMS = {
    "MRK": ("0000064978", "0000310158", "2009-11-03", "SGP"),
    "TPL": ("0000097517", "0001811074", "2021-01-11", None),
}


def test_the_revision_4_register_dates_the_reverse_mergers_from_the_accounting_predecessor(tmp_path):
    """P18: MRK and TPL get dated register windows. In MRK's reverse merger the legal acquirer filed under its own
    symbol before the seam, yet the accounting predecessor (whose prices MRK carries) owns the whole pre-seam window."""
    rows, owners = [], {}
    for i, (ticker, (predecessor, successor, seam, own_symbol)) in enumerate(REVISION_4_SEAMS.items()):
        rows += [(ticker, predecessor, "2000-01-03", seam, 50), (ticker, successor, seam, None, 50)]
        if own_symbol:
            rows.append((own_symbol, successor, "2000-01-03", seam, 40))
        owners[predecessor] = owners[successor] = {f"{i:04d}{k:06d}" for k in range(10)}
    tenure = _tenure(rows)
    roster = _roster([(ticker, successor) for ticker, (_, successor, _, _) in REVISION_4_SEAMS.items()])

    before = derive_entity_lineage(tenure, roster, _owner_pairs(owners), _config(tmp_path))
    after = derive_entity_lineage(tenure, roster, _owner_pairs(owners), CONFIG_DIR)
    windows = _rows(after, role="cik_window").set_index("cik")
    for ticker, (predecessor, successor, seam, _) in REVISION_4_SEAMS.items():
        assert predecessor not in set(_rows(before, role="cik_window")["cik"]), ticker
        assert windows.loc[predecessor, "valid_from"] == L.SENTINEL_START and windows.loc[predecessor, "valid_to"] == pd.Timestamp(seam), ticker
        assert windows.loc[successor, "valid_from"] == pd.Timestamp(seam) and pd.isna(windows.loc[successor, "valid_to"]), ticker
        assert windows.loc[predecessor, "oracle"] == windows.loc[successor, "oracle"] == "register", ticker
        assert windows.loc[predecessor, "canonical_ticker"] == windows.loc[successor, "canonical_ticker"] == ticker
    assert after.backlog[after.backlog["kind"].eq("multi_cik_no_window") & after.backlog["canonical_ticker"].isin(REVISION_4_SEAMS)].empty

    print("\n=== SANITY CHECK: revision-4 register windows (P18) ===")
    for ticker, (predecessor, successor, seam, own_symbol) in REVISION_4_SEAMS.items():
        note = f"; {successor} typed {own_symbol} before the seam and still owns nothing before it" if own_symbol else ""
        print(f"  {ticker:4s} {predecessor} [1900-01-01, {seam}) -> {successor} [{seam}, open){note}")
    print("  OK: without the register no predecessor window exists; with it each seam is dated and the accounting predecessor owns the past")


#: The acquired targets of the traded-security view: (target, roster CIK, seam, the roster CIK's own pre-seam symbol).
ACQUIRED_TARGETS = {
    "PLD": ("0000899881", "0001045609", "2011-06-03", "AMB"),
    "JCI": ("0000053669", "0000833444", "2016-09-02", "TYC"),
}


def test_an_acquired_target_is_an_event_only_cik_declared_by_the_manual(tmp_path):
    """D3a: with no register entry the roster CIK owns the ticker from the sentinel start; the target that typed the
    ticker before the seam stays in the entity as `cik_event`. Without the manual entry it is backlogged instead."""
    rows, owners = [], {}
    for i, (ticker, (target, home, seam, own_symbol)) in enumerate(ACQUIRED_TARGETS.items()):
        rows += [(ticker, target, "2000-01-03", seam, 50), (ticker, home, seam, None, 50), (own_symbol, home, "2000-01-03", seam, 40)]
        owners[target] = owners[home] = {f"{i:04d}{k:06d}" for k in range(10)}
    tenure = _tenure(rows)
    roster = _roster([(ticker, home) for ticker, (_, home, _, _) in ACQUIRED_TARGETS.items()])

    before = derive_entity_lineage(tenure, roster, _owner_pairs(owners), _config(tmp_path))
    after = derive_entity_lineage(tenure, roster, _owner_pairs(owners), CONFIG_DIR)
    pending = before.backlog[before.backlog["kind"].eq("multi_cik_no_window")]
    windows = _rows(after, role="cik_window").set_index("cik")
    events = _rows(after, role="cik_event").set_index("cik")
    print("\n=== SANITY CHECK: acquired targets (D3a) ===")
    print(f"  without the manual: multi_cik_no_window {sorted(pending['canonical_ticker'])}")
    for ticker, (target, home, seam, own_symbol) in ACQUIRED_TARGETS.items():
        print(
            f"  {ticker}: {home} window [{windows.loc[home, 'valid_from'].date()}, open) ({own_symbol} before {seam}); "
            f"{target} cik_event via {events.loc[target, 'oracle']}"
        )
        assert windows.loc[home, "valid_from"] == L.SENTINEL_START and pd.isna(windows.loc[home, "valid_to"]), ticker
        assert target not in windows.index, ticker
        assert events.loc[target, "oracle"] == "manual" and events.loc[target, "entity_id"] == windows.loc[home, "entity_id"], ticker
    assert sorted(pending["canonical_ticker"]) == sorted(ACQUIRED_TARGETS)
    assert after.backlog[after.backlog["kind"].eq("multi_cik_no_window") & after.backlog["canonical_ticker"].isin(list(ACQUIRED_TARGETS))].empty
    print("  OK: the traded security's CIK owns the past; the target is a cited event-only CIK, never a window.")


def test_a_manual_verdict_clears_the_cpt_grey_band():
    """CIK 0000096345 typed `(CPT)` once; the curated `own_entity` verdict keeps it out without a backlog row."""
    camden, typo = "0000906345", "0000096345"
    tenure = _tenure([("CPT", camden, "2007-05-15", None, 684), ("CPT", typo, "2006-01-12", "2006-01-13", 1)])
    owners = {camden: {f"{i:010d}" for i in range(31)}, typo: {"0000000000"}}
    shared, jaccard = score_overlap(owners[typo], owners[camden])
    assert classify_overlap(shared, jaccard) == "grey"
    build = derive_entity_lineage(tenure, _roster([("CPT", camden)]), _owner_pairs(owners), CONFIG_DIR)
    assert build.backlog[build.backlog["kind"].eq("grey_band")].empty
    assert set(_rows(build, cik=typo)["entity_id"]) == {"E0000096345"} and set(_rows(build, cik=typo)["oracle"]) == {"manual"}
    assert set(_rows(build, cik=camden)["entity_id"]) == {"E0000906345"}

    print("\n=== SANITY CHECK: CPT grey band settled ===")
    print(f"  overlap shared={shared} jaccard={jaccard:.3f} -> grey; curated own_entity -> no backlog, {typo} is its own entity")
    print("  OK: the one-digit-off CIK stays out of Camden's entity by a recorded decision")


def test_manual_config_records_the_2026_10_08_verdicts():
    """The seven event-only predecessor / acquired-target verdicts of the review load, each citing an SEC accession."""
    manual = load_manual_lineage(CONFIG_DIR)
    expected = {
        "CDW": {"0000899171", "0001402057"},
        "KMI": {"0000054502", "0001506307"},
        "CB": {"0000020171", "0000896159"},
        "DOC": {"0001574540", "0000765880"},
        "DELL": {"0000826083", "0001571996"},
        "GM": {"0000040730", "0001467858"},
        "VMRK": {"0000915912", "0000906107"},
    }
    accession = re.compile(r"\d{10}-\d{2}-\d{6}")
    found: dict[str, tuple[str, dict]] = {}
    for key, entry in manual.items():
        for ticker, ciks in expected.items():
            if set(entry["same_entity"]) == ciks:
                found[ticker] = (key, entry)
    missing = sorted(set(expected) - set(found))
    assert not missing, f"no same_entity verdict for {missing}"
    for ticker, (key, entry) in found.items():
        assert key.startswith(f"{ticker}/"), f"{ticker}: verdict filed under {key!r}"
        assert accession.search(entry["evidence"]), f"{key}: evidence cites no SEC accession"
        assert not entry["own_entity"], f"{key}: a same_entity verdict also declares own_entity"

    print("\n=== SANITY CHECK: 2026-10-08 same_entity verdicts ===")
    for key, entry in found.values():
        print(f"  {key:32s} {sorted(entry['same_entity'])}  cites {accession.search(entry['evidence']).group(0)}")
    print(f"  OK: {len(found)} verdicts load through load_manual_lineage, each evidenced by an SEC accession")


def test_d19_allowlist_is_loaded_from_the_curated_file():
    allow = load_d19_allowlist(CONFIG_DIR)
    assert {"BF-B", "BRK-B", "FOXA", "NWSA", "LEN", "VMRK"} <= set(allow)
    print("\n=== SANITY CHECK: D19 allow-list ===")
    print(f"  {len(allow)} evidenced entries, including the six share-class spellings")


def test_symbol_rows_carry_their_own_change_stamp(tmp_path):
    """A symbol-only change restamps that entity's symbol rows; its CIK rows (the EDGAR relist stamp) keep theirs."""
    config = _config(tmp_path)
    roster = _roster([("AAA", "0000000100"), ("BBB", "0000000200")])
    tenure = _tenure([("AAA", "0000000100", "2006-01-05", None, 300), ("BBB", "0000000200", "2006-01-05", None, 300)])
    t1, t2 = pd.Timestamp("2026-10-01 01:00:00"), pd.Timestamp("2026-10-02 01:00:00")
    first = derive_entity_lineage(tenure, roster, _owner_pairs({}), config, built_at=t1).rows
    renamed = pd.concat([tenure, _tenure([("BBX", "0000000200", "2010-01-05", "2012-01-01", 300)])], ignore_index=True)
    second = derive_entity_lineage(renamed, roster, _owner_pairs({}), config, existing=first, built_at=t2).rows

    stamps = (
        second.assign(group=second["role"].eq("symbol").map({True: "symbol", False: "cik"}))
        .groupby(["canonical_ticker", "group"])["scope_changed_at"]
        .agg(set)
    )
    assert stamps.to_dict() == {("AAA", "cik"): {t1}, ("AAA", "symbol"): {t1}, ("BBB", "cik"): {t1}, ("BBB", "symbol"): {t2}}
    print("\n=== SANITY CHECK: symbol change stamp ===")
    print(f"  BBB gains the BBX symbol interval: its symbol rows -> {t2}, its CIK rows keep {t1}; AAA untouched")
