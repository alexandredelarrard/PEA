"""
`entity_lineage` -- the id minting, the priority order and the two refusals, on synthetic
known-truth inputs, plus a real-data check that the live register and curated files actually
produce the verdicts the plan specifies.

The refusals are the point of this file. Two of them are load-bearing:
  * a merge joining two UNIVERSE tickers is rejected -- the only failure in this design that
    corrupts rows rather than dropping them;
  * an owner-overlap score in the grey band RAISES rather than guessing -- the measured grey
    band holds a genuine predecessor and a genuine symbol reuse at 0.037 vs 0.032.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_extract.utils.common import entity_lineage as lineage_module
from src.data_extract.utils.common.entity_lineage import (
    OVERLAP_JACCARD_SAME,
    OVERLAP_SHARED_SAME,
    UndecidedGreyBandError,
    _Union,
    candidate_ciks,
    classify_overlap,
    derive_entity_lineage,
    entity_id_for,
    load_d19_allowlist,
    load_manual_lineage,
    score_overlap,
)
from src.data_extract.utils.common.registrant import load_registrants
from src.data_store.schema import Tables

CONFIG_DIR = "./configs"
CACHE = Path("data/sec_insider_transactions")


def _tenure(rows: list[tuple[str, str, str, str | None, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": s,
                "issuer_cik": c,
                "valid_from": pd.Timestamp(f),
                "valid_to": pd.Timestamp(t) if t else pd.NaT,
                "n_filings": n,
                "source": "form345",
                "evidence": "",
            }
            for s, c, f, t, n in rows
        ]
    )


def _roster(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "cik": c} for t, c in rows])


def test_entity_id_is_the_oldest_cik_in_the_group():
    """No allocation state, so two independent derivations agree by construction."""
    group = {"0000030554", "0001666700", "0000029915"}
    assert entity_id_for(group) == "E0000029915"
    assert entity_id_for({"0001666700"}) == "E0001666700"

    print("\n=== SANITY CHECK: entity_id minting ===")
    print(f"  {sorted(group)} -> {entity_id_for(group)}")
    print("  OK: the numerically smallest (oldest) CIK names the group")
    print("  -> A re-derive from scratch reproduces the id; nothing is allocated.")


def test_merge_order_does_not_change_the_id():
    """Idempotence in the form that actually bites: the union-find visits pairs in oracle
    order, and a different order must not rename an entity."""
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
    print("  -> Oracle order can change without renaming a single entity.")


def test_a_merge_joining_two_universe_tickers_is_refused():
    """The one failure that corrupts rather than drops. `ITW`/`NTRS` is the live instance."""
    union = _Union(roster_ciks=frozenset({"0000049826", "0000073124"}))
    assert union.union("0000049826", "0000073124", source="owner_overlap[NTRS]", confidence=0.0397) is False
    assert union.find("0000049826") != union.find("0000073124")
    assert len(union.blocked) == 1 and union.blocked[0][2] == "owner_overlap[NTRS]"
    # a merge that brings in a NON-roster CIK is still allowed
    assert union.union("0000073124", "0001063537", source="owner_overlap[NTRS]") is True

    print("\n=== SANITY CHECK: the two-universe-tickers refusal ===")
    print(f"  blocked: {union.blocked[0]}")
    print("  OK: two roster CIKs stay in two entities; a non-roster CIK still merges")
    print("  -> Without this, one ticker's entity absorbs the other and relabels its rows.")


def test_the_grey_band_raises_instead_of_guessing():
    """`shared > 0` but below both thresholds, with no curated row, must stop the build."""
    assert classify_overlap(0, 0.0) == "unrelated"
    assert classify_overlap(3, 0.037) == "grey"
    assert classify_overlap(3, OVERLAP_JACCARD_SAME) == "same"
    assert classify_overlap(OVERLAP_SHARED_SAME, 0.001) == "same"

    # COHR's real shape, in miniature: 3 owners shared out of 31 and 53, jaccard 0.037
    tenure = _tenure([("AAA", "0000000111", "2006-01-05", "2012-01-01", 700), ("AAA", "0000000222", "2013-01-05", None, 700)])
    roster = _roster([("AAA", "0000000222")])
    owners = {"0000000111": {f"o{i}" for i in range(31)}, "0000000222": {f"o{i}" for i in range(28, 81)}}
    assert score_overlap(owners["0000000111"], owners["0000000222"])[0] == 3
    with pytest.raises(UndecidedGreyBandError) as exc:
        derive_entity_lineage(CACHE, tenure, roster, CONFIG_DIR, owners=owners)
    assert "AAA/0000000111" in str(exc.value)  # it names the pair, not merely a count

    print("\n=== SANITY CHECK: grey band ===")
    print(f"  thresholds: jaccard >= {OVERLAP_JACCARD_SAME} OR shared >= {OVERLAP_SHARED_SAME}")
    print(f"  classify(3, 0.037) = {classify_overlap(3, 0.037)}  (COHR's real score)")
    print(f"  classify(2, 0.032) = {classify_overlap(2, 0.032)}  (CEG's real score)")
    print("  OK: the two land in the same band and the builder refuses to separate them")
    print("  -> A fitted threshold would delete COHR's history or import CEG's.")


def test_score_overlap_is_symmetric_and_empty_safe():
    assert score_overlap(set(), {"a"}) == (0, 0.0)
    assert score_overlap({"a", "b"}, {"b", "c"}) == score_overlap({"b", "c"}, {"a", "b"})
    assert score_overlap({"a", "b"}, {"b", "c"}) == (1, 1 / 3)

    print("\n=== SANITY CHECK: overlap scoring ===")
    print(f"  {{a,b}} vs {{b,c}} -> {score_overlap({'a', 'b'}, {'b', 'c'})}")
    print("  OK: symmetric, and a CIK with no Form 345 owner scores 0 rather than raising")
    print("  -> 3 of the 621 live candidates have no owners at all; they read as unrelated.")


def test_entity_lineage_build_logs_changed_ciks_and_affected_tickers(monkeypatch, caplog):
    columns = ["cik", "entity_id", "source", "confidence", "evidence"]
    old = pd.DataFrame(
        [
            ("0000000001", "E0000000001", "singleton", 1.0, "old"),
        ],
        columns=columns,
    )
    new = pd.DataFrame(
        [
            ("0000000001", "E0000000002", "register", 1.0, "successor"),
        ],
        columns=columns,
    )
    roster = _roster([("AAA", "0000000001")])
    tenure = _tenure([("AAA", "0000000001", "2020-01-01", None, 1)])

    class Store:
        def load(self, table, **kwargs):
            del kwargs
            if table is Tables.symbol_tenure:
                return tenure
            if table is Tables.sp500_tickers:
                return roster
            if table is Tables.entity_lineage:
                return old
            raise AssertionError(table)

        def replace(self, table, frame):
            assert table is Tables.entity_lineage and frame.equals(new)
            return len(frame)

    monkeypatch.setattr(lineage_module, "derive_entity_lineage", lambda *args, **kwargs: (new, pd.DataFrame()))
    monkeypatch.setattr(lineage_module, "record_run", lambda *args, **kwargs: None)
    context = SimpleNamespace(store=Store(), log=logging.getLogger("test.entity_lineage"))
    caplog.set_level(logging.INFO, logger="test.entity_lineage")

    lineage_module.build_entity_lineage(context, CACHE, CONFIG_DIR)

    assert "1 changed CIK assignment(s): 0000000001" in caplog.text
    assert "affected current ticker(s): AAA" in caplog.text

    print("\n=== SANITY CHECK: entity-lineage refresh visibility ===")
    print("  changed CIK 0000000001 and affected current ticker AAA are named")
    print("  OK: identity reassignment is visible before symbol-only consumers run")


def test_candidate_set_is_universe_symbols_plus_roster_ciks():
    tenure = _tenure([("AAA", "0000000111", "2006-01-05", "2012-01-01", 5), ("ZZZ", "0000000999", "2006-01-05", None, 5)])
    roster = _roster([("AAA", "0000000222"), ("BBB", "0000000333")])
    candidates, by_ticker, roster_cik = candidate_ciks(tenure, roster)

    assert candidates == frozenset({"0000000111", "0000000222", "0000000333"})
    assert "0000000999" not in candidates  # ZZZ is not in the universe
    assert by_ticker["AAA"] == {"0000000111", "0000000222"}
    assert by_ticker["BBB"] == {"0000000333"}  # a roster CIK with no tenure still appears
    assert roster_cik == {"AAA": "0000000222", "BBB": "0000000333"}

    print("\n=== SANITY CHECK: candidate set ===")
    print(f"  candidates={sorted(candidates)}")
    print("  OK: scoped to universe symbols + roster CIKs; every other EDGAR CIK is a")
    print("      singleton by default")
    print("  -> That default IS the verdict `owns()` needs: an unseen reuse is dropped.")


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
    loaded = load_manual_lineage(str(tmp_path))
    assert loaded["X"]["own_entity"] == ["0001004440"]  # zero-padded on load

    print("\n=== SANITY CHECK: curated-file validation ===")
    print("  empty evidence / 1-CIK same_entity / an entry asserting nothing all RAISE")
    print("  OK: CIKs are zero-padded on load, so a config written bare still joins")
    print("  -> An undocumented verdict cannot enter the table.")


# --------------------------------------------------------------------------- #
# Real data: the live register + curated files against the live tables         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def live_lineage():
    if not CACHE.exists() or len(list(CACHE.glob("*.zip"))) < 40:
        pytest.skip(f"no cached Form 345 quarters under {CACHE}")
    from src.context import get_config_context
    from src.data_store.schema import Tables

    try:
        _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
        tenure = context.store.load(Tables.symbol_tenure, project=True, optional=True)
        roster = context.store.load(Tables.sp500_tickers, optional=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable ({type(exc).__name__})")
    if tenure is None or roster is None or tenure.empty or roster.empty:
        pytest.skip("symbol_tenure / sp500_tickers empty -- run `identity-tables` first")
    return derive_entity_lineage(CACHE, tenure, roster, CONFIG_DIR)


def test_no_entity_holds_two_universe_tickers(live_lineage):
    """The invariant the whole design rests on, asserted on the live tables."""
    lineage, blocked = live_lineage
    from src.context import get_config_context
    from src.data_store.schema import Tables

    _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    roster = context.store.load(Tables.sp500_tickers)
    entity = dict(zip(lineage["cik"].astype(str), lineage["entity_id"].astype(str), strict=False))

    per_entity: dict[str, list[str]] = {}
    for ticker, cik in zip(roster["ticker"].astype(str), roster["cik"].astype("string").str.zfill(10), strict=False):
        per_entity.setdefault(entity.get(cik, "E" + str(cik)), []).append(ticker)
    collisions = {e: t for e, t in per_entity.items() if len(t) > 1}
    assert not collisions, f"entities holding two universe tickers: {collisions}"
    assert len(per_entity) == len(roster)

    print("\n=== SANITY CHECK: one entity per universe ticker ===")
    print(f"  {len(roster)} tickers -> {len(per_entity)} entities, 0 collisions")
    print(f"  merges refused to keep it that way: {len(blocked)}")
    if len(blocked):
        print(blocked.to_string(index=False))
    print("  OK: no entity can relabel one universe ticker's rows onto another")
    print("  -> This is the only failure mode here that corrupts rather than drops.")


def test_the_worked_cases_resolve_as_the_plan_specifies(live_lineage):
    """Predecessors join their successor; reuses stay separate; `IR` falls out of `TT`."""
    lineage, _ = live_lineage
    entity = dict(zip(lineage["cik"].astype(str), lineage["entity_id"].astype(str), strict=False))
    source = dict(zip(lineage["cik"].astype(str), lineage["source"].astype(str), strict=False))

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

    # `IR` is the case that defeats a single-axis oracle: BOTH its historic CIKs belong to
    # TT's entity, and NOTHING in the code or the config mentions IR.
    assert func_e("0001160497") == func_e("0001466258") == func_e("0000836102")  # TT's entity
    assert func_e("0001160497") != func_e("0001699150")  # not today's IR
    register_json = Path(CONFIG_DIR) / "sec" / "registrant_cutover.json"
    manual_json = Path(CONFIG_DIR) / "sec" / "entity_lineage_manual.json"
    assert '"IR"' not in register_json.read_text(encoding="utf-8")
    assert '"IR/' not in manual_json.read_text(encoding="utf-8")
    # the CEG verdict is RECORDED, not merely absent
    assert source.get("0001004440") == "manual"

    print("\n=== SANITY CHECK: the worked identity cases ===")
    for label, (pred, succ) in same.items():
        print(f"  SAME   {label:26s} {func_e(pred)} == {func_e(succ)}  (src {source.get(pred, '-')})")
    for label, (other, home) in apart.items():
        print(f"  APART  {label:26s} {func_e(other)} != {func_e(home)}  (src {source.get(other, 'singleton')})")
    print(f"  IR     both historic CIKs -> {func_e('0001160497')} (TT), today's IR -> {func_e('0001699150')}")
    print("  OK: every verdict matches the plan, and there is no IR-specific rule anywhere")
    print("  -> The two-axis design is what settles IR; a symbol test could not.")


def test_d19_allowlist_covers_every_live_disagreement(live_lineage):
    """The roster CIK is Wikipedia-sourced and has been wrong (XOM). Every disagreement
    between it and `symbol_tenure` must be explained in writing or it is that defect again."""
    lineage, _ = live_lineage
    from src.context import get_config_context
    from src.data_store.schema import Tables

    _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    roster = context.store.load(Tables.sp500_tickers)
    tenure = context.store.load(Tables.symbol_tenure, project=True)
    entity = dict(zip(lineage["cik"].astype(str), lineage["entity_id"].astype(str), strict=False))

    def func_e(c):
        return entity.get(c, "E" + c)

    allow = load_d19_allowlist(CONFIG_DIR)

    disagree = []
    for ticker, cik in zip(roster["ticker"].astype(str), roster["cik"].astype("string").str.zfill(10), strict=False):
        rows = tenure[tenure["symbol"].astype(str) == ticker]
        if rows.empty:
            disagree.append((ticker, cik, None))
            continue
        openrows = rows[rows["valid_to"].isna()]
        pick = (openrows if not openrows.empty else rows).sort_values("n_filings").iloc[-1]
        if func_e(str(pick["issuer_cik"])) != func_e(str(cik)):
            disagree.append((ticker, cik, str(pick["issuer_cik"])))

    unexplained = [d for d in disagree if d[0] not in allow]
    assert not unexplained, (
        f"{len(unexplained)} roster CIK(s) disagree with symbol_tenure and are not in the "
        f"D19 allow-list: {unexplained}. Each is the XOM class of defect until a written "
        "reading says otherwise."
    )

    print("\n=== SANITY CHECK: D19 roster-CIK cross-check ===")
    print(f"  tickers {len(roster)}; agree {len(roster) - len(disagree)}; " f"disagree {len(disagree)}; all allow-listed with evidence")
    for ticker, cik, tenure_cik in disagree:
        print(f"    {ticker:6s} roster {cik} vs tenure {tenure_cik or 'NO TENURE':10s}" f" -- {allow[ticker][:78]}")
    print("  OK: no unexplained disagreement remains")
    print("  -> This is the free check that would have caught XOM's wrong CIK in 2026-08.")


def test_the_register_beats_owner_overlap(live_lineage):
    """Priority order, on a live case where the two genuinely disagree.

    `XOM`'s successor shell 0002115436 shares ZERO Form 345 reporting owners with the
    predecessor -- it has filed almost nothing -- so the overlap oracle says `unrelated`. The
    register says otherwise, with prose, and wins.
    """
    lineage, _ = live_lineage
    entity = dict(zip(lineage["cik"].astype(str), lineage["entity_id"].astype(str), strict=False))
    source = dict(zip(lineage["cik"].astype(str), lineage["source"].astype(str), strict=False))
    assert entity["0000034088"] == entity["0002115436"]
    assert source["0000034088"] == source["0002115436"] == "register"

    register_ciks = {c for entry in load_registrants(CONFIG_DIR).values() for c in entry.all_ciks()}
    from_register = {c for c, s in source.items() if s == "register"}
    assert from_register == register_ciks & set(source)

    print("\n=== SANITY CHECK: oracle priority ===")
    print(f"  XOM 0000034088 / 0002115436 -> {entity['0000034088']} via " f"{source['0000034088']} (owner overlap scored them UNRELATED)")
    print(f"  every one of the {len(from_register)} register CIKs is sourced `register`")
    print("  OK: the curated layer is never overruled by the automatic oracle")
    print("  -> A hand-evidenced chain outranks a statistic, which is the whole point of it.")
