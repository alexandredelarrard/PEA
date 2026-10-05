"""
Superinvestor roster WRITER — pure logic
(src/data_extract/utils/institutionals/fetch_superinvestors.py).

CIKs are resolved from SEC EDGAR company search (fund NAME -> 13F CIK), decoupled from any
local 13F cache. Network is stubbed; we test the Dataroma roster parse, the EDGAR atom parse
(lower-case `<cik>`; single- vs multi-match), the best-match pick, name->CIK resolution, the
per-code memoised resolver (with its rename fallback), the `superinvestor_roster` row shape,
and the resolution GATE — an unresolved manager that is not a recorded exception must raise,
because a manager that quietly drops out of the roster is the survivorship bug the table
exists to remove.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.utils.institutionals import fetch_superinvestors as si

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")
OVERRIDES = si.load_superinvestor_overrides(CONFIG_DIR)

#: Hand overrides the research proved wrong (trust / fund / unrelated entities that never filed the manager's 13F-HR).
WRONG_OVERRIDE_CIKS = {"0000827280", "0000858172", "0001570775", "0001549574", "0000932223", "0000200305", "0001002778", "0000885665"}
#: Dataroma's 2026 code migration (old fund-ticker code, new adviser code): each pair is one manager.
MIGRATED_CODES = [("oaklx", "HA"), ("MPGFX", "MPF"), ("FPACX", "FPA"), ("SEQUX", "RC"), ("CMAFX", "VAN")]
#: Fixture hand resolutions for the content-independent tests.
FIXTURE_CIK_OVERRIDES = {"BRK": "0001067983"}
FIXTURE_UNRESOLVABLE = {"CMAFX": "fixture: never filed a 13F-HR"}

_SINGLE_ATOM = (
    '<?xml version="1.0"?><feed><company-info>'
    "<cik>0001079114</cik><cik-href>x</cik-href>"
    "<conformed-name>GREENLIGHT CAPITAL INC</conformed-name>"
    "</company-info></feed>"
)
_MULTI_ATOM = (
    '<?xml version="1.0"?><feed>'
    '<company-info name="ARRAY(0x1)"><cik>0001336528</cik></company-info>'
    '<company-info name="ARRAY(0x2)"><cik>0002026053</cik></company-info></feed>'
)


def test_overrides_config_carries_evidence_and_valid_chains():
    """Checks the PRODUCTION overrides on purpose: every hand resolution is a 10-digit CIK carrying its
    13F evidence, every chain window carries evidence and the loader validates the chains, no proven-wrong
    CIK is left, and each 2026 migrated code resolves to the same manager as the code it replaced."""
    blob = json.loads((Path(CONFIG_DIR) / "superinvestors" / "overrides.json").read_text(encoding="utf-8"))
    for code, entry in blob["cik_overrides"].items():
        assert re.fullmatch(r"\d{10}", entry["cik"]), code
        assert "13F-HR" in entry.get("evidence", ""), code
    windows = [w for chain in blob["manager_ciks"].values() for w in chain]
    assert all(re.fullmatch(r"\d{10}", w["cik"]) and "13F-HR" in w.get("evidence", "") for w in windows)
    assert set(OVERRIDES.manager_ciks) == {"0001006438", "0000728014", "0001079114", "0001056258"}
    assert not WRONG_OVERRIDE_CIKS & set(OVERRIDES.cik_by_code.values())
    assert not {"CMAFX", "LUK"} & set(OVERRIDES.unresolvable)
    assert all(r.reason and r.window.start for r in OVERRIDES.inactive.values())
    for old, new in MIGRATED_CODES:
        assert OVERRIDES.manager_id(OVERRIDES.cik_by_code[old]) == OVERRIDES.manager_id(OVERRIDES.cik_by_code[new]), (old, new)
    assert si.load_superinvestor_overrides(f"{CONFIG_DIR}/../configs") is OVERRIDES  # cached per resolved directory
    print("\n=== SANITY: superinvestor overrides config ===")
    print(
        f"  {len(OVERRIDES.cik_by_code)} overrides, each a 10-digit CIK with 13F evidence; {len(OVERRIDES.manager_ciks)} chains "
        f"({len(windows)} windows) validated; inactive {sorted(OVERRIDES.inactive)}; unresolvable {sorted(OVERRIDES.unresolvable)}; "
        f"none of the {len(WRONG_OVERRIDE_CIKS)} proven-wrong CIKs remain; {len(MIGRATED_CODES)} migrated code pairs share a manager. Validated."
    )


def test_pad_cik_canonical_10_digit():
    assert si.pad_cik("1067983") == "0001067983"
    assert si.pad_cik("1067983.0") == "0001067983"  # float artifact tolerated
    assert si.pad_cik(1067983) == "0001067983"
    assert si.pad_cik("") == "" and si.pad_cik("N/A") == ""
    print("\n=== SANITY: CIK canonicalization ===")
    print("  '1067983' / '1067983.0' / int -> '0001067983'; junk -> ''. Validated.")


def test_parse_dataroma_roster_strips_updated_suffix():
    html = """<table>
      <tr><td><a href="holdings.php?m=BRK">Warren Buffett - Berkshire Hathaway Updated 15 May 2026</a></td></tr>
      <tr><td><a href="/m/holdings.php?m=GLRE">David Einhorn - Greenlight Capital Updated 10 Jul 2026</a></td></tr>
      <tr><td><a href="holdings.php?m=BRK">dupe link</a></td></tr>
      <tr><td><a href="/m/managers.php">All managers</a></td></tr></table>"""
    roster = si._parse_dataroma_roster(html)
    assert [r["code"] for r in roster] == ["BRK", "GLRE"]  # deduped, non-manager link dropped
    assert roster[0]["name"] == "Warren Buffett - Berkshire Hathaway"  # "Updated ..." stripped
    print("\n=== SANITY: Dataroma roster parse ===")
    print(f"  {[r['code'] for r in roster]}; 'Updated <date>' stripped, dupe/non-holdings dropped. Validated.")


def test_parse_edgar_matches_lowercase_cik_single_and_multi():
    assert si._parse_edgar_matches(_SINGLE_ATOM) == [("0001079114", "GREENLIGHT CAPITAL INC")]
    assert si._parse_edgar_matches(_MULTI_ATOM) == [("0001336528", ""), ("0002026053", "")]
    assert si._parse_edgar_matches("no matches") == []
    print("\n=== SANITY: EDGAR atom parse ===")
    print("  lower-case <cik> parsed; single->name kept, multi->2 blocks (name may be empty). Validated.")


def test_pick_best_match():
    # single -> trusted outright
    assert si._pick_best_match([("0001079114", "GREENLIGHT CAPITAL INC")], "Greenlight Capital") == ("0001079114", "GREENLIGHT CAPITAL INC")
    # multi WITH names -> highest token overlap
    pairs = [("0000000001", "ACME HOLDINGS"), ("0000000002", "PERSHING SQUARE CAPITAL")]
    match = si._pick_best_match(pairs, "Pershing Square")
    assert match is not None
    assert match[0] == "0000000002"
    # multi WITHOUT names -> EDGAR's first (most-relevant) block
    match = si._pick_best_match([("0001336528", ""), ("0002026053", "")], "Pershing Square")
    assert match is not None
    assert match[0] == "0001336528"
    assert si._pick_best_match([], "x") is None
    print("\n=== SANITY: best-CIK pick ===")
    print("  single trusted; multi by token overlap; no-name multi -> first block; empty -> None. Validated.")


def test_edgar_cik_for_name_stubbed():
    calls = {}

    def fake_get(url):
        calls["url"] = url
        return SimpleNamespace(text=_SINGLE_ATOM)

    cik, filer = si._edgar_cik_for_name("David Einhorn - Greenlight Capital", get_fn=fake_get)
    assert filer is not None
    assert cik == "0001079114" and "GREENLIGHT" in filer
    assert "company=Greenlight" in calls["url"]  # searched the FUND part, url-quoted

    def boom(url):
        raise RuntimeError("network down")

    assert si._edgar_cik_for_name("X - Y Capital", get_fn=boom) == (None, None)  # no raise
    print("\n=== SANITY: name -> CIK via EDGAR ===")
    print("  'Greenlight Capital' -> 0001079114 (fund part queried); network error -> (None,None). Validated.")


def test_resolver_is_memoised_per_code_and_falls_back_to_older_names():
    """The seed replays 879 manager-rows over 104 codes, and 52 codes were renamed at least
    once. So the resolver keys on the CODE (one EDGAR call per manager, not per row) and,
    when the current name misses, retries the names that code carried before."""
    calls = []

    def fake_get(url):
        calls.append(url)
        return SimpleNamespace(text=_SINGLE_ATOM if "Greenlight" in url else "no company-info")

    resolve = si._make_resolver(fake_get, FIXTURE_CIK_OVERRIDES, {"GLRE": ["Greenlight Capital", "Greenlight Re"]})
    # the name in hand does not resolve; the older one does
    assert resolve("GLRE", "David Einhorn - Some Rebrand") == ("0001079114", si.RESOLUTION_EDGAR)
    n_after_first = len(calls)
    assert n_after_first == 2  # the rebrand, then the old name
    for _ in range(5):  # same code, 5 more snapshots
        resolve("GLRE", "David Einhorn - Some Rebrand")
    assert len(calls) == n_after_first  # memoised: no extra EDGAR calls
    assert resolve("BRK", "anything") == ("0001067983", si.RESOLUTION_OVERRIDE)
    assert len(calls) == n_after_first  # an override never hits the network
    print("\n=== SANITY: per-code memoised resolution ===")
    print(
        f"  6 snapshot-rows of one code cost {len(calls)} EDGAR calls (2 = failed current "
        "name + successful older name); an override costs 0. Validated."
    )


def test_snapshot_rows_shape_and_padding():
    rows = si.snapshot_rows(
        [{"code": "BRK", "name": "Warren Buffett - Berkshire"}, {"code": "CMAFX", "name": "Century Management"}],
        date(2016, 1, 1),
        "https://web.archive.org/web/2016/x",
        lambda code, name: ("1067983", si.RESOLUTION_OVERRIDE) if code == "BRK" else (None, si.RESOLUTION_UNRESOLVED),
        overrides=si.SuperinvestorOverrides(cik_by_code=FIXTURE_CIK_OVERRIDES, unresolvable=FIXTURE_UNRESOLVABLE),
    )
    assert [r["cik"] for r in rows] == ["0001067983", None]  # padded; unresolved -> NULL, not ""
    assert {r["snapshot_date"] for r in rows} == {date(2016, 1, 1)}
    assert set(rows[0]) == {"snapshot_date", "dataroma_code", "manager_name", "cik", "resolution", "source_url"}
    print("\n=== SANITY: superinvestor_roster row shape ===")
    print(
        f"  {len(rows)} rows keyed (snapshot_date, dataroma_code); CIK zero-padded to 10; "
        "an unresolved manager keeps its row with cik=None (never dropped). Validated."
    )


def test_unresolved_manager_raises_unless_recorded():
    """D22's gate: 100% resolution, or every exception named with its reason. An unresolved
    manager silently leaving the eligible pool IS the survivorship bug, so it must be loud."""
    unknown = [
        {
            "snapshot_date": date(2016, 1, 1),
            "dataroma_code": "ZZZ",
            "manager_name": "Nobody - Unlisted Boutique",
            "cik": None,
            "resolution": si.RESOLUTION_UNRESOLVED,
            "source_url": "x",
        }
    ]
    with pytest.raises(si.SuperinvestorResolutionError, match="Unlisted Boutique"):
        si.assert_fully_resolved(unknown, FIXTURE_UNRESOLVABLE)

    recorded = [dict(unknown[0], dataroma_code="CMAFX", manager_name="Century Management")]
    assert si.assert_fully_resolved(recorded, FIXTURE_UNRESOLVABLE) == ["CMAFX"]  # recorded -> passes, reported
    resolved = [dict(unknown[0], cik="0001067983", resolution=si.RESOLUTION_EDGAR)]
    assert si.assert_fully_resolved(resolved, FIXTURE_UNRESOLVABLE) == []
    print("\n=== SANITY: resolution gate ===")
    print(
        f"  an unknown unresolved code RAISES; the {len(FIXTURE_UNRESOLVABLE)} "
        f"recorded exceptions {sorted(FIXTURE_UNRESOLVABLE)} pass and are returned "
        "for the caller to report. Validated."
    )


def _seed_13f(store: Any, hr: list[tuple[str, str]] | None = None, books: list[tuple[str, str]] | None = None) -> None:
    """Seed (cik, period) 13F activity: `hr` into `sec13f_hr` (the S&P slice), `books` into `sec13f_manager_holdings`."""
    if hr:
        store.save(si.Tables.sec13f_hr, pd.DataFrame([{"cik": c, "period": p, "ticker": "AAPL", "cusip": "037833100", "shares": 1.0} for c, p in hr]))
    if books:
        store.save(si.Tables.sec13f_manager_holdings, pd.DataFrame([{"cik": c, "period": p, "cusip": "037833100", "shares": 1.0} for c, p in books]))


def _write_overrides(tmp_path: Path, blob: dict[str, Any]) -> str:
    """A fixture `<config_dir>/superinvestors/overrides.json`; returns the config dir."""
    config_dir = tmp_path / "configs"
    (config_dir / "superinvestors").mkdir(parents=True, exist_ok=True)
    (config_dir / "superinvestors" / "overrides.json").write_text(json.dumps(blob), encoding="utf-8")
    return str(config_dir)


def _gate_row(code: str, cik: str | None, snapshot: str) -> dict[str, Any]:
    """One roster row as the writer builds it."""
    return {
        "snapshot_date": date.fromisoformat(snapshot),
        "dataroma_code": code,
        "manager_name": f"{code} - Fixture Fund",
        "cik": cik,
        "resolution": si.RESOLUTION_OVERRIDE if cik else si.RESOLUTION_UNRESOLVED,
        "source_url": "x",
    }


def _no_listing(cik: str) -> set[date]:
    raise AssertionError(f"no EDGAR listing expected on a local hit: {cik}")


_ROSTER_HTML = (
    '<a href="holdings.php?m=GLRE">David Einhorn - Greenlight Capital</a><a href="holdings.php?m=BRK">Warren Buffett - Berkshire Hathaway</a>'
)


def _refresh_fixture(monkeypatch, sqlite_store, tmp_path) -> Any:
    """Live-refresh context: the two-manager Dataroma page, GLRE via stubbed EDGAR search, BRK via override,
    both with 13F activity at the quarter before today."""
    monkeypatch.setattr(si, "_http_get", lambda url: SimpleNamespace(text=_ROSTER_HTML))
    q = str(si.snapshot_quarter(datetime.now(UTC).date()))
    _seed_13f(sqlite_store, hr=[("1079114", q)], books=[("0001067983", q)])
    config_dir = _write_overrides(tmp_path, {"cik_overrides": {c: {"cik": k} for c, k in FIXTURE_CIK_OVERRIDES.items()}, "unresolvable": {}})
    return cast(Any, SimpleNamespace(store=sqlite_store, config_dir=config_dir))


def _fake_edgar(url: str) -> Any:
    return SimpleNamespace(text=_SINGLE_ATOM if "Greenlight" in url else "no company-info")


def _stored_snapshot(store: Any, snapshot: str, mapping: dict[str, str]) -> None:
    rows = [dict(_gate_row(code, cik, snapshot), source_url=si.DATAROMA_HOME_URL) for code, cik in mapping.items()]
    store.save(si.Tables.superinvestor_roster, pd.DataFrame(rows))


def test_live_refresh_noop_when_unchanged(monkeypatch, sqlite_store, tmp_path):
    """D-6: a refresh whose (code -> cik) mapping equals the latest stored snapshot writes nothing, on a
    later day and on a same-day repeat."""
    ctx = _refresh_fixture(monkeypatch, sqlite_store, tmp_path)
    _stored_snapshot(sqlite_store, "2026-09-08", {"GLRE": "0001079114", "BRK": "0001067983"})
    before = sqlite_store.row_count(si.Tables.superinvestor_roster)
    first = si.upsert_roster_snapshot(ctx, get_fn=_fake_edgar, listing_fn=_no_listing)
    second = si.upsert_roster_snapshot(ctx, get_fn=_fake_edgar, listing_fn=_no_listing)
    stored = sqlite_store.load(si.Tables.superinvestor_roster)
    assert first.empty and second.empty
    assert len(stored) == before == 2 and set(stored["snapshot_date"].astype(str)) == {"2026-09-08"}
    print("\n=== SANITY: live refresh is a no-op when nothing changed ===")
    print(
        f"  stored snapshot 2026-09-08 = today's mapping -> two refreshes wrote 0 rows; table stays at {len(stored)} rows. Validated on the real store."
    )


def test_live_refresh_writes_on_change(monkeypatch, sqlite_store, tmp_path):
    """A membership change, then a CIK change (an override correction), each write a full gated snapshot dated
    today; an unchanged repeat in between writes nothing."""
    ctx = _refresh_fixture(monkeypatch, sqlite_store, tmp_path)
    _stored_snapshot(sqlite_store, "2026-09-08", {"BRK": "0001067983"})
    df = si.upsert_roster_snapshot(ctx, get_fn=_fake_edgar, listing_fn=_no_listing)
    today = datetime.now(UTC).date()
    assert dict(zip(df["dataroma_code"], df["cik"], strict=True)) == {"GLRE": "0001079114", "BRK": "0001067983"}
    assert set(df["snapshot_date"]) == {today}
    assert si.upsert_roster_snapshot(ctx, get_fn=_fake_edgar, listing_fn=_no_listing).empty

    corrected = "0000000007"
    _seed_13f(sqlite_store, books=[(corrected, str(si.snapshot_quarter(today)))])
    ctx.config_dir = _write_overrides(tmp_path / "corrected", {"cik_overrides": {"BRK": {"cik": "0001067983"}, "GLRE": {"cik": corrected}}})
    moved = si.upsert_roster_snapshot(ctx, get_fn=_fake_edgar, listing_fn=_no_listing)
    assert dict(zip(moved["dataroma_code"], moved["cik"], strict=True)) == {"GLRE": corrected, "BRK": "0001067983"}
    stored = sqlite_store.load(si.Tables.superinvestor_roster)
    assert len(stored) == 3, stored  # 1 old row + today's 2, upserted in place on the CIK change
    print("\n=== SANITY: live refresh writes on change ===")
    print(
        f"  GLRE joined -> a {len(df)}-row snapshot dated {today} (gated on local 13F activity); the repeat wrote 0; "
        f"an override moving GLRE to {corrected} rewrote today's snapshot. {len(stored)} rows stored. Validated on the real store."
    )


def test_gate_raises_on_inactive_cik(sqlite_store):
    """The RC -> TRAC Intermodal shape: a CIK with no 13F-HR near the snapshot raises, naming code, snapshot,
    CIK and the nearest known period; a CIK with local activity passes without an EDGAR listing call."""
    trac, lapsed, brk = "0001570775", "0000099999", "0001067983"
    _seed_13f(sqlite_store, hr=[("99999", "2019-12-31")], books=[(brk, "2026-06-30")])
    ctx = cast(Any, SimpleNamespace(store=sqlite_store))
    evidence = si.activity_evidence(ctx, {trac, lapsed, brk})
    assert evidence == {(lapsed, date(2019, 12, 31)), (brk, date(2026, 6, 30))}  # unpadded hr row normalised
    listed: list[str] = []

    def listing(cik: str) -> set[date]:
        listed.append(cik)
        return {date(2020, 3, 31)} if cik == lapsed else set()

    rows = [_gate_row("RC", trac, "2026-09-08"), _gate_row("LAPSED", lapsed, "2026-09-08"), _gate_row("BRK", brk, "2026-09-08")]
    with pytest.raises(si.SuperinvestorResolutionError) as err:
        si.assert_active(rows, evidence, {}, listing)
    msg = str(err.value)
    assert '"RC"' in msg and trac in msg and "2026-09-08" in msg and "none" in msg
    assert '"LAPSED"' in msg and "2020-03-31" in msg  # the listing's period is nearer than the local 2019-12-31
    assert "BRK" not in msg and sorted(listed) == sorted([trac, lapsed])
    print("\n=== SANITY: activity gate raises on an inactive CIK ===")
    print(
        f"  RC -> {trac} (never filed) and a lapsed filer raise with code/snapshot/CIK/nearest period; BRK passes on "
        "local evidence, the listing was asked only for the 2 misses. Validated."
    )


def test_gate_passes_recorded_inactive_exception():
    """A dated `inactive` range skips the gate only for snapshots whose Q(snap) it covers; outside it the gate applies."""
    aq = "0000000042"
    inactive = {"aq": si.InactiveRange(si.PeriodRange(date(2022, 9, 30), date(2023, 12, 31)), "fixture: no 13F after 2022-06")}
    evidence = {(aq, date(2022, 6, 30))}

    def no_listing(cik: str) -> set[date]:
        return set()

    inside = [_gate_row("aq", aq, "2023-02-15"), _gate_row("aq", aq, "2024-01-10")]  # Q 2022-12-31, 2023-12-31
    assert si.assert_active(inside, evidence, inactive, no_listing) == ["aq"]
    assert si.assert_active([_gate_row("aq", aq, "2022-08-01")], evidence, inactive, no_listing) == []  # before: active
    with pytest.raises(si.SuperinvestorResolutionError, match="2024-05-01"):
        si.assert_active([_gate_row("aq", aq, "2024-05-01")], evidence, inactive, no_listing)  # Q 2024-03-31: after the range
    print("\n=== SANITY: dated inactive exception ===")
    print("  aq inactive 2022-09-30..2023-12-31: snapshots inside pass (reported), 2022-08 passes on evidence, 2024-05 raises. Validated.")


def test_gate_clamps_pre_xml_window():
    """Before the XML era the local tables are sparse, so a pre-XML window extends to the first XML quarter;
    after it the window is the plain [Q-4q, Q+2q]."""
    a, b, c = "0000000001", "0000000002", "0000000003"
    evidence = {(a, date(2013, 6, 30)), (b, date(2013, 9, 30)), (c, date(2015, 12, 31))}
    assert si.assert_active([_gate_row("A", a, "2012-03-07")], evidence, {}, _no_listing) == []  # Q 2011-12-31
    with pytest.raises(si.SuperinvestorResolutionError) as err:
        si.assert_active([_gate_row("B", b, "2012-03-07"), _gate_row("C", c, "2015-03-01")], evidence, {}, lambda cik: set())
    assert '"B"' in str(err.value) and '"C"' in str(err.value)
    assert si.SEC13F_XML_ERA_START == date(2013, 6, 30)
    print("\n=== SANITY: pre-XML window clamp ===")
    print(
        "  2012-03-07 passes on a 2013-06-30 book (window end clamped to the XML-era start) but not on 2013-09-30; a 2015 snapshot is not clamped. Validated."
    )


def test_gate_uses_listing_on_local_miss(sqlite_store):
    """A 2012-only filer with no local rows passes through the EDGAR listing, asked once per CIK; a CIK with
    local evidence never triggers the listing."""
    old, brk = "0000846222", "0001067983"
    _seed_13f(sqlite_store, books=[(brk, "2012-03-31")])
    evidence = si.activity_evidence(cast(Any, SimpleNamespace(store=sqlite_store)), {old, brk})
    calls: list[str] = []

    def listing(cik: str) -> set[date]:
        calls.append(cik)
        return {date(2012, 3, 31), date(2012, 6, 30)}

    rows = [_gate_row("GH", old, "2012-05-01"), _gate_row("GH", old, "2012-08-01")]
    assert si.assert_active(rows, evidence, {}, listing) == []
    assert calls == [old]  # memoised per CIK
    assert si.assert_active([_gate_row("BRK", brk, "2012-05-01")], evidence, {}, _no_listing) == []
    print("\n=== SANITY: EDGAR listing only on a local miss ===")
    print(f"  {old} (no local rows) passed 2 snapshots on 1 listing call; BRK passed on its local book without one. Validated on the real store.")


def test_history_and_overrides_read_from_config_dir(sqlite_store, tmp_path):
    """The seed reads the roster history and the overrides from `<config_dir>/superinvestors/`,
    never from the data store: a decoy history under DATA_STORE must be ignored."""
    config_dir = tmp_path / "configs"
    (config_dir / "superinvestors").mkdir(parents=True)
    (config_dir / "superinvestors" / "overrides.json").write_text(
        json.dumps({"cik_overrides": {"AAA": {"cik": "1234"}}, "unresolvable": {"ZZZ": "fixture: never filed a 13F-HR"}}), encoding="utf-8"
    )
    (config_dir / "superinvestors" / "dataroma_roster_history.json").write_text(
        json.dumps(
            {
                "_README": ["fixture"],
                "snapshots": [
                    {
                        "captured_at": "2015-03-30T10:00:00Z",
                        "source_url": "wayback-a",
                        "managers": {"AAA": "Alice - Alpha Fund", "ZZZ": "Zed - Zeta"},
                    },
                    {"captured_at": "2016-03-15T10:00:00Z", "source_url": "wayback-b", "managers": {"AAA": "Alice - Alpha Fund"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    data_store = tmp_path / "data_store"
    (data_store / "superinvestors").mkdir(parents=True)
    (data_store / "superinvestors" / "dataroma_roster_history.json").write_text(json.dumps({"2014": {"DECOY": "Decoy - Fund"}}), encoding="utf-8")

    queried: list[str] = []

    def empty_edgar(url):
        queried.append(url)
        return SimpleNamespace(text="no company-info")

    _seed_13f(sqlite_store, books=[("0000001234", "2015-06-30")])
    ctx = cast(Any, SimpleNamespace(store=sqlite_store, config_dir=str(config_dir), paths={"DATA_STORE": data_store}))
    df = si.rebuild_roster(ctx, get_fn=empty_edgar, listing_fn=_no_listing)
    got = {(str(r.snapshot_date), r.dataroma_code, None if pd.isna(r.cik) else r.cik, r.resolution) for r in df.itertuples(index=False)}
    assert got == {
        ("2015-03-30", "AAA", "0000001234", si.RESOLUTION_OVERRIDE),
        ("2015-03-30", "ZZZ", None, si.RESOLUTION_UNRESOLVED),
        ("2016-03-15", "AAA", "0000001234", si.RESOLUTION_OVERRIDE),
    }
    assert "DECOY" not in set(sqlite_store.load(si.Tables.superinvestor_roster)["dataroma_code"])
    assert len(queried) == 1 and "company=Zeta" in queried[0]  # only the non-overridden code searched EDGAR
    print("\n=== SANITY: roster config location ===")
    print(
        f"  seed wrote {len(df)} rows from <config_dir>/superinvestors/dataroma_roster_history.json; AAA took the "
        "fixture override CIK (no EDGAR call) and ZZZ stayed NULL as the fixture's recorded exception; "
        "the DATA_STORE decoy was never read. Validated on the real store."
    )


def test_writer_picks_member_valid_at_snapshot(sqlite_store, tmp_path):
    """A resolved CIK is stored as the member of its manager's chain valid at the last quarter end
    strictly before the snapshot date: a snapshot before the successor's first book stores the
    predecessor, even when the code resolves to the successor."""
    old, new = "0001006438", "0001656456"
    config_dir = tmp_path / "configs"
    (config_dir / "superinvestors").mkdir(parents=True)
    overrides = {
        "cik_overrides": {"AM": {"cik": new}, "BRK": {"cik": "0001067983"}},
        "unresolvable": {},
        "manager_ciks": {old: [{"cik": old, "to": "2015-12-31"}, {"cik": new, "from": "2016-03-31"}]},
    }
    (config_dir / "superinvestors" / "overrides.json").write_text(json.dumps(overrides), encoding="utf-8")
    stamps = ["2015-11-20T10:00:00Z", "2016-02-15T10:00:00Z", "2016-03-31T23:00:00Z", "2016-04-01T01:00:00Z", "2016-08-01T10:00:00Z"]
    history = {
        "_README": ["fixture"],
        "snapshots": [{"captured_at": t, "source_url": f"wb-{t}", "managers": {"AM": "Tepper", "BRK": "Buffett"}} for t in stamps],
    }
    (config_dir / "superinvestors" / "dataroma_roster_history.json").write_text(json.dumps(history), encoding="utf-8")

    _seed_13f(sqlite_store, hr=[(old, "2015-09-30"), (new, "2016-03-31"), ("0001067983", "2015-12-31")])
    ctx = cast(Any, SimpleNamespace(store=sqlite_store, config_dir=str(config_dir)))
    df = si.rebuild_roster(ctx, get_fn=lambda url: pytest.fail(f"no EDGAR call expected: {url}"), listing_fn=_no_listing)
    am = {str(r.snapshot_date): r.cik for r in df.itertuples(index=False) if r.dataroma_code == "AM"}
    assert am == {"2015-11-20": old, "2016-02-15": old, "2016-03-31": old, "2016-04-01": new, "2016-08-01": new}
    assert set(df.loc[df["dataroma_code"] == "BRK", "cik"]) == {"0001067983"}  # a singleton is stored as resolved
    assert si.snapshot_quarter(date(2016, 3, 31)) == date(2015, 12, 31)
    assert si.snapshot_quarter(date(2016, 4, 1)) == date(2016, 3, 31)
    assert si.snapshot_quarter(date(2016, 1, 1)) == date(2015, 12, 31)
    print("\n=== SANITY: writer stores the chain member valid at the snapshot ===")
    print(
        f"  AM resolves to Appaloosa LP ({new}); snapshots 2015-11-20, 2016-02-15 and 2016-03-31 (Q = 2015-12-31, inside "
        f"the succession quarter) store {old}, 2016-04-01 and 2016-08-01 store {new}. Validated on the real store."
    )


# --------------------------------------------------------------------------- #
# Full rebuild (P-2): committed history + stored live snapshots, resolved fresh  #
# --------------------------------------------------------------------------- #
_BRK, _GLRE, _GLRE_WRONG, _PSC, _PSC_WRONG = "0001067983", "0001079114", "0000846222", "0001336528", "0002026053"
_PSC_ATOM = (
    '<?xml version="1.0"?><feed><company-info><cik>0001336528</cik>'
    "<conformed-name>PERSHING SQUARE CAPITAL MANAGEMENT, L.P.</conformed-name></company-info></feed>"
)


def _rebuild_fixture(sqlite_store, tmp_path, extra_history: list[dict[str, Any]] | None = None) -> tuple[Any, list[str]]:
    """Two committed history snapshots (BRK, GLRE by override) and a stored table as an old run left it: a stale
    1-January seed snapshot, two identical live snapshots with a wrong GLRE CIK (stored `edgar`) and a wrong PSC CIK
    (live-only code, resolved by EDGAR), plus 13F activity near every snapshot. Returns the context and the
    list the stub EDGAR search appends its queries to."""
    config_dir = _write_overrides(tmp_path, {"cik_overrides": {"BRK": {"cik": _BRK}, "GLRE": {"cik": _GLRE}}, "unresolvable": {}})
    pair = {"BRK": "Warren Buffett - Berkshire Hathaway", "GLRE": "David Einhorn - Greenlight Capital"}
    history = {
        "_README": ["fixture"],
        "snapshots": [
            {"captured_at": "2015-03-30T10:00:00Z", "source_url": "https://web.archive.org/web/a", "managers": pair},
            {"captured_at": "2016-03-15T10:00:00Z", "source_url": "https://web.archive.org/web/b", "managers": pair},
            *(extra_history or []),
        ],
    }
    (Path(config_dir) / "superinvestors" / "dataroma_roster_history.json").write_text(json.dumps(history), encoding="utf-8")
    stale = [dict(_gate_row(c, k, "2015-01-01"), source_url="https://web.archive.org/web/old") for c, k in (("BRK", _BRK), ("GLRE", _GLRE_WRONG))]
    live = [
        dict(_gate_row(code, cik, day), resolution=si.RESOLUTION_EDGAR, source_url=si.DATAROMA_HOME_URL, manager_name=name)
        for day in ("2026-09-08", "2026-09-09")
        for code, cik, name in (
            ("BRK", _BRK, pair["BRK"]),
            ("GLRE", _GLRE_WRONG, pair["GLRE"]),
            ("PSC", _PSC_WRONG, "Bill Ackman - Pershing Square Capital Management"),
        )
    ]
    sqlite_store.save(si.Tables.superinvestor_roster, pd.DataFrame(stale + live))
    _seed_13f(
        sqlite_store, hr=[(_BRK, "2015-03-31"), (_GLRE, "2015-03-31"), (_GLRE, "2026-06-30")], books=[(_BRK, "2026-06-30"), (_PSC, "2026-06-30")]
    )
    queried: list[str] = []

    def edgar(url: str) -> Any:
        queried.append(url)
        return SimpleNamespace(text=_PSC_ATOM if "Pershing" in url else "no company-info")

    return cast(Any, SimpleNamespace(store=sqlite_store, config_dir=config_dir, edgar=edgar)), queried


def _table_hash(store: Any) -> str:
    """sha256 of the whole roster table, every column, rows sorted."""
    df = store.load(si.Tables.superinvestor_roster).astype(str)
    return hashlib.sha256(df.sort_values(list(df.columns)).to_csv(index=False).encode()).hexdigest()


def test_rebuild_roster_combines_history_and_live_resolved_fresh(sqlite_store, tmp_path):
    """The rebuild replaces the table with every committed history snapshot plus every stored live snapshot (none
    collapsed), all resolved fresh: a wrong stored CIK is corrected by the override or by EDGAR, a stale seed
    snapshot outside the history disappears, and a second rebuild leaves the table byte-identical."""
    ctx, queried = _rebuild_fixture(sqlite_store, tmp_path)
    df = si.rebuild_roster(ctx, get_fn=ctx.edgar, listing_fn=_no_listing)
    stored = sqlite_store.load(si.Tables.superinvestor_roster)
    got = {(str(d)[:10], c, k, r) for d, c, k, r in stored[["snapshot_date", "dataroma_code", "cik", "resolution"]].itertuples(index=False)}
    history = {(d, c, k, si.RESOLUTION_OVERRIDE) for d in ("2015-03-30", "2016-03-15") for c, k in (("BRK", _BRK), ("GLRE", _GLRE))}
    live = {
        (d, c, k, r)
        for d in ("2026-09-08", "2026-09-09")
        for c, k, r in (("BRK", _BRK, si.RESOLUTION_OVERRIDE), ("GLRE", _GLRE, si.RESOLUTION_OVERRIDE), ("PSC", _PSC, si.RESOLUTION_EDGAR))
    }
    assert got == history | live and len(stored) == len(df) == 10
    assert set(stored.loc[stored["snapshot_date"].astype(str).str[:10] >= "2026", "source_url"]) == {si.DATAROMA_HOME_URL}
    assert len(queried) == 1 and "Pershing" in queried[0]  # stored resolution ignored; memoised per code
    first = _table_hash(sqlite_store)
    si.rebuild_roster(ctx, get_fn=ctx.edgar, listing_fn=_no_listing)
    assert _table_hash(sqlite_store) == first
    print("\n=== SANITY: full roster rebuild ===")
    print(
        f"  stale 2015-01-01 seed dropped; 2 history + 2 identical live snapshots kept = {len(stored)} rows; GLRE {_GLRE_WRONG} -> {_GLRE} "
        f"(override), PSC {_PSC_WRONG} -> {_PSC} (fresh EDGAR, 1 query); second rebuild hash {first[:12]} unchanged. Validated on the real store."
    )


@pytest.mark.parametrize("failure", ["gate", "pk"])
def test_rebuild_roster_refuses_to_write(sqlite_store, tmp_path, failure):
    """A gate failure (a CIK with no 13F activity near its snapshot) or a duplicate (snapshot_date, code) between
    history and live raises before the replace, leaving the table untouched."""
    overlap = [{"captured_at": "2026-09-08T01:00:00Z", "source_url": "https://web.archive.org/web/c", "managers": {"BRK": "Warren Buffett"}}]
    ctx, _ = _rebuild_fixture(sqlite_store, tmp_path, extra_history=overlap if failure == "pk" else None)
    if failure == "gate":
        sqlite_store.delete(si.Tables.sec13f_manager_holdings, where={"cik": [_PSC]})
    before = _table_hash(sqlite_store)
    expected = (si.SuperinvestorResolutionError, "no 13F-HR activity") if failure == "gate" else (ValueError, "duplicate")
    with pytest.raises(expected[0], match=expected[1]):
        si.rebuild_roster(ctx, get_fn=ctx.edgar, listing_fn=lambda cik: set())
    assert _table_hash(sqlite_store) == before
    print(
        f"\n=== SANITY: rebuild refuses on a {failure} failure ===\n  raised {expected[0].__name__}; table hash {before[:12]} unchanged. Validated."
    )
