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

import json
import re
from datetime import date
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


def test_upsert_roster_snapshot_writes_one_dated_snapshot(monkeypatch, sqlite_store, tmp_path):
    roster_html = (
        '<a href="holdings.php?m=GLRE">David Einhorn - Greenlight Capital</a><a href="holdings.php?m=BRK">Warren Buffett - Berkshire Hathaway</a>'
    )
    monkeypatch.setattr(si, "_http_get", lambda url: SimpleNamespace(text=roster_html))

    def fake_edgar(url):
        return SimpleNamespace(text=_SINGLE_ATOM if "Greenlight" in url else "no company-info")

    config_dir = tmp_path / "configs"
    (config_dir / "superinvestors").mkdir(parents=True)
    overrides = {"cik_overrides": {code: {"cik": cik} for code, cik in FIXTURE_CIK_OVERRIDES.items()}, "unresolvable": {}}
    (config_dir / "superinvestors" / "overrides.json").write_text(json.dumps(overrides), encoding="utf-8")
    ctx = cast(Any, SimpleNamespace(store=sqlite_store, config_dir=str(config_dir)))
    df = si.upsert_roster_snapshot(ctx, get_fn=fake_edgar)
    assert set(df["cik"]) == {"0001079114", "0001067983"}  # EDGAR + override
    assert df["snapshot_date"].nunique() == 1
    si.upsert_roster_snapshot(ctx, get_fn=fake_edgar)  # same day, again
    stored = sqlite_store.load(si.Tables.superinvestor_roster)
    assert len(stored) == 2, stored  # upserted on the PK, not doubled
    print("\n=== SANITY: live snapshot write ===")
    print(
        f"  scraped 2 managers -> {len(stored)} rows in superinvestor_roster "
        f"(resolution {df['resolution'].value_counts().to_dict()}); re-running the same day "
        "upserts the same PK rather than duplicating. Validated on the real store."
    )


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

    ctx = cast(Any, SimpleNamespace(store=sqlite_store, config_dir=str(config_dir), paths={"DATA_STORE": data_store}))
    df = si.seed_roster_history(ctx, get_fn=empty_edgar)
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

    ctx = cast(Any, SimpleNamespace(store=sqlite_store, config_dir=str(config_dir)))
    df = si.seed_roster_history(ctx, get_fn=lambda url: pytest.fail(f"no EDGAR call expected: {url}"))
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
