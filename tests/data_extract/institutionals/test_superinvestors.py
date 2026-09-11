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

from datetime import date
from types import SimpleNamespace

import pytest

from src.data_extract.utils.institutionals import fetch_superinvestors as si


_SINGLE_ATOM = ('<?xml version="1.0"?><feed><company-info>'
                '<cik>0001079114</cik><cik-href>x</cik-href>'
                '<conformed-name>GREENLIGHT CAPITAL INC</conformed-name>'
                '</company-info></feed>')
_MULTI_ATOM = ('<?xml version="1.0"?><feed>'
               '<company-info name="ARRAY(0x1)"><cik>0001336528</cik></company-info>'
               '<company-info name="ARRAY(0x2)"><cik>0002026053</cik></company-info></feed>')


def test_pad_cik_canonical_10_digit():
    assert si.pad_cik("1067983") == "0001067983"
    assert si.pad_cik("1067983.0") == "0001067983"        # float artifact tolerated
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
    assert [r["code"] for r in roster] == ["BRK", "GLRE"]   # deduped, non-manager link dropped
    assert roster[0]["name"] == "Warren Buffett - Berkshire Hathaway"   # "Updated ..." stripped
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
    assert si._pick_best_match([("0001079114", "GREENLIGHT CAPITAL INC")],
                               "Greenlight Capital") == ("0001079114", "GREENLIGHT CAPITAL INC")
    # multi WITH names -> highest token overlap
    pairs = [("0000000001", "ACME HOLDINGS"), ("0000000002", "PERSHING SQUARE CAPITAL")]
    assert si._pick_best_match(pairs, "Pershing Square")[0] == "0000000002"
    # multi WITHOUT names -> EDGAR's first (most-relevant) block
    assert si._pick_best_match([("0001336528", ""), ("0002026053", "")], "Pershing Square")[0] == "0001336528"
    assert si._pick_best_match([], "x") is None
    print("\n=== SANITY: best-CIK pick ===")
    print("  single trusted; multi by token overlap; no-name multi -> first block; empty -> None. Validated.")


def test_edgar_cik_for_name_stubbed():
    calls = {}

    def fake_get(url):
        calls["url"] = url
        return SimpleNamespace(text=_SINGLE_ATOM)

    cik, filer = si._edgar_cik_for_name("David Einhorn - Greenlight Capital", get_fn=fake_get)
    assert cik == "0001079114" and "GREENLIGHT" in filer
    assert "company=Greenlight" in calls["url"]              # searched the FUND part, url-quoted

    def boom(url):
        raise RuntimeError("network down")
    assert si._edgar_cik_for_name("X - Y Capital", get_fn=boom) == (None, None)   # no raise
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

    resolve = si._make_resolver(fake_get, {"GLRE": ["Greenlight Capital", "Greenlight Re"]})
    # the name in hand does not resolve; the older one does
    assert resolve("GLRE", "David Einhorn - Some Rebrand") == ("0001079114", si.RESOLUTION_EDGAR)
    n_after_first = len(calls)
    assert n_after_first == 2                                # the rebrand, then the old name
    for _ in range(5):                                       # same code, 5 more snapshots
        resolve("GLRE", "David Einhorn - Some Rebrand")
    assert len(calls) == n_after_first                       # memoised: no extra EDGAR calls
    assert resolve("BRK", "anything") == ("0001067983", si.RESOLUTION_OVERRIDE)
    assert len(calls) == n_after_first                       # an override never hits the network
    print("\n=== SANITY: per-code memoised resolution ===")
    print(f"  6 snapshot-rows of one code cost {len(calls)} EDGAR calls (2 = failed current "
          "name + successful older name); an override costs 0. Validated.")


def test_snapshot_rows_shape_and_padding():
    rows = si.snapshot_rows(
        [{"code": "BRK", "name": "Warren Buffett - Berkshire"},
         {"code": "CMAFX", "name": "Century Management"}],
        date(2016, 1, 1), "https://web.archive.org/web/2016/x",
        lambda code, name: (("1067983", si.RESOLUTION_OVERRIDE) if code == "BRK"
                            else (None, si.RESOLUTION_UNRESOLVED)))
    assert [r["cik"] for r in rows] == ["0001067983", None]   # padded; unresolved -> NULL, not ""
    assert {r["snapshot_date"] for r in rows} == {date(2016, 1, 1)}
    assert set(rows[0]) == {"snapshot_date", "dataroma_code", "manager_name", "cik",
                            "resolution", "source_url"}
    print("\n=== SANITY: superinvestor_roster row shape ===")
    print(f"  {len(rows)} rows keyed (snapshot_date, dataroma_code); CIK zero-padded to 10; "
          "an unresolved manager keeps its row with cik=None (never dropped). Validated.")


def test_unresolved_manager_raises_unless_recorded():
    """D22's gate: 100% resolution, or every exception named with its reason. An unresolved
    manager silently leaving the eligible pool IS the survivorship bug, so it must be loud."""
    unknown = [{"snapshot_date": date(2016, 1, 1), "dataroma_code": "ZZZ",
                "manager_name": "Nobody - Unlisted Boutique", "cik": None,
                "resolution": si.RESOLUTION_UNRESOLVED, "source_url": "x"}]
    with pytest.raises(si.SuperinvestorResolutionError, match="Unlisted Boutique"):
        si.assert_fully_resolved(unknown)

    recorded = [dict(unknown[0], dataroma_code="CMAFX", manager_name="Century Management")]
    assert si.assert_fully_resolved(recorded) == ["CMAFX"]        # recorded -> passes, reported
    assert si.assert_fully_resolved([dict(unknown[0], cik="0001067983",
                                          resolution=si.RESOLUTION_EDGAR)]) == []
    print("\n=== SANITY: resolution gate ===")
    print(f"  an unknown unresolved code RAISES; the {len(si.SUPERINVESTOR_UNRESOLVABLE)} "
          f"recorded exceptions {sorted(si.SUPERINVESTOR_UNRESOLVABLE)} pass and are returned "
          "for the caller to report. Validated.")


def test_upsert_roster_snapshot_writes_one_dated_snapshot(monkeypatch, sqlite_store):
    roster_html = ('<a href="holdings.php?m=GLRE">David Einhorn - Greenlight Capital</a>'
                   '<a href="holdings.php?m=BRK">Warren Buffett - Berkshire Hathaway</a>')
    monkeypatch.setattr(si, "_http_get", lambda url: SimpleNamespace(text=roster_html))

    def fake_edgar(url):
        return SimpleNamespace(text=_SINGLE_ATOM if "Greenlight" in url else "no company-info")

    ctx = SimpleNamespace(store=sqlite_store)
    df = si.upsert_roster_snapshot(ctx, get_fn=fake_edgar)
    assert set(df["cik"]) == {"0001079114", "0001067983"}         # EDGAR + override
    assert df["snapshot_date"].nunique() == 1
    si.upsert_roster_snapshot(ctx, get_fn=fake_edgar)             # same day, again
    stored = sqlite_store.load(si.Tables.superinvestor_roster)
    assert len(stored) == 2, stored                               # upserted on the PK, not doubled
    print("\n=== SANITY: live snapshot write ===")
    print(f"  scraped 2 managers -> {len(stored)} rows in superinvestor_roster "
          f"(resolution {df['resolution'].value_counts().to_dict()}); re-running the same day "
          "upserts the same PK rather than duplicating. Validated on the real store.")
