"""
Superinvestors roster builder — pure logic
(src/data_extract/utils/prices/fetch_superinvestors.py).

CIKs are resolved from SEC EDGAR company search (fund NAME -> 13F CIK), decoupled
from any local 13F cache. Network is stubbed; we test the Dataroma roster parse,
the EDGAR atom parse (lower-case `<cik>`; single- vs multi-match), the best-match
pick, name->CIK resolution, and the {cik: name} JSON build (Dataroma + override).
"""
from __future__ import annotations

from types import SimpleNamespace

from src.data_extract.utils.prices import fetch_superinvestors as si


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


def test_build_superinvestors_json_cik_to_name(monkeypatch, tmp_path):
    roster_html = ('<a href="holdings.php?m=GLRE">David Einhorn - Greenlight Capital</a>'
                   '<a href="holdings.php?m=BRK">Warren Buffett - Berkshire Hathaway</a>'
                   '<a href="holdings.php?m=zz">Nobody - Unlisted Boutique</a>')
    monkeypatch.setattr(si, "_http_get", lambda url: SimpleNamespace(text=roster_html))
    monkeypatch.setattr(si, "SUPERINVESTOR_CIK_OVERRIDES", {"BRK": "1067983"})

    def fake_edgar(url):                                     # Greenlight resolves; boutique doesn't
        return SimpleNamespace(text=_SINGLE_ATOM if "Greenlight" in url else "no company-info")

    ctx = SimpleNamespace(paths={"DATA_STORE": tmp_path})
    out = si.build_superinvestors_json(ctx, get_fn=fake_edgar)
    m = out["cik_to_name"]

    assert m["0001067983"] == "Warren Buffett - Berkshire Hathaway"   # via override
    assert m["0001079114"] == "David Einhorn - Greenlight Capital"    # via EDGAR
    assert out["n_roster"] == 3 and out["n_resolved"] == 2            # boutique unresolved
    assert si.load_superinvestors(ctx)["cik_to_name"] == m            # persisted + reloadable
    print("\n=== SANITY: build {cik: name} roster JSON ===")
    print(f"  {out['n_resolved']}/{out['n_roster']} resolved (EDGAR + override); "
          "unresolved dropped + logged; JSON round-trips. Validated.")


if __name__ == "__main__":
    test_pad_cik_canonical_10_digit()
    test_parse_dataroma_roster_strips_updated_suffix()
    test_parse_edgar_matches_lowercase_cik_single_and_multi()
    test_pick_best_match()
    test_edgar_cik_for_name_stubbed()
