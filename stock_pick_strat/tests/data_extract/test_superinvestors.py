"""
Superinvestors roster builder — pure logic
(src/data_extract/utils/prices/fetch_superinvestors.py).

Network (Dataroma), cached-zip reads and the DB AUM query are thin IO wrappers;
here we test the deterministic pieces: roster parse, CIK canonicalization, name->CIK
resolution (manual override + fuzzy filer-name match), and AUM ranking + weighting.
"""
from __future__ import annotations

import pytest

from src.data_extract.utils.prices import fetch_superinvestors as si


def test_pad_cik_canonical_10_digit():
    assert si._pad_cik("1067983") == "0001067983"
    assert si._pad_cik("1067983.0") == "0001067983"      # float artifact tolerated
    assert si._pad_cik("0001067983") == "0001067983"     # already padded -> stable
    assert si._pad_cik(1067983) == "0001067983"          # int tolerated
    assert si._pad_cik("") == "" and si._pad_cik("N/A") == ""
    print("\n=== SANITY: CIK canonicalization ===")
    print("  '1067983' / '1067983.0' / int -> '0001067983'; junk -> ''. Validated.")


def test_parse_dataroma_roster():
    # real Dataroma link text carries a trailing "Updated <date>" that must be stripped
    html = """
    <table>
      <tr><td><a href="holdings.php?m=BRK">Warren Buffett - Berkshire Hathaway Updated 15 May 2026</a></td></tr>
      <tr><td><a href="/m/holdings.php?m=psc">Bill Ackman - Pershing Square Updated 10 Jul 2026</a></td></tr>
      <tr><td><a href="holdings.php?m=BRK">Berkshire (dupe link)</a></td></tr>
      <tr><td><a href="/m/managers.php">All managers</a></td></tr>
    </table>"""
    roster = si._parse_dataroma_roster(html)
    codes = [r["code"] for r in roster]
    assert codes == ["BRK", "psc"]                       # deduped, order preserved, non-manager link ignored
    assert roster[0]["name"] == "Warren Buffett - Berkshire Hathaway"   # "Updated ..." stripped
    # the date must NOT leak into the matching tokens (real bug caught live)
    assert si._name_tokens(si._fund_part(roster[1]["name"])) == frozenset({"PERSHING", "SQUARE"})
    print("\n=== SANITY: Dataroma roster parse ===")
    print(f"  parsed {len(roster)} managers ({codes}); 'Updated <date>' stripped, dupe + "
          f"non-holdings link dropped; fund tokens clean ({{PERSHING, SQUARE}}). Validated.")


def test_resolve_ciks_override_and_fuzzy():
    roster = [
        {"code": "BRK", "name": "Warren Buffett - Berkshire Hathaway"},
        {"code": "psc", "name": "Bill Ackman - Pershing Square"},
        {"code": "xx", "name": "Nobody - Unlisted Boutique Advisers"},   # no match -> None
    ]
    # (name-tokens, cik, raw filer name) index as built from cached 13F SUBMISSION.tsv
    filer_index = [
        (si._name_tokens("PERSHING SQUARE CAPITAL MANAGEMENT LP"), "0001336528",
         "PERSHING SQUARE CAPITAL MANAGEMENT LP"),
        (si._name_tokens("BERKSHIRE HATHAWAY INC"), "0001067983", "BERKSHIRE HATHAWAY INC"),
    ]
    overrides = {"BRK": "1067983"}                       # BRK resolved by override, not fuzzy

    resolved = si._resolve_ciks(roster, filer_index, overrides)
    by_code = {r["code"]: r for r in resolved}
    assert by_code["BRK"]["cik"] == "0001067983" and by_code["BRK"]["matched_filer"] == "override"
    assert by_code["psc"]["cik"] == "0001336528"          # fuzzy: {PERSHING,SQUARE} fully contained
    assert by_code["xx"]["cik"] is None                   # nothing above threshold -> unresolved
    print("\n=== SANITY: name -> CIK resolution ===")
    print("  BRK via override; 'Pershing Square' fuzzy-matched to filer CIK; unlisted -> None. Validated.")


def test_rank_and_weight_by_aum_positive_decay():
    resolved = [
        {"code": "a", "name": "A", "cik": "0000000001"},
        {"code": "b", "name": "B", "cik": "0000000002"},
        {"code": "c", "name": "C", "cik": "0000000003"},
        {"code": "d", "name": "D", "cik": "0000000004"},   # no AUM -> dropped (not in our 13F)
    ]
    aum = {"0000000001": 300.0, "0000000002": 900.0, "0000000003": 100.0}
    top = si._rank_and_weight(resolved, aum, top_n=2, weighting="rank")

    assert [m["cik"] for m in top] == ["0000000002", "0000000001"]   # sorted by AUM desc
    assert [m["rank"] for m in top] == [1, 2]
    w = [m["weight"] for m in top]
    assert w[0] > w[1] > 0                                # rank-decay, all positive
    assert abs(sum(w) - 1.0) < 1e-9                       # normalized
    assert all(m["cik"] != "0000000004" for m in top)     # AUM-less manager excluded
    print("\n=== SANITY: AUM rank + rank-decay weights ===")
    print(f"  top_n=2 by AUM -> {[m['cik'][-1] for m in top]} (900,300); "
          f"weights {[round(x,3) for x in w]} sum 1, top heaviest; AUM-less dropped. Validated.")


if __name__ == "__main__":
    test_pad_cik_canonical_10_digit()
    test_parse_dataroma_roster()
    test_resolve_ciks_override_and_fuzzy()
    test_rank_and_weight_by_aum_positive_decay()
