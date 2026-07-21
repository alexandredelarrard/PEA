"""
Wikipedia article resolution for pageviews
(src/data_extract/utils/behavioral/fetch_wiki_pageviews.py).

The naive suffix-strip heuristic misses ~27 S&P names because the "Security" name
carries list artifacts — "Coca-Cola Company (The)", "Alphabet Inc. (Class A)",
"Lilly (Eli)", "Deere & Company" — so the guessed article 404s. The fix resolves the
title via Wikipedia search on a CLEANED name. Search is stubbed (no network).
"""
from __future__ import annotations

from src.data_extract.utils.behavioral.fetch_wiki_pageviews import (
    _clean_company_name, _resolve_wiki_article,
)


def test_clean_company_name_handles_sp_artifacts():
    assert _clean_company_name("Coca-Cola Company (The)") == "The Coca-Cola Company"
    assert _clean_company_name("Home Depot (The)") == "The Home Depot"
    assert _clean_company_name("Alphabet Inc. (Class A)") == "Alphabet Inc."
    assert _clean_company_name("Lilly (Eli)") == "Eli Lilly"          # Surname (First) -> First Surname
    assert _clean_company_name("Deere & Company") == "Deere & Company"  # ordinary name untouched
    assert _clean_company_name("S&P Global") == "S&P Global"
    print("\n=== SANITY CHECK: S&P name cleaning ===")
    print("  '(The)'->'The X'; '(Class A)' dropped; 'Lilly (Eli)'->'Eli Lilly'; plain names kept. Validated.")


def test_resolve_uses_search_hit_then_falls_back():
    seen = {}

    def fake_search(q):
        seen["q"] = q
        return "The Coca-Cola Company"                           # a search hit

    art = _resolve_wiki_article("Coca-Cola Company (The)", search_fn=fake_search)
    assert art == "The_Coca-Cola_Company"                        # underscored article title
    assert seen["q"] == "The Coca-Cola Company"                  # searched the CLEANED name

    # search returns nothing -> fall back to the CLEANED name (usually the real article)
    assert _resolve_wiki_article("Coca-Cola Company (The)", search_fn=lambda q: None) == "The_Coca-Cola_Company"
    assert _resolve_wiki_article("Apple Inc.", search_fn=lambda q: None) == "Apple_Inc."

    # search raises -> cleaned-name fallback, still no crash
    def boom(q):
        raise RuntimeError("network down")
    assert _resolve_wiki_article("Alphabet Inc. (Class A)", search_fn=boom) == "Alphabet_Inc."
    print("\n=== SANITY CHECK: article resolver ===")
    print("  search hit -> underscored title (cleaned query); miss/error -> cleaned-name fallback "
          "('The_Coca-Cola_Company', 'Alphabet_Inc.'). Validated.")


if __name__ == "__main__":
    test_clean_company_name_handles_sp_artifacts()
    test_resolve_uses_search_hit_then_falls_back()
