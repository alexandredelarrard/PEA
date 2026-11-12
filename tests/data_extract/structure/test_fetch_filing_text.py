"""
Unit tests for the edgartools-based 10-K/10-Q narrative-text fetcher
(fetch_filing_text.py). Pure-synthetic, no network -- filings and their typed
`.obj()` results (TenK/TenQ) are faked with SimpleNamespace, matching the
convention in test_fetch_8k_13d_edgar.py, so the PRIMARY (structured) + FALLBACK
(regex carve) extraction logic is exercised without a live
`Company(ticker).get_filings(...)` call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_extract.utils.edgar_extract.edgar_driver import EdgarRun, FilingStamp, plan_fetch
from src.data_extract.utils.edgar_extract.fetch_filing_text import (
    FILING_SECTION_BUSINESS,
    FILING_SECTION_MDA,
    FILING_SECTION_RISK,
    FILING_TEXT_FETCH,
    FILING_TEXT_MIN_CHARS,
    _filing_sections,
    _structured_sections,
    extract_item_sections,
    filing_text_done,
    save_filing_text,
)
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_context, identity_for, seed_index


def _pad(text: str, min_chars: int = FILING_TEXT_MIN_CHARS) -> str:
    """Pad a snippet past FILING_TEXT_MIN_CHARS so it survives the stub-length gate."""
    return text + " filler " * ((min_chars - len(text)) // 8 + 5)


# --- Real-data-shaped 10-K sample (trimmed excerpt style, like the 13D fixture) - #
_REAL_10K_TEXT_SAMPLE = f"""
Table of Contents
Item 1. Business 3
Item 1A. Risk Factors 9

Item 1. Business

{_pad("The Company designs, manufactures and markets smartphones and personal computers. See Item 1A for risks.")}

Item 1A. Risk Factors

{_pad("Our business is subject to numerous risks, including competition, regulation.")}

Item 1B. Unresolved Staff Comments

None.

Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations

{_pad("Revenue increased year over year driven by higher volumes across all segments.")}

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

We are exposed to interest rate and foreign currency risk.
"""

# A 10-K filed before 2005 has no Item 1A: Business runs to Item 2 (Properties).
_PRE_2005_10K_TEXT_SAMPLE = f"""
Item 1. Business

{_pad("The Company operates retail stores selling general merchandise and groceries.")}

Item 2. Properties

The Company owns its headquarters.
"""

# Cross-references inside sentences (Item 1 pointing at Item 1A, MD&A pointing back at Items 1 and 1A) are not headings:
# Business must keep its whole body and neither section may be re-anchored on the MD&A mention.
_CROSS_REF_10K_TEXT_SAMPLE = f"""
Item 1. Business

{_pad("The Company sells industrial equipment. These risks are described in Item 1A. Risk Factors of this report.")}
KEEP-AFTER-CROSS-REF {_pad("Our segments are Machinery and Services.")}

Item 1A. Risk Factors

{_pad("Demand for our equipment is cyclical.")}

Item 1B. Unresolved Staff Comments

None.

Item 7. Management's Discussion and Analysis

{_pad("As discussed in Item 1. Business, our segments grew. Risks described in Item 1A. Risk Factors remain.")}
{_pad("FINANCIAL STATEMENTS AND PART III TEXT THAT IS NOT BUSINESS ", 6000)}
"""

# Headings older filers use: combined Items 1 and 2, "Description of Business", "Description of Property".
_COMBINED_10K_TEXT_SAMPLE = f"""
Items 1 and 2. Business and Properties

{_pad("The Trust owns shopping centers across the United States.")}

Item 3. Legal Proceedings

None.
"""
_DESCRIPTION_10K_TEXT_SAMPLE = f"""
Item 1. Description of Business

{_pad("The Bank takes deposits and makes commercial loans.")}

Item 2. Description of Property

The Bank owns its branches.
"""

_REAL_10Q_TEXT_SAMPLE = f"""
Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations

{_pad("Net sales for the quarter grew due to strong demand in our core markets.")}

Item 3. Quantitative and Qualitative Disclosures About Market Risk

No material change.
"""


def test_extract_item_sections_10k_recovers_business_risk_and_mda():
    sections = extract_item_sections(_REAL_10K_TEXT_SAMPLE, "10-K")
    business = sections[FILING_SECTION_BUSINESS]
    assert business.startswith("The Company designs") and "Risk Factors" not in business, business[:200]
    assert "numerous risks" in sections[FILING_SECTION_RISK]
    assert "Revenue increased" in sections[FILING_SECTION_MDA]


def test_extract_item_sections_pre_2005_business_ends_at_properties():
    business = extract_item_sections(_PRE_2005_10K_TEXT_SAMPLE, "10-K")[FILING_SECTION_BUSINESS]
    assert business.startswith("The Company operates retail stores") and "headquarters" not in business


def test_extract_item_sections_ignores_cross_references():
    sections = extract_item_sections(_CROSS_REF_10K_TEXT_SAMPLE, "10-K")
    business, risk = sections[FILING_SECTION_BUSINESS], sections[FILING_SECTION_RISK]
    assert business.startswith("The Company sells industrial equipment") and "KEEP-AFTER-CROSS-REF" in business, business[:300]
    assert "FINANCIAL STATEMENTS" not in business
    assert risk.startswith("Demand for our equipment") and "FINANCIAL STATEMENTS" not in risk, risk[:300]


def test_extract_item_sections_older_business_headings():
    combined = extract_item_sections(_COMBINED_10K_TEXT_SAMPLE, "10-K")[FILING_SECTION_BUSINESS]
    assert combined.startswith("The Trust owns shopping centers") and "Legal Proceedings" not in combined
    described = extract_item_sections(_DESCRIPTION_10K_TEXT_SAMPLE, "10-K")[FILING_SECTION_BUSINESS]
    assert described.startswith("The Bank takes deposits") and "branches" not in described


def test_extract_item_sections_business_without_an_end_heading_is_not_stored():
    """A Business heading with no later Item heading would run to the end of the filing: nothing is stored."""
    text = "Item 1. Business\n\n" + _pad("Body with no end.", 5000)
    assert FILING_SECTION_BUSINESS not in extract_item_sections(text, "10-K")


def test_extract_item_sections_10q_recovers_mda_only():
    """10-Q must never carry a business or risk_factors key (Part II Item 1A is usually
    'no material change' -- deliberately not extracted, per module docstring)."""
    sections = extract_item_sections(_REAL_10Q_TEXT_SAMPLE, "10-Q")
    assert FILING_SECTION_RISK not in sections and FILING_SECTION_BUSINESS not in sections
    assert "Net sales" in sections[FILING_SECTION_MDA]


def test_extract_item_sections_below_min_chars_returns_empty():
    assert extract_item_sections("Item 1A. Risk Factors\n\nToo short.", "10-K") == {}


# --- PRIMARY: edgartools structured TenK/TenQ access -------------------------- #
def _fake_ten_k(risk_factors=None, management_discussion=None, business=None):
    return SimpleNamespace(business=business, risk_factors=risk_factors, management_discussion=management_discussion)


def test_structured_sections_reads_ten_k_named_properties():
    obj = _fake_ten_k(
        risk_factors=_pad("structured risk body"), management_discussion=_pad("structured mda body"), business=_pad("structured business body")
    )
    sections = _structured_sections(obj, "10-K")
    assert "structured business body" in sections[FILING_SECTION_BUSINESS]
    assert "structured risk body" in sections[FILING_SECTION_RISK]
    assert "structured mda body" in sections[FILING_SECTION_MDA]


def test_structured_sections_discards_a_stub_under_min_chars():
    """A cross-reference/TOC stub (edgartools found the heading but no real body)
    must be discarded here so the caller falls through to the regex carve,
    exactly like fetch_13d_edgar's has_structured_data-empty fallback trigger."""
    obj = _fake_ten_k(risk_factors="See Part I, Item 1A.", management_discussion=_pad("real mda"))
    sections = _structured_sections(obj, "10-K")
    assert FILING_SECTION_RISK not in sections
    assert FILING_SECTION_MDA in sections


def test_structured_sections_10q_uses_part_i_item_2():
    class FakeTenQ:
        def __getitem__(self, key):
            assert key == "Part I, Item 2"
            return _pad("structured 10-Q mda")

    sections = _structured_sections(FakeTenQ(), "10-Q")
    assert FILING_SECTION_RISK not in sections
    assert "structured 10-Q mda" in sections[FILING_SECTION_MDA]


def test_structured_sections_survives_obj_attribute_errors():
    """A TenK whose properties raise (a real edgartools parse edge case) must
    yield an empty dict, not propagate -- the caller then falls back to the
    regex carve on filing.text()."""

    class BrokenTenK:
        @property
        def risk_factors(self):
            raise RuntimeError("parser exploded")

    assert _structured_sections(BrokenTenK(), "10-K") == {}


# --- Combined PRIMARY + FALLBACK ---------------------------------------------- #
def _fake_filing(*, form="10-K", accession="0001-24-000099", filing_date="2024-03-01", period_of_report="2023-12-31", obj=None, text=None):
    filing = SimpleNamespace(
        accession_number=accession,
        form=form,
        filing_date=filing_date,
        period_of_report=period_of_report,
    )
    filing.obj = (lambda: obj) if obj is not None else (lambda: (_ for _ in ()).throw(RuntimeError("no parse")))
    filing.text = (lambda: text) if text is not None else (lambda: (_ for _ in ()).throw(RuntimeError("no text")))
    return filing


def test_filing_sections_falls_back_to_regex_carve_when_structured_empty():
    """The structured parse returning nothing for BOTH sections must not lose the
    filing -- the regex carve over filing.text() recovers them."""
    obj = _fake_ten_k(risk_factors=None, management_discussion=None)
    filing = _fake_filing(obj=obj, text=_REAL_10K_TEXT_SAMPLE)
    sections = _filing_sections(filing)
    assert "smartphones" in sections[FILING_SECTION_BUSINESS]
    assert "numerous risks" in sections[FILING_SECTION_RISK]
    assert "Revenue increased" in sections[FILING_SECTION_MDA]


def test_filing_sections_only_falls_back_for_the_missing_section():
    """When structured recovers risk_factors but NOT mda, only mda should come
    from the fallback carve -- the structured risk_factors body must pass
    through untouched (not be re-derived/overwritten by the regex carve)."""
    obj = _fake_ten_k(risk_factors=_pad("STRUCTURED RISK BODY MARKER"), management_discussion=None)
    filing = _fake_filing(obj=obj, text=_REAL_10K_TEXT_SAMPLE)
    sections = _filing_sections(filing)
    assert "STRUCTURED RISK BODY MARKER" in sections[FILING_SECTION_RISK]
    assert "Revenue increased" in sections[FILING_SECTION_MDA]


def test_filing_sections_business_prefers_the_carve_and_keeps_edgartools_otherwise():
    """Business takes the regex carve over edgartools' Item 1 when the carve finds it, and keeps edgartools' when not."""
    obj = _fake_ten_k(risk_factors=_pad("STRUCTURED RISK"), management_discussion=_pad("STRUCTURED MDA"), business=_pad("STRUCTURED BUSINESS"))
    carved = _filing_sections(_fake_filing(obj=obj, text=_REAL_10K_TEXT_SAMPLE))
    assert "smartphones" in carved[FILING_SECTION_BUSINESS] and "STRUCTURED RISK" in carved[FILING_SECTION_RISK]
    kept = _filing_sections(_fake_filing(obj=obj, text=_REAL_10Q_TEXT_SAMPLE))
    assert "STRUCTURED BUSINESS" in kept[FILING_SECTION_BUSINESS]


def test_filing_sections_survives_a_totally_unparseable_filing():
    """Both .obj() and .text() failing (a genuinely bad filing) must return an
    empty dict rather than raising -- the caller (FILING_TEXT_FETCH.parse) then
    emits zero rows for it (one empty-filing marker in the driver)."""
    filing = _fake_filing()  # obj=None -> raises; text=None -> raises
    assert _filing_sections(filing) == {}


# --- Per-filing parse ------------------------------------------------------- #
def _parse(filing) -> pd.DataFrame:
    return FILING_TEXT_FETCH.parse("AAPL", "0000320193", FilingStamp.of(filing, "0000320193"), identity_for({"AAPL": "320193"}))[
        Tables.filing_risk_text
    ]


def test_filing_text_parse_returns_all_sections_stamped_with_the_filing_date():
    obj = _fake_ten_k(risk_factors=_pad("risk body"), management_discussion=_pad("mda body"), business=_pad("business body"))
    out = _parse(_fake_filing(accession="0001-new", filing_date="2024-06-01", obj=obj))
    assert set(out["accession_number"]) == {"0001-new"}
    assert set(out["section"]) == {FILING_SECTION_BUSINESS, FILING_SECTION_RISK, FILING_SECTION_MDA}
    assert (out["filed"] == pd.Timestamp("2024-06-01")).all()


def test_filing_text_parse_stores_an_empty_row_per_expected_section_it_cannot_read():
    """An unreadable 10-K gets an empty-text row for Items 1, 1A and 7; a 10-K/A or 10-Q expects MD&A only, and a
    10-K/A keeps any other section it does carry."""
    ten_k = _parse(_fake_filing(accession="0001-bad"))  # obj/text both raise
    assert set(ten_k["section"]) == {FILING_SECTION_BUSINESS, FILING_SECTION_RISK, FILING_SECTION_MDA}
    assert ten_k["text"].isna().all() and ten_k["n_words"].eq(0).all()
    assert list(_parse(_fake_filing(form="10-Q", accession="0001-q"))["section"]) == [FILING_SECTION_MDA]
    amended = _parse(_fake_filing(form="10-K/A", accession="0001-a", obj=_fake_ten_k(risk_factors=_pad("risk body"))))
    assert dict(zip(amended["section"], amended["text"].notna(), strict=True)) == {FILING_SECTION_MDA: False, FILING_SECTION_RISK: True}
    print("\n=== SANITY CHECK: an expected section the filing lacks is stored empty ===")
    print(f"  unreadable 10-K -> {sorted(ten_k['section'])} with NULL text; 10-Q -> [mda]; 10-K/A -> mda empty + its risk_factors")


def _stored(accession: str, form: str, sections: dict[str, str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "accession_number": accession,
                "form": form,
                "filed": pd.Timestamp("2024-03-01"),
                "period_of_report": pd.Timestamp("2023-12-31"),
                "section": section,
                "text": text,
                "n_words": len(text.split()) if text else 0,
            }
            for section, text in sections.items()
        ]
    )


def test_a_filing_is_done_once_every_expected_section_has_a_row(sqlite_store):
    """Done = a row, text or empty, per expected section. A 10-K missing Item 1 and a legacy empty-filing marker are
    listed again; an amendment and a 10-Q need MD&A only."""
    table = Tables.filing_risk_text
    sqlite_store.save(table, _stored("k-full", "10-K", {"business": None, "risk_factors": _pad("r"), "mda": _pad("m")}))
    sqlite_store.save(table, _stored("k-old", "10-K", {"risk_factors": _pad("r"), "mda": _pad("m")}))
    sqlite_store.save(table, _stored("k-amend", "10-K/A", {"mda": None}))
    sqlite_store.save(table, _stored("q-full", "10-Q", {"mda": _pad("m")}))
    sqlite_store.save(table, _stored("q-legacy", "10-Q", {"_empty": None}))
    context = SimpleNamespace(store=sqlite_store)

    done = filing_text_done(context, ["AAPL", "MSFT"], None)  # type: ignore[arg-type]

    assert done == {"AAPL": {"k-full", "k-amend", "q-full"}, "MSFT": set()}, done
    print("\n=== SANITY CHECK: filing-text done rule ===")
    print(f"  done {sorted(done['AAPL'])}; listed again: k-old (no business row), q-legacy (marker only). Validated.")


def test_a_re_read_adds_the_missing_section_and_never_blanks_stored_text(sqlite_store):
    """A 10-K re-read for its missing Item 1 stores it, while its empty re-read MD&A keeps the stored text; the filing
    is then done and not read again."""
    table = Tables.filing_risk_text
    sqlite_store.save(table, _stored("k-old", "10-K", {"risk_factors": _pad("stored risk"), "mda": _pad("stored mda")}))
    context = SimpleNamespace(store=sqlite_store, log=SimpleNamespace(warning=lambda *a, **k: None))
    re_read = _stored("k-old", "10-K", {"business": _pad("new business"), "risk_factors": None, "mda": None})

    counts, failed = save_filing_text(context, FILING_TEXT_FETCH, "AAPL", {table: [re_read]})  # type: ignore[arg-type]

    rows = sqlite_store.load(table).set_index("section")
    assert not failed and counts == {table: 1}, counts
    assert rows.loc["mda", "text"].startswith("stored mda") and rows.loc["risk_factors", "text"].startswith("stored risk")
    assert rows.loc["business", "text"].startswith("new business")
    assert filing_text_done(context, ["AAPL"], None) == {"AAPL": {"k-old"}}  # type: ignore[arg-type]
    print("\n=== SANITY CHECK: re-read save ===")
    print("  1 row saved (business); the empty re-read mda/risk rows dropped, stored text kept; k-old now done. Validated.")


def test_a_null_or_mixed_form_still_counts_the_accession_done(sqlite_store):
    """An accession is a 10-K when any of its rows says so, and a NULL form expects MD&A only, so neither is listed
    again every night."""
    table = Tables.filing_risk_text
    mixed = _stored("k-mixed", "10-K", {"business": None, "risk_factors": _pad("r"), "mda": _pad("m")})
    mixed.loc[mixed["section"].eq("business"), "form"] = None
    sqlite_store.save(table, mixed)
    sqlite_store.save(table, _stored("k-short", "10-K", {"business": None, "mda": _pad("m")}).assign(form=["10-K", None]))
    sqlite_store.save(table, _stored("q-null", "10-Q", {"mda": _pad("m")}).assign(form=None))

    done = filing_text_done(SimpleNamespace(store=sqlite_store), ["AAPL"], None)  # type: ignore[arg-type]

    assert done == {"AAPL": {"k-mixed", "q-null"}}, done
    print("\n=== SANITY CHECK: NULL / mixed form ===")
    print("  k-mixed (one NULL-form row) done; q-null (NULL form, mda) done; k-short (10-K, no risk row) listed again. Validated.")


def test_a_key_that_read_nothing_saves_nothing_without_failing(sqlite_store):
    """Every filing of the key failed to read: no frame, no row, no save failure, so the filings are simply listed again."""
    context = SimpleNamespace(store=sqlite_store, log=SimpleNamespace(warning=lambda *a, **k: None))

    counts, failed = save_filing_text(context, FILING_TEXT_FETCH, "AAPL", {})  # type: ignore[arg-type]

    assert counts == {} and not failed and not sqlite_store.exists(Tables.filing_risk_text)
    print("\n=== SANITY CHECK: empty key save ===")
    print("  no frames -> counts {}, failed_save False, no table written. Validated.")


_AAPL = {"AAPL": "320193"}
_INDEX = [
    (320193, "Apple Inc", "10-K", "2024-11-01", "0000320193-24-000123"),
    (320193, "Apple Inc", "10-Q", "2025-01-31", "0000320193-25-000008"),
    (320193, "Apple Inc", "10-K", "2023-11-03", "0000320193-23-000106"),
]


def test_the_planner_lists_only_filings_missing_an_expected_section_and_full_lists_all(tmp_path, sqlite_store):
    """Through `plan_fetch`: the stored complete 10-K and 10-Q are skipped, the 10-K without Item 1 is listed; `full`
    lists all three."""
    ctx = fake_context(tmp_path, sqlite_store, list(_AAPL), ciks=list(_AAPL.values()))
    seed_index(ctx, _INDEX)
    table = Tables.filing_risk_text
    sqlite_store.save(table, _stored(_INDEX[0][4], "10-K", {"business": None, "risk_factors": _pad("r"), "mda": _pad("m")}))
    sqlite_store.save(table, _stored(_INDEX[1][4], "10-Q", {"mda": _pad("m")}))
    sqlite_store.save(table, _stored(_INDEX[2][4], "10-K", {"risk_factors": _pad("r"), "mda": _pad("m")}))
    cik_map = pd.DataFrame({"ticker": ["AAPL"], "cik": ["0000320193"]})

    def plan(full: bool) -> list[str]:
        run = EdgarRun(["AAPL"], cik_map, identity_for(_AAPL), pd.Timestamp("2025-03-01"), 5, full=full)
        work = plan_fetch(ctx, FILING_TEXT_FETCH, run)
        return sorted(work.units["AAPL"]["accession"]) if "AAPL" in work.units else []

    daily, full = plan(False), plan(True)

    assert daily == [_INDEX[2][4]], daily
    assert full == sorted(row[4] for row in _INDEX), full
    print("\n=== SANITY CHECK: filing-text planner ===")
    print(f"  daily lists {daily} (10-K without business); -F lists all {len(full)}. Validated.")


def test_sanity_check_prints_conclusion():
    print("\n=== SANITY CHECK: edgartools 10-K/10-Q filing-text extraction ===")
    print("  Structured PRIMARY path (TenK.business/.risk_factors/.management_discussion,")
    print("  TenQ['Part I, Item 2']) is used first; a stub/cross-reference result under")
    print("  FILING_TEXT_MIN_CHARS is discarded so ONLY that missing section falls back")
    print("  to the hardened regex carve over filing.text() -- the other, successfully")
    print("  structured section passes through untouched (not re-derived/overwritten).")
    print("  10-K Item 1 (Business) and Item 1A match headings that start a line: in-sentence cross-references neither")
    print("  start nor cut a section; Business ends at the next Item 1A/1B/2/3 heading, else nothing is stored.")
    print("  Both .obj() and .text() failing yields an empty dict, not a crash.")
    print("  FILING_TEXT_FETCH.parse returns one filing's rows keyed by destination table; the")
    print("  shared driver lists only unstored index filings. No local HTML cache involved anywhere.")
    print("  Validated.")
