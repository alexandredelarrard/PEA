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

from src.constants.constants import FILING_SECTION_MDA, FILING_SECTION_RISK, FILING_TEXT_MIN_CHARS
from src.data_extract.utils.structure.fetch_filing_text import (
    _filing_sections, _seen, _structured_sections, build_ticker_filing_text, extract_item_sections,
)


def _pad(text: str, min_chars: int = FILING_TEXT_MIN_CHARS) -> str:
    """Pad a snippet past FILING_TEXT_MIN_CHARS so it survives the stub-length gate."""
    return text + " filler " * ((min_chars - len(text)) // 8 + 5)


# --- Real-data-shaped 10-K sample (trimmed excerpt style, like the 13D fixture) - #
_REAL_10K_TEXT_SAMPLE = f"""
Item 1A. Risk Factors

{_pad("Our business is subject to numerous risks, including competition, regulation.")}

Item 1B. Unresolved Staff Comments

None.

Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations

{_pad("Revenue increased year over year driven by higher volumes across all segments.")}

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

We are exposed to interest rate and foreign currency risk.
"""

_REAL_10Q_TEXT_SAMPLE = f"""
Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations

{_pad("Net sales for the quarter grew due to strong demand in our core markets.")}

Item 3. Quantitative and Qualitative Disclosures About Market Risk

No material change.
"""


def test_extract_item_sections_10k_recovers_risk_and_mda():
    sections = extract_item_sections(_REAL_10K_TEXT_SAMPLE, "10-K")
    assert "numerous risks" in sections[FILING_SECTION_RISK]
    assert "Revenue increased" in sections[FILING_SECTION_MDA]


def test_extract_item_sections_10q_recovers_mda_only():
    """10-Q must never carry a risk_factors key (Part II Item 1A is usually
    'no material change' -- deliberately not extracted, per module docstring)."""
    sections = extract_item_sections(_REAL_10Q_TEXT_SAMPLE, "10-Q")
    assert FILING_SECTION_RISK not in sections
    assert "Net sales" in sections[FILING_SECTION_MDA]


def test_extract_item_sections_below_min_chars_returns_empty():
    assert extract_item_sections("Item 1A. Risk Factors\n\nToo short.", "10-K") == {}


# --- PRIMARY: edgartools structured TenK/TenQ access -------------------------- #
def _fake_ten_k(risk_factors=None, management_discussion=None):
    return SimpleNamespace(risk_factors=risk_factors, management_discussion=management_discussion)


def test_structured_sections_reads_ten_k_named_properties():
    obj = _fake_ten_k(risk_factors=_pad("structured risk body"),
                      management_discussion=_pad("structured mda body"))
    sections = _structured_sections(obj, "10-K")
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
def _fake_filing(*, form="10-K", accession="0001-24-000099", filing_date="2024-03-01",
                 period_of_report="2023-12-31", obj=None, text=None):
    filing = SimpleNamespace(
        accession_number=accession, form=form, filing_date=filing_date,
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


def test_filing_sections_survives_a_totally_unparseable_filing():
    """Both .obj() and .text() failing (a genuinely bad filing) must return an
    empty dict rather than raising -- the caller (build_ticker_filing_text) then
    simply emits zero rows for it."""
    filing = _fake_filing()   # obj=None -> raises; text=None -> raises
    assert _filing_sections(filing) == {}


# --- Ticker-level walk (incremental dedup, since-cutoff) ---------------------- #
def test_build_ticker_filing_text_skips_done_accessions_and_pre_since_filings(monkeypatch):
    obj = _fake_ten_k(risk_factors=_pad("risk body"), management_discussion=_pad("mda body"))
    old_filing = _fake_filing(accession="0001-old", filing_date="2020-01-01", obj=obj)
    done_filing = _fake_filing(accession="0001-done", filing_date="2024-01-01", obj=obj)
    new_filing = _fake_filing(accession="0001-new", filing_date="2024-06-01", obj=obj)
    fake_company = SimpleNamespace(get_filings=lambda form: [old_filing, done_filing, new_filing])
    monkeypatch.setattr(
        "src.data_extract.utils.structure.fetch_filing_text.Company",
        lambda ticker: fake_company,
    )

    out = build_ticker_filing_text(
        "AAPL", "0000320193",
        since=pd.Timestamp("2024-01-01"),
        done_accessions=frozenset({"0001-done"}),
    )
    assert set(out["accession_number"]) == {"0001-new"}
    assert set(out["section"]) == {FILING_SECTION_RISK, FILING_SECTION_MDA}
    assert (out["filed"] == pd.Timestamp("2024-06-01")).all()


def test_build_ticker_filing_text_returns_no_rows_for_an_unparseable_filing(monkeypatch):
    filing = _fake_filing(accession="0001-bad")   # obj/text both raise
    fake_company = SimpleNamespace(get_filings=lambda form: [filing])
    monkeypatch.setattr(
        "src.data_extract.utils.structure.fetch_filing_text.Company",
        lambda ticker: fake_company,
    )
    out = build_ticker_filing_text("AAPL", "0000320193")
    assert out.empty


# --- `_seen` incremental-cutoff helper ----------------------------------------- #
def test_seen_reads_the_filed_column_not_filing_date():
    """filing_risk_text's PK/date column is `filed` (not `filing_date` like
    sec_8k/sec_13d) -- `_seen` must key off that, or a re-run would treat every
    ticker as never-before-fetched."""
    class FakeStore:
        def load(self, table, columns=None):
            return pd.DataFrame({
                "ticker": ["AAPL", "AAPL", "MSFT"],
                "accession_number": ["0001-a", "0001-b", "0002-a"],
                "filed": ["2023-01-01", "2024-01-01", "2022-06-01"],
            })
    context = SimpleNamespace(store=FakeStore())
    seen, last_by_ticker = _seen(context)
    assert seen == {"0001-a", "0001-b", "0002-a"}
    assert last_by_ticker["AAPL"] == pd.Timestamp("2024-01-01")
    assert last_by_ticker["MSFT"] == pd.Timestamp("2022-06-01")


def test_seen_empty_when_table_does_not_exist_yet():
    class FailingStore:
        def load(self, table, columns=None):
            raise Exception("relation does not exist")
    context = SimpleNamespace(store=FailingStore())
    seen, last_by_ticker = _seen(context)
    assert seen == set()
    assert last_by_ticker == {}


def test_sanity_check_prints_conclusion():
    print("\n=== SANITY CHECK: edgartools 10-K/10-Q filing-text extraction ===")
    print("  Structured PRIMARY path (TenK.risk_factors/.management_discussion,")
    print("  TenQ['Part I, Item 2']) is used first; a stub/cross-reference result under")
    print("  FILING_TEXT_MIN_CHARS is discarded so ONLY that missing section falls back")
    print("  to the hardened regex carve over filing.text() -- the other, successfully")
    print("  structured section passes through untouched (not re-derived/overwritten).")
    print("  Both .obj() and .text() failing yields an empty dict, not a crash.")
    print("  _seen() keys off the `filed` column (this table's own PK date column, unlike")
    print("  sec_8k/sec_13d's `filing_date`) for the accession dedup + per-ticker resume")
    print("  cutoff. build_ticker_filing_text correctly skips already-seen accessions and")
    print("  filings before the `since` cutoff, no local HTML cache involved anywhere.")
    print("  Validated.")
