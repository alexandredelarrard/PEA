"""
Tests for the DEF 14A LLM extraction pipeline (trimmed-schema + targeted-slice design).

test_def14a_schema_roundtrip        — Def14AExtract serializes / deserializes cleanly
test_compensation_anchor_skips_noise— the SCT anchor skips CD&A prose AND "realized/Total
                                      Cash" pay tables, landing on the real Summary Comp Table
test_sayonpay_anchor_matches_approval — say-on-pay anchor catches "NN% approval"/"NN% of votes"
test_payratio_and_ownership_anchors — pay-ratio & beneficial-ownership content anchors match
test_def14a_sections_extraction     — prepare_def14a_sections() isolates the right blocks
test_llm_extractor_mock             — LLMExtractor.extract() with a mocked OpenAI call
test_flatten_surfaces_all_signals   — _flatten emits every governance / comp / ownership column
test_flatten_ceo_age_fallback       — ceo_age recovered from the directors list
test_llm_extractor_real_apple       — live EDGAR + OpenAI (skips without OPENAI_API_KEY)
test_fetch_def14a_llm_to_postgres   — fetcher drives the schema-constrained LLM and upserts to DB
test_fetch_def14a_llm_incremental   — per-ticker year-incremental cutoff (skips without DB)
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_extract.utils.structure.def14a_schema import (
    Def14AExtract,
    DirectorInfo,
    ExecutiveCompensation,
    GovernanceProfile,
)
from src.data_extract.utils.structure.fetch_def14a_llm import (
    _COMPENSATION_CONTENT_RE,
    _DEF14A_PROMPT,
    _DIRECTOR_ROW_RE,
    _OWNERSHIP_CONTENT_RE,
    _OWNERSHIP_ROW_RE,
    _PAYRATIO_CONTENT_RE,
    _SAYONPAY_CONTENT_RE,
    _densest_window,
    _flatten,
    prepare_def14a_sections,
)

# --------------------------------------------------------------------------- #
# Synthetic DEF 14A fixture — exercises the tricky section-anchoring cases      #
# (CD&A prose before the real SCT; a "realized / Total Cash" pay table before   #
#  it; a directors-and-officers-as-a-group ownership row; a "NN% approval"      #
#  say-on-pay result; a pay-ratio sentence). Padded > 5k chars so the prose,    #
#  anchor-only CORPORATE GOVERNANCE section clears the TOC-skip floor.          #
# --------------------------------------------------------------------------- #
_FILLER = ("The Board is committed to strong governance and long-term value creation. " * 12)

_SYNTHETIC = f"""
ACME CORPORATION — 2024 PROXY STATEMENT
Table of Contents: Election of Directors; Executive Compensation; Summary Compensation
Table; Security Ownership; CEO Pay Ratio; Say on Pay; Auditor Fees. {_FILLER}

PROPOSAL 1 — ELECTION OF DIRECTORS
{_FILLER}
Alice Johnson, 58, has served as our President and Chief Executive Officer since 2019.
Ms. Johnson co-founded the Company in 2002 and serves on the Executive Committee.
Robert Williams, 64, has served as an independent director since 2010 and chairs the Audit
Committee. Mary Chen, 52, has been an independent director since 2015.
{_FILLER}

COMPENSATION DISCUSSION AND ANALYSIS
Our program balances base salary, annual cash incentives, and long-term equity awards to
align pay with performance; Total direct compensation is targeted near the median. {_FILLER}

REALIZED PAY / TOTAL CASH SUMMARY (supplemental, non-GAAP)
Name Year Salary Bonus Stock-Based Awards Total Cash Vesting of Previous Awards
Alice Johnson 2023 850,000 500,000 3,200,000 1,350,000 2,000,000
{_FILLER}

SUMMARY COMPENSATION TABLE
Name and Principal Position Year Salary Bonus Stock Awards Option Awards Non-Equity Incentive Plan Compensation All Other Compensation Total
Alice Johnson, Chief Executive Officer 2023 850,000 500,000 3,200,000 250,000 100,000 50,000 4,900,000
James Thompson, Chief Financial Officer 2023 620,000 280,000 1,800,000 0 0 0 2,900,000
{_FILLER}

CORPORATE GOVERNANCE
The Board has a majority-voting standard for directors and a Lead Independent Director; the
Chair and CEO roles are combined. There is no classified board and no dual-class stock. {_FILLER}

SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT
Name Shares Beneficially Owned Percent of Class
Alice Johnson 5,250,000 8.2%
The Vanguard Group 6,000,000 9.4%
All directors and executive officers as a group (3 persons) 5,327,000 8.4%
{_FILLER}

CEO PAY RATIO
The annual total compensation of our median employee was $60,000; the ratio of the annual
total compensation of our CEO to that of the median employee was 250 to 1. {_FILLER}

SAY ON PAY
At the 2023 Annual Meeting, our Say on Pay proposal received 91% approval from shareholders. {_FILLER}

AUDITOR FEES
Audit Fees billed by Ernst & Young LLP were $5,000,000 for the year. {_FILLER}
"""


def _make_expected() -> Def14AExtract:
    """A fully-populated extract matching the current (trimmed) schema."""
    return Def14AExtract(
        company_name="ACME Corporation",
        fiscal_year=2023,
        ceo_name="Alice Johnson", ceo_age=58,
        ceo_since_year=2019, ceo_is_founder=True, ceo_is_board_chair=True,
        directors=[
            DirectorInfo(name="Alice Johnson", age=58, tenure_years=5.0, is_independent=False,
                         gender="female", other_public_company_boards=1),
            DirectorInfo(name="Robert Williams", age=64, tenure_years=14.0, is_independent=True,
                         gender="male", other_public_company_boards=2),
            DirectorInfo(name="Mary Chen", age=52, tenure_years=9.0, is_independent=True,
                         gender="female", other_public_company_boards=0),
        ],
        compensation=[
            ExecutiveCompensation(
                name="Alice Johnson", title="Chief Executive Officer", fiscal_year=2023,
                salary_usd=850_000, bonus_usd=500_000, stock_awards_usd=3_200_000,
                option_awards_usd=250_000, non_equity_incentive_usd=100_000,
                all_other_comp_usd=50_000, total_compensation_usd=4_900_000,
            ),
            ExecutiveCompensation(
                name="James Thompson", title="Chief Financial Officer", fiscal_year=2023,
                salary_usd=620_000, bonus_usd=280_000,
                stock_awards_usd=1_800_000, total_compensation_usd=2_900_000,
            ),
        ],
        governance=GovernanceProfile(
            board_size=3, n_independent_directors=2, n_women_directors=2,
            independent_chair=False, ceo_is_board_chair=True, lead_independent_director=True,
            classified_board=False, dual_class_shares=False, poison_pill=False,
            majority_voting_for_directors=True,
            say_on_pay_support_pct=0.91, ceo_pay_ratio=250.0,
            median_employee_pay_usd=60_000.0, auditor_fees_usd=5_000_000.0,
            insider_ownership_pct=0.084, ceo_ownership_pct=0.082, n_five_percent_holders=1,
        ),
    )


# --------------------------------------------------------------------------- #
# Schema                                                                        #
# --------------------------------------------------------------------------- #
def test_def14a_schema_roundtrip():
    """Def14AExtract serializes to JSON and roundtrips without loss."""
    original = _make_expected()
    roundtrip = Def14AExtract.model_validate_json(original.model_dump_json())

    assert roundtrip.company_name == original.company_name
    assert len(roundtrip.directors) == 3
    assert roundtrip.directors[1].is_independent is True
    assert roundtrip.compensation[0].salary_usd == 850_000.0
    assert roundtrip.ceo_age == 58 and roundtrip.ceo_is_board_chair is True
    assert roundtrip.governance.ceo_pay_ratio == 250.0
    assert roundtrip.governance.insider_ownership_pct == pytest.approx(0.084)
    assert roundtrip.governance.n_five_percent_holders == 1

    print("\n=== SANITY CHECK: Def14AExtract schema roundtrip ===")
    print(f"  company={roundtrip.company_name}  directors={[d.name for d in roundtrip.directors]}")
    print(f"  CEO {roundtrip.compensation[0].name}: salary=${roundtrip.compensation[0].salary_usd:,.0f} "
          f"total=${roundtrip.compensation[0].total_compensation_usd:,.0f}; "
          f"insider_own={roundtrip.governance.insider_ownership_pct:.3f}. Validated.")


# --------------------------------------------------------------------------- #
# Section-anchoring regexes (the fixes made this session)                       #
# --------------------------------------------------------------------------- #
def test_compensation_anchor_skips_noise():
    """The SCT anchor must skip (a) CD&A prose ('salary, ... equity awards ... Total')
    and (b) the 'realized / Total Cash' pay table, landing on the real Summary
    Compensation Table (whose total column is a plain 'Total')."""
    m = _COMPENSATION_CONTENT_RE.search(_SYNTHETIC)
    assert m is not None, "SCT anchor found nothing"

    cda_pos = _SYNTHETIC.index("base salary, annual cash incentives")
    realized_pos = _SYNTHETIC.index("Total Cash Vesting")
    sct_pos = _SYNTHETIC.index("Name and Principal Position Year Salary Bonus Stock Awards")

    # the match lands at the real SCT header, after both decoys
    assert m.start() > realized_pos > cda_pos
    assert abs(m.start() - sct_pos) < 60, (m.start(), sct_pos)
    window = _SYNTHETIC[m.start(): m.start() + 200]
    assert "Option Awards" in window and "Total Cash" not in window

    print("\n=== SANITY CHECK: SCT anchor skips CD&A + realized-pay ===")
    print(f"  CD&A @{cda_pos}, realized/Total-Cash @{realized_pos}, real SCT @{sct_pos}; "
          f"anchor landed @{m.start()} (on the SCT header). Validated.")


def test_sayonpay_anchor_matches_approval():
    """Say-on-pay anchor catches 'NN% approval' (not just 'NN% of votes')."""
    m = _SAYONPAY_CONTENT_RE.search(_SYNTHETIC)
    assert m is not None and "91%" in m.group(0)
    # a bare qualified percent with no say-on-pay context nearby must NOT match
    assert _SAYONPAY_CONTENT_RE.search("Turnout was 91% of shares outstanding.") is None

    print("\n=== SANITY CHECK: say-on-pay anchor ===")
    print(f"  matched {m.group(0)[:60]!r} (91% approval); context-free percent ignored. Validated.")


def test_payratio_and_ownership_anchors():
    """Pay-ratio anchors on the SEC 'median employee / ratio of the … compensation'
    sentence; ownership anchors on 'directors and … officers as a group'."""
    pr = _PAYRATIO_CONTENT_RE.search(_SYNTHETIC)
    ow = _OWNERSHIP_CONTENT_RE.search(_SYNTHETIC)
    assert pr is not None and ow is not None
    assert "median employee" in _SYNTHETIC[pr.start(): pr.start() + 40].lower() \
        or "ratio of the" in _SYNTHETIC[pr.start(): pr.start() + 40].lower()
    # the ownership anchor lands in the beneficial-ownership block; the "as a group"
    # insider row sits a few lines below it (well inside the slice window)
    ow_window = _SYNTHETIC[ow.start(): ow.start() + 400].lower()
    assert "beneficial" in ow_window and "as a group" in ow_window

    print("\n=== SANITY CHECK: pay-ratio & ownership anchors ===")
    print(f"  pay-ratio @{pr.start()}, ownership 'as a group' @{ow.start()}. Validated.")


def test_def14a_sections_extraction():
    """prepare_def14a_sections() returns text containing every key section + its data."""
    focused = prepare_def14a_sections(_SYNTHETIC)
    for label in ("DIRECTOR NOMINEES", "EXECUTIVE COMPENSATION", "SECURITY OWNERSHIP",
                  "PAY RATIO & MEDIAN PAY", "SAY ON PAY", "AUDITOR FEES"):
        assert f"=== {label} ===" in focused, f"missing section {label}"
    # target data present
    assert "4,900,000" in focused                          # CEO SCT total (real table)
    assert "as a group" in focused and "8.4%" in focused   # insider ownership
    assert "250 to 1" in focused                           # pay ratio
    assert "91% approval" in focused                       # say-on-pay
    assert "$5,000,000" in focused                         # auditor fees

    print("\n=== SANITY CHECK: prepare_def14a_sections ===")
    print(f"  focused={len(focused):,} chars (orig {len(_SYNTHETIC):,}); all sections + "
          f"CEO total / ownership / pay-ratio / say-on-pay / auditor present. Validated.")


def _section_body(focused: str, label: str) -> str:
    import re
    parts = re.split(r"\n\n=== (.+?) ===\n", focused)
    for i in range(1, len(parts), 2):
        if parts[i] == label:
            return parts[i + 1]
    return ""


def test_densest_window_lands_on_table_not_prose():
    """_densest_window ignores isolated prose row-tokens and anchors on the dense
    cluster (the table) — the fix for director bios in matrix/table layouts."""
    prose = "The company was founded in 1998. " * 30       # noise, no row tokens
    lone = "An independent director since 2005 chairs the audit committee. "  # 1 lone token
    table = ("Name Age Director Since\n"
             "Alice Johnson, 58 2019\nRobert Williams, 64 2010\n"
             "Mary Chen, 52 2015\nDavid Park, 47 2020\n")     # dense cluster of tokens
    text = prose + lone + prose + table + prose
    pos = _densest_window(text, _DIRECTOR_ROW_RE, 4000)
    assert pos != -1
    window = text[pos: pos + 4000]
    # the window must contain the table (>=4 director rows), not the lone prose token
    assert len(_DIRECTOR_ROW_RE.findall(window)) >= 4
    assert "Alice Johnson" in window and "David Park" in window

    print("\n=== SANITY CHECK: densest-window anchoring ===")
    print(f"  ignored lone 'director since 2005' prose; landed on the 5-row table "
          f"({len(_DIRECTOR_ROW_RE.findall(window))} row tokens). Validated.")


def test_tabular_sections_capture_rows_in_synthetic():
    """In the synthetic proxy, the densest-window director & ownership slices carry the
    actual rows (director ages; 5%-holder + as-a-group ownership lines)."""
    focused = prepare_def14a_sections(_SYNTHETIC)
    directors = _section_body(focused, "DIRECTOR NOMINEES")
    ownership = _section_body(focused, "SECURITY OWNERSHIP")

    assert len(_DIRECTOR_ROW_RE.findall(directors)) >= 3        # Johnson 58, Williams 64, Chen 52
    assert "Alice Johnson" in directors
    own_rows = _OWNERSHIP_ROW_RE.findall(ownership)
    assert len(own_rows) >= 3 and "as a group" in ownership.lower()

    print("\n=== SANITY CHECK: tabular slices carry rows ===")
    print(f"  DIRECTOR slice: {len(_DIRECTOR_ROW_RE.findall(directors))} director rows; "
          f"OWNERSHIP slice: {len(own_rows)} rows incl 'as a group'. Validated.")


# --------------------------------------------------------------------------- #
# LLMExtractor (mocked)                                                         #
# --------------------------------------------------------------------------- #
def test_llm_extractor_mock():
    """LLMExtractor.extract() forwards the schema, the tailored instructions and a
    stable prompt-cache key to the OpenAI Responses API."""
    from src.data_extract.utils.common.llm_extractor import LLMExtractor

    mock_response = MagicMock()
    mock_response.output_parsed = _make_expected()

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("src.data_extract.utils.common.llm_extractor.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.responses.parse.return_value = mock_response
            mock_cls.return_value = mock_client

            extractor = LLMExtractor(model="gpt-5-mini")
            result = extractor.extract(Def14AExtract, _SYNTHETIC, instructions=_DEF14A_PROMPT)

    assert isinstance(result, Def14AExtract)
    assert result.company_name == "ACME Corporation"
    call_kw = mock_client.responses.parse.call_args.kwargs
    assert call_kw["model"] == "gpt-5-mini"
    assert call_kw["text_format"] is Def14AExtract
    assert call_kw["instructions"] == _DEF14A_PROMPT           # tailored prompt forwarded
    assert call_kw["prompt_cache_key"] == "gpt-5-mini:Def14AExtract"

    print("\n=== SANITY CHECK: LLMExtractor mock ===")
    print(f"  extract() -> {result.company_name}; parse() got tailored instructions + "
          f"prompt_cache_key={call_kw['prompt_cache_key']!r}. Validated.")


# --------------------------------------------------------------------------- #
# _flatten                                                                      #
# --------------------------------------------------------------------------- #
def test_flatten_surfaces_all_signals():
    """_flatten emits ceo_age and every governance / comp / ownership column with the
    right values, sourcing board composition & ownership from the governance summary."""
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"),
                        "period_of_report": "2023-12-31",
                        "accession_number": "0000-24-000001"})
    row = _flatten("ACME", filing, _make_expected())

    # CEO
    assert row["ceo_age"] == 58
    assert row["ceo_is_board_chair"] == 1.0 and row["ceo_is_founder"] == 1.0
    assert row["ceo_since_year"] == 2019
    assert row["ceo_salary"] == 850_000 and row["ceo_non_equity_incentive"] == 100_000
    assert row["ceo_equity_pay_pct"] == pytest.approx((3_200_000 + 250_000) / 4_900_000, abs=1e-3)
    # board composition — from governance counts / board_size
    assert row["board_size"] == 3
    assert row["pct_independent_directors"] == pytest.approx(2 / 3, abs=1e-3)
    assert row["pct_female_directors"] == pytest.approx(2 / 3, abs=1e-3)
    assert row["avg_other_public_boards"] == pytest.approx((1 + 2 + 0) / 3, abs=1e-3)
    # NEO aggregate
    assert row["n_neos"] == 2
    assert row["total_neo_comp"] == 4_900_000 + 2_900_000
    # ownership — direct from governance summary
    assert row["insider_ownership_pct"] == pytest.approx(0.084)
    assert row["ceo_ownership_pct"] == pytest.approx(0.082)
    assert row["n_five_percent_holders"] == 1
    # governance provisions (bool -> numeric flag)
    assert row["classified_board"] == 0.0 and row["dual_class_shares"] == 0.0
    assert row["majority_voting"] == 1.0 and row["lead_independent_director"] == 1.0
    assert row["say_on_pay_support_pct"] == pytest.approx(0.91)
    assert row["ceo_pay_ratio"] == 250.0 and row["median_employee_pay"] == 60_000.0
    assert row["auditor_fees"] == 5_000_000.0
    # dropped columns must NOT reappear
    assert "n_financial_experts" not in row and "auditor_name" not in row and "n_officers" not in row

    print("\n=== SANITY CHECK: _flatten expanded signals ===")
    print(f"  ceo_age={row['ceo_age']} pay_ratio={row['ceo_pay_ratio']:.0f}:1 "
          f"equity_pay={row['ceo_equity_pay_pct']:.2f} say_on_pay={row['say_on_pay_support_pct']:.2f}")
    print(f"  board: size={row['board_size']} %indep={row['pct_independent_directors']:.2f} "
          f"%female={row['pct_female_directors']:.2f}")
    print(f"  ownership: insider={row['insider_ownership_pct']:.3f} ceo={row['ceo_ownership_pct']:.3f} "
          f"5%+={row['n_five_percent_holders']}; auditor_fees=${row['auditor_fees']:,.0f}. Validated.")


def test_flatten_ceo_age_fallback():
    """When ceo_age isn't given top-level, _flatten recovers it from the CEO's entry
    in the directors list (the CEO is a director nominee)."""
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"),
                        "period_of_report": None, "accession_number": "acc-x"})
    extract = Def14AExtract(
        company_name="FallbackCo", ceo_name="Bob Stone",   # ceo_age omitted
        directors=[DirectorInfo(name="Bob Stone", age=61, is_independent=False),
                   DirectorInfo(name="Jane Roe", age=55, is_independent=True)],
    )
    row = _flatten("FBK", filing, extract)
    assert row["ceo_age"] == 61
    print("\n=== SANITY CHECK: ceo_age fallback ===")
    print(f"  ceo_age omitted top-level -> recovered {row['ceo_age']} from directors. Validated.")


# --------------------------------------------------------------------------- #
# Live integration (skips without OPENAI_API_KEY)                              #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")),
    reason="OPENAI_API_KEY not set — skipping live Apple DEF 14A extraction",
)
def test_llm_extractor_real_apple():
    """Live: fetch Apple's latest DEF 14A, extract with the tailored prompt, sanity-check."""
    from src.context import get_config_context
    from src.data_extract.utils.common.edgar_extract import html_to_text
    from src.data_extract.utils.common.edgar_fillings import list_filings
    from src.data_extract.utils.common.llm_extractor import LLMExtractor
    from src.data_extract.utils.common.sec_utils import sec_get

    _, ctx = get_config_context("./configs", use_cache=True, save=False)
    model = ctx.config.data_extract.llm_model

    filings = list_filings("0000320193", ["DEF 14A"], years=2, company_name="Apple Inc.")
    if filings.empty:
        pytest.skip("No DEF 14A filings found for AAPL via EDGAR")

    latest = filings.sort_values("filing_date").iloc[-1]
    focused = prepare_def14a_sections(html_to_text(sec_get(latest["doc_url"]).text))
    result = LLMExtractor(model=model).extract(Def14AExtract, focused, instructions=_DEF14A_PROMPT)
    row = _flatten("AAPL", latest, result)

    assert len(result.directors) >= 5
    ceo = next((c for c in result.compensation
                if "chief executive" in (c.title or "").lower()), None)
    assert ceo is not None and (ceo.stock_awards_usd or 0) > 0, "CEO SCT breakdown not extracted"
    assert row["ceo_total_comp"] and row["ceo_pay_ratio"]

    print("\n=== SANITY CHECK: Real DEF 14A extraction (Apple) ===")
    print(f"  {result.company_name} FY{result.fiscal_year}; focused={len(focused):,} chars")
    print(f"  CEO {ceo.name}: salary=${(ceo.salary_usd or 0):,.0f} stock=${(ceo.stock_awards_usd or 0):,.0f} "
          f"total=${row['ceo_total_comp']:,.0f}; pay_ratio={row['ceo_pay_ratio']:.0f}:1")
    print(f"  board size={row['board_size']} %indep={row['pct_independent_directors']} "
          f"say_on_pay={row['say_on_pay_support_pct']}. Validated.")


# --------------------------------------------------------------------------- #
# fetch_def14a_llm(): schema-constrained LLM -> Postgres def14a_llm table       #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.getenv("DATABASE_URL"),
                    reason="DATABASE_URL not set — needs the Postgres DB")
def test_fetch_def14a_llm_to_postgres(monkeypatch):
    """End-to-end (network + LLM mocked): the fetcher (1) constrains the LLM to the
    Def14AExtract schema and (2) UPSERTS the flattened row into Postgres."""
    from sqlalchemy import text
    from src.context import get_config_context
    from src.data_extract.utils.structure import fetch_def14a_llm as mod

    try:
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        ctx.store.exists("def14a_llm")
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"DB not reachable: {e}")

    TICKER, ACC = "ZZTEST", "9999999999-99-999999"

    def _cleanup():
        if ctx.store.exists("def14a_llm"):
            with ctx.store.engine.begin() as c:
                c.execute(text('DELETE FROM def14a_llm WHERE ticker = :t'), {"t": TICKER})

    _cleanup()
    try:
        captured: dict = {}

        class _FakeExtractor:                               # no OPENAI key needed
            def __init__(self, **kwargs):
                pass

            def extract(self, schema, text_, instructions=None):
                captured["schema"] = schema
                captured["instructions"] = instructions
                return _make_expected()

        filings = pd.DataFrame([{
            "accession_number": ACC, "doc_url": "http://example/def14a.htm",
            "filing_date": pd.Timestamp("2024-04-01"),
            "period_of_report": "2023-12-31", "form": "DEF 14A",
        }])

        class _Resp:
            text = "<html>proxy statement</html>"

        monkeypatch.setattr(mod, "LLMExtractor", _FakeExtractor)
        monkeypatch.setattr(mod, "list_filings", lambda *a, **k: filings)
        monkeypatch.setattr(mod, "sec_get", lambda url, **k: _Resp())
        monkeypatch.setattr(mod, "load_cik_mapping", lambda _ctx: pd.DataFrame(
            {"ticker": [TICKER], "cik": ["0000000000"], "company_name": ["Z"]}))
        monkeypatch.setattr(mod, "_is_up_to_date", lambda _ctx, _n: False)

        mod.fetch_def14a_llm(ctx, tickers=[TICKER])

        assert captured["schema"] is Def14AExtract, captured
        assert captured["instructions"] == mod._DEF14A_PROMPT   # tailored prompt used

        back = ctx.store.load("def14a_llm")
        row = back[back["ticker"] == TICKER]
        assert len(row) == 1, "row not found in def14a_llm table"
        r = row.iloc[0]
        assert r["accession_number"] == ACC
        assert int(r["n_directors"]) == 3
        assert float(r["ceo_salary"]) == 850_000.0
        assert int(r["ceo_age"]) == 58
        assert float(r["ceo_pay_ratio"]) == 250.0
        assert float(r["pct_female_directors"]) == pytest.approx(2 / 3, abs=1e-3)
        assert float(r["insider_ownership_pct"]) == pytest.approx(0.084, abs=1e-3)
        assert json.loads(r["def14a_json"])["company_name"] == "ACME Corporation"

        print("\n=== SANITY CHECK: DEF 14A LLM -> Postgres ===")
        print(f"  LLM constrained to {captured['schema'].__name__} with tailored prompt.")
        print(f"  Upserted: ticker={r['ticker']} n_directors={int(r['n_directors'])} "
              f"ceo_age={int(r['ceo_age'])} ceo_salary=${float(r['ceo_salary']):,.0f} "
              f"pay_ratio={float(r['ceo_pay_ratio']):.0f}:1 insider_own={float(r['insider_ownership_pct']):.3f}")
        print("  Persisted to Postgres, NOT parquet. Validated.")
    finally:
        _cleanup()


@pytest.mark.skipif(not os.getenv("DATABASE_URL"),
                    reason="DATABASE_URL not set — needs the Postgres DB")
def test_fetch_def14a_llm_incremental(monkeypatch):
    """Gap-filling per-filing incremental: the FULL window is listed (no `since` cutoff), and only
    filings whose accession is NOT already in the table hit the LLM — so a MISSING year (a hole
    between two present years) is filled while the present ones are never re-extracted."""
    from sqlalchemy import text
    from src.context import get_config_context
    from src.data_extract.utils.structure import fetch_def14a_llm as mod

    try:
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        ctx.store.exists("def14a_llm")
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"DB not reachable: {e}")

    # 2022 + 2024 already stored; 2023 is a HOLE in the middle; a NEW 2025 also appears
    TICKER = "ZZINC"
    A22, A23, A24, A25 = (f"{y}{y}{y}{y}{y}{y}{y}{y}{y}{y}-{y%100}{y%100}-{y}11" for y in (1, 2, 3, 4))
    have_years = {"2022": A22, "2024": A24}
    gap_years = {"2023": A23, "2025": A25}                   # the two MISSING filings to fill

    def _cleanup():
        if ctx.store.exists("def14a_llm"):
            with ctx.store.engine.begin() as c:
                c.execute(text('DELETE FROM def14a_llm WHERE ticker = :t'), {"t": TICKER})

    _cleanup()
    try:
        for yr, acc in have_years.items():
            mod._save_ticker_rows(ctx, [_seed_row(TICKER, acc, pd.Timestamp(f"{yr}-04-01"))])
        captured: dict = {"since_seen": [], "extracted": []}

        class _FakeExtractor:
            def __init__(self, **kwargs):
                pass

            def extract(self, schema, text_, instructions=None):
                return _make_expected()

        def _fake_list_filings(cik, forms, years, company="", since=None):
            captured["since_seen"].append(since)            # must be None now (full window)
            rows = {**have_years, **gap_years}
            return pd.DataFrame([{
                "accession_number": acc, "doc_url": f"http://x/{yr}.htm",
                "filing_date": pd.Timestamp(f"{yr}-04-01"),
                "period_of_report": f"{int(yr)-1}-12-31", "form": "DEF 14A",
            } for yr, acc in rows.items()])

        class _Resp:
            text = "<html>proxy</html>"

        # count which accessions actually reach the LLM
        orig_process = mod._process_filing

        def _spy_process(ticker, filing, extractor):
            captured["extracted"].append(filing["accession_number"])
            return orig_process(ticker, filing, extractor)

        monkeypatch.setattr(mod, "LLMExtractor", _FakeExtractor)
        monkeypatch.setattr(mod, "list_filings", _fake_list_filings)
        monkeypatch.setattr(mod, "_process_filing", _spy_process)
        monkeypatch.setattr(mod, "sec_get", lambda url, **k: _Resp())
        monkeypatch.setattr(mod, "load_cik_mapping", lambda _ctx: pd.DataFrame(
            {"ticker": [TICKER], "cik": ["0000000001"], "company_name": ["Z"]}))
        monkeypatch.setattr(mod, "_is_up_to_date", lambda _ctx, _n: False)

        mod.fetch_def14a_llm(ctx, tickers=[TICKER])

        # full window listed (no since cutoff), and ONLY the two missing years hit the LLM
        assert captured["since_seen"] == [None]
        assert set(captured["extracted"]) == set(gap_years.values()), captured["extracted"]
        back = ctx.store.load("def14a_llm")
        accs = set(back[back["ticker"] == TICKER]["accession_number"])
        assert accs == set(have_years.values()) | set(gap_years.values()), accs

        print("\n=== SANITY CHECK: DEF 14A gap-filling incremental ===")
        print(f"  had 2022+2024; listed full window (since={captured['since_seen'][0]}); "
              f"LLM ran ONLY on the missing {sorted(gap_years)} (2023 hole + new 2025), "
              f"skipped the 2 present. Table now {len(accs)} filings. Validated.")
    finally:
        _cleanup()


def _seed_row(ticker: str, acc: str, as_of: pd.Timestamp) -> dict:
    """Minimal def14a_llm row for seeding the incremental test (current columns only)."""
    return {
        "ticker": ticker, "as_of": as_of, "period": as_of,
        "accession_number": acc, "company_name": "Z", "fiscal_year_extract": as_of.year - 1,
        "n_directors": 3, "board_size": 3, "avg_director_age": 60.0, "avg_board_tenure": 8.0,
        "pct_independent_directors": 0.66,
        "ceo_name_proxy": "Old CEO", "ceo_salary": 800_000.0, "ceo_total_comp": 1_000_000.0,
        "def14a_json": "{}",
    }
