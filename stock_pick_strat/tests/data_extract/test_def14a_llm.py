"""
Tests for the DEF 14A LLM extraction pipeline.

test_def14a_schema_roundtrip      — Def14AExtract serializes / deserializes cleanly
test_def14a_sections_extraction   — prepare_def14a_sections() isolates the right blocks
test_llm_extractor_mock           — LLMExtractor.extract() with a mocked OpenAI call
test_llm_extractor_real_apple     — Live EDGAR + OpenAI extraction of Apple's DEF 14A
                                    (skips when OPENAI_API_KEY is not set)
test_fetch_def14a_llm_to_postgres — fetch_def14a_llm() drives the LLM against the
                                    Def14AExtract schema and upserts the row into
                                    the def14a_llm Postgres table (network + LLM
                                    mocked; skips when the DB is unreachable)
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
    ExecutiveOfficer,
    GovernanceProfile,
    ShareOwnership,
)
from src.data_extract.utils.structure.fetch_def14a_llm import _flatten, prepare_def14a_sections

# --------------------------------------------------------------------------- #
# Synthetic DEF 14A fixture                                                    #
# --------------------------------------------------------------------------- #
_SYNTHETIC = """
ACME CORPORATION
2024 PROXY STATEMENT

PROPOSAL 1 — ELECTION OF DIRECTORS

Alice Johnson, 58, has served as our President and Chief Executive Officer since 2019.
Ms. Johnson co-founded the Company in 2002 and serves on the Executive Committee.

Robert Williams, 64, has served as an independent director since 2010 and chairs the
Audit Committee. Mr. Williams also serves on the Compensation Committee.

Mary Chen, 52, has been an independent director since 2015. Ms. Chen serves on the
Audit Committee and the Nominating and Governance Committee.

EXECUTIVE COMPENSATION

SUMMARY COMPENSATION TABLE

Name and Principal Position   Year   Salary      Bonus      Stock Awards   Total
Alice Johnson, CEO            2023   $850,000    $500,000   $3,200,000     $4,900,000
James Thompson, CFO           2023   $620,000    $280,000   $1,800,000     $2,900,000

SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT

Name                          Shares Beneficially Owned    Percent of Class
Alice Johnson (Director/CEO)        5,250,000                   8.2%
Robert Williams (Director)             45,000                   0.1%
Mary Chen (Director)                   32,000                   0.1%
"""


def _make_expected() -> Def14AExtract:
    return Def14AExtract(
        company_name="ACME Corporation",
        fiscal_year=2023,
        ceo_name="Alice Johnson", ceo_age=58, ceo_title="President and CEO",
        ceo_since_year=2019, ceo_is_founder=True, ceo_is_board_chair=True,
        directors=[
            DirectorInfo(name="Alice Johnson", age=58, tenure_years=5.0, is_independent=False,
                         is_board_chair=True, gender="female", other_public_company_boards=1),
            DirectorInfo(name="Robert Williams", age=64, tenure_years=14.0, is_independent=True,
                         gender="male", other_public_company_boards=2,
                         audit_committee_financial_expert=True,
                         committees=["Audit", "Compensation"]),
            DirectorInfo(name="Mary Chen", age=52, tenure_years=9.0, is_independent=True,
                         gender="female", other_public_company_boards=0,
                         committees=["Audit", "Nominating and Governance"]),
        ],
        executive_officers=[
            ExecutiveOfficer(name="Alice Johnson", age=58, title="President and CEO"),
            ExecutiveOfficer(name="James Thompson", title="CFO"),
        ],
        compensation=[
            ExecutiveCompensation(
                name="Alice Johnson", title="CEO", fiscal_year=2023,
                salary_usd=850_000, bonus_usd=500_000, stock_awards_usd=3_200_000,
                option_awards_usd=250_000, non_equity_incentive_usd=100_000,
                all_other_comp_usd=50_000, total_compensation_usd=4_900_000,
            ),
            ExecutiveCompensation(
                name="James Thompson", title="CFO", fiscal_year=2023,
                salary_usd=620_000, bonus_usd=280_000,
                stock_awards_usd=1_800_000, total_compensation_usd=2_900_000,
            ),
        ],
        share_ownership=[
            ShareOwnership(name="Alice Johnson", is_director=True, is_officer=True,
                           shares_owned=5_250_000, percent_owned=0.082),
            ShareOwnership(name="Robert Williams", is_director=True, is_officer=False,
                           shares_owned=45_000, percent_owned=0.001),
            ShareOwnership(name="Mary Chen", is_director=True, is_officer=False,
                           shares_owned=32_000, percent_owned=0.001),
            ShareOwnership(name="The Vanguard Group", is_five_percent_owner=True,
                           shares_owned=6_000_000, percent_owned=0.094),
        ],
        governance=GovernanceProfile(
            board_size=3, independent_chair=False, ceo_is_board_chair=True,
            classified_board=False, dual_class_shares=False, poison_pill=False,
            say_on_pay_support_pct=0.95, ceo_pay_ratio=250.0,
            median_employee_pay_usd=60_000.0, auditor_name="Ernst & Young LLP",
            auditor_fees_usd=5_000_000.0,
        ),
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_def14a_schema_roundtrip():
    """Def14AExtract serializes to JSON and roundtrips without loss."""
    original = _make_expected()
    roundtrip = Def14AExtract.model_validate_json(original.model_dump_json())

    assert roundtrip.company_name == original.company_name
    assert roundtrip.fiscal_year == original.fiscal_year
    assert len(roundtrip.directors) == 3
    assert roundtrip.directors[1].is_independent is True
    assert roundtrip.directors[1].committees == ["Audit", "Compensation"]
    assert roundtrip.compensation[0].salary_usd == 850_000.0
    assert roundtrip.share_ownership[0].percent_owned == pytest.approx(0.082)
    # expanded fields survive the roundtrip
    assert roundtrip.ceo_age == 58 and roundtrip.ceo_is_board_chair is True
    assert roundtrip.governance.ceo_pay_ratio == 250.0
    assert roundtrip.governance.auditor_name == "Ernst & Young LLP"
    assert roundtrip.directors[1].other_public_company_boards == 2

    print("\n=== SANITY CHECK: Def14AExtract schema roundtrip ===")
    print(f"  company={roundtrip.company_name}  fiscal_year={roundtrip.fiscal_year}")
    print(f"  directors={[d.name for d in roundtrip.directors]}")
    print(f"  NEO compensation: {roundtrip.compensation[0].name} "
          f"salary=${roundtrip.compensation[0].salary_usd:,.0f}  "
          f"total=${roundtrip.compensation[0].total_compensation_usd:,.0f}")
    print(f"  ownership rows: {len(roundtrip.share_ownership)}.  Validated.")


def test_def14a_sections_extraction():
    """prepare_def14a_sections() returns text containing all three key sections."""
    focused = prepare_def14a_sections(_SYNTHETIC)

    assert "DIRECTOR NOMINEES" in focused
    assert "EXECUTIVE COMPENSATION" in focused
    assert "SECURITY OWNERSHIP" in focused
    # Each section should carry its content
    assert "Alice Johnson" in focused
    assert "Summary Compensation" in focused or "850,000" in focused
    assert "5,250,000" in focused

    print("\n=== SANITY CHECK: prepare_def14a_sections ===")
    print(f"  focused text length: {len(focused):,} chars "
          f"(original: {len(_SYNTHETIC):,} chars)")
    print("  All three sections present (DIRECTOR NOMINEES, EXECUTIVE COMPENSATION, "
          "SECURITY OWNERSHIP). Validated.")


def test_llm_extractor_mock():
    """LLMExtractor.extract() with a mocked OpenAI Responses API call."""
    from src.data_extract.utils.common.llm_extractor import LLMExtractor

    expected = _make_expected()
    mock_response = MagicMock()
    mock_response.output_parsed = expected

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("src.data_extract.utils.common.llm_extractor.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.responses.parse.return_value = mock_response
            mock_cls.return_value = mock_client

            extractor = LLMExtractor(model="gpt-4o-mini")   # defaults: temperature=0, cache=True
            result = extractor.extract(Def14AExtract, _SYNTHETIC)

    assert isinstance(result, Def14AExtract)
    assert result.company_name == "ACME Corporation"
    assert len(result.directors) == 3
    assert result.compensation[0].salary_usd == 850_000.0

    call_kw = mock_client.responses.parse.call_args.kwargs
    assert call_kw["model"] == "gpt-4o-mini"
    assert call_kw["text_format"] is Def14AExtract
    assert call_kw["instructions"]
    # deterministic + cacheable
    assert call_kw["temperature"] == 0.0
    assert call_kw["prompt_cache_key"] == "gpt-4o-mini:Def14AExtract"

    print("\n=== SANITY CHECK: LLMExtractor mock ===")
    print(f"  extract() -> {result.company_name}: {len(result.directors)} directors, "
          f"CEO salary ${result.compensation[0].salary_usd:,.0f}.")
    print(f"  responses.parse() called with temperature={call_kw['temperature']} "
          f"and prompt_cache_key={call_kw['prompt_cache_key']!r}. Validated.")


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")),
    reason="OPENAI_API_KEY not set — skipping live Apple DEF 14A extraction",
)
def test_llm_extractor_real_apple():
    """Live integration: fetch Apple's most recent DEF 14A from EDGAR, extract with OpenAI."""
    from src.data_extract.utils.common.edgar_extract import html_to_text
    from src.data_extract.utils.common.edgar_fillings import list_filings
    from src.data_extract.utils.common.llm_extractor import LLMExtractor
    from src.data_extract.utils.common.sec_utils import sec_get

    AAPL_CIK = "0000320193"
    filings = list_filings(AAPL_CIK, ["DEF 14A"], years=2, company_name="Apple Inc.")
    if filings.empty:
        pytest.skip("No DEF 14A filings found for AAPL via EDGAR")

    latest = filings.iloc[-1]
    raw_text = html_to_text(sec_get(latest["doc_url"]).text)
    assert len(raw_text) > 5_000, "DEF 14A text suspiciously short — fetch likely failed"

    focused = prepare_def14a_sections(raw_text)

    extractor = LLMExtractor(model="gpt-4o-mini")
    result = extractor.extract(Def14AExtract, focused)

    assert isinstance(result, Def14AExtract)
    assert len(result.directors) >= 5, (
        f"Apple should have ≥5 directors, got {len(result.directors)}: "
        f"{[d.name for d in result.directors]}"
    )
    assert len(result.compensation) >= 2, (
        f"Expected ≥2 NEOs in Apple's comp table, got {len(result.compensation)}"
    )
    ceo = next(
        (c for c in result.compensation
         if "chief executive" in c.title.lower() or "ceo" in c.title.lower()),
        result.compensation[0] if result.compensation else None,
    )

    print("\n=== SANITY CHECK: Real DEF 14A extraction (Apple) ===")
    print(f"  Filing date: {latest['filing_date'].date()}")
    print(f"  Focused text: {len(focused):,} chars (original: {len(raw_text):,} chars)")
    print(f"  Company: {result.company_name}  Fiscal year: {result.fiscal_year}")
    print(f"  Directors ({len(result.directors)}): "
          f"{[d.name for d in result.directors[:4]]}...")
    ind = [d for d in result.directors if d.is_independent]
    print(f"  Independent: {len(ind)}/{len(result.directors)}  "
          f"Avg age: {round(sum(d.age for d in result.directors if d.age) / max(len([d for d in result.directors if d.age]), 1), 1)}")
    print(f"  Officers ({len(result.executive_officers)}): "
          f"{[o.name for o in result.executive_officers[:3]]}...")
    if ceo:
        sal = f"${ceo.salary_usd:,.0f}" if ceo.salary_usd else "n/a"
        tot = f"${ceo.total_compensation_usd:,.0f}" if ceo.total_compensation_usd else "n/a"
        print(f"  CEO: {ceo.name} ({ceo.title}) — salary {sal}  total {tot}")
    if result.share_ownership:
        print(f"  Ownership ({len(result.share_ownership)}): "
              f"{[s.name for s in result.share_ownership[:4]]}...")
    print("  Validated.")


# --------------------------------------------------------------------------- #
# fetch_def14a_llm(): schema-constrained LLM -> Postgres def14a_llm table      #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.getenv("DATABASE_URL"),
                    reason="DATABASE_URL not set — needs the Postgres DB")
def test_fetch_def14a_llm_to_postgres(monkeypatch):
    """End-to-end (network + LLM mocked): confirms the fetcher (1) drives the LLM
    against the Def14AExtract Pydantic schema, and (2) UPSERTS the flattened row
    into the `def14a_llm` Postgres table — not a parquet file."""
    from sqlalchemy import text
    from src.context import get_config_context
    from src.data_extract.utils.structure import fetch_def14a_llm as mod

    try:
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        ctx.store.exists("def14a_llm")                      # force a DB round-trip
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

            def extract(self, schema, text_):
                captured["schema"] = schema                 # record what the LLM must respect
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

        # (1) the LLM was constrained to the passed Pydantic schema
        assert captured["schema"] is Def14AExtract, captured

        # (2) the flattened output was persisted to the Postgres table
        back = ctx.store.load("def14a_llm")
        row = back[back["ticker"] == TICKER]
        assert len(row) == 1, "row not found in def14a_llm table"
        r = row.iloc[0]
        assert r["accession_number"] == ACC
        assert int(r["n_directors"]) == 3
        assert float(r["ceo_salary"]) == 850_000.0
        assert int(r["ceo_age"]) == 58                        # the user's explicit ask
        assert float(r["ceo_pay_ratio"]) == 250.0             # expanded governance signal
        assert float(r["pct_female_directors"]) == pytest.approx(2 / 3, abs=1e-3)
        assert json.loads(r["def14a_json"])["company_name"] == "ACME Corporation"

        print("\n=== SANITY CHECK: DEF 14A LLM -> Postgres ===")
        print(f"  LLM constrained to schema: {captured['schema'].__name__}")
        print(f"  Upserted into DB table 'def14a_llm': ticker={r['ticker']} "
              f"acc={r['accession_number']} n_directors={int(r['n_directors'])} "
              f"ceo_age={int(r['ceo_age'])} ceo_salary=${float(r['ceo_salary']):,.0f} "
              f"pay_ratio={float(r['ceo_pay_ratio']):.0f}:1")
        print(f"  Full schema JSON stored in def14a_json (company="
              f"{json.loads(r['def14a_json'])['company_name']}).")
        print("  Persisted to Postgres, NOT parquet. Validated.")
    finally:
        _cleanup()


# --------------------------------------------------------------------------- #
# Year-incremental cutoff + per-ticker save                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.getenv("DATABASE_URL"),
                    reason="DATABASE_URL not set — needs the Postgres DB")
def test_fetch_def14a_llm_incremental_by_year(monkeypatch):
    """A ticker with rows already in SQL only sends filings AFTER its latest
    stored `as_of` to the LLM (year-incremental), and the new row is upserted
    without disturbing the already-saved one."""
    from sqlalchemy import text
    from src.context import get_config_context
    from src.data_extract.utils.structure import fetch_def14a_llm as mod

    try:
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        ctx.store.exists("def14a_llm")
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"DB not reachable: {e}")

    TICKER, OLD_ACC, NEW_ACC = "ZZINC", "1111111111-11-111111", "2222222222-22-222222"

    def _cleanup():
        if ctx.store.exists("def14a_llm"):
            with ctx.store.engine.begin() as c:
                c.execute(text('DELETE FROM def14a_llm WHERE ticker = :t'), {"t": TICKER})

    _cleanup()
    try:
        # seed one already-processed 2022 filing for the ticker
        seed = _flatten_row(TICKER, OLD_ACC, pd.Timestamp("2022-04-01"))
        mod._save_ticker_rows(ctx, [seed])

        captured: dict = {}

        class _FakeExtractor:
            def __init__(self, **kwargs):
                pass

            def extract(self, schema, text_):
                return _make_expected()

        def _fake_list_filings(cik, forms, years, company="", since=None):
            captured["since"] = since                        # record the cutoff
            # EDGAR would return only filings strictly after `since`
            return pd.DataFrame([{
                "accession_number": NEW_ACC, "doc_url": "http://x/new.htm",
                "filing_date": pd.Timestamp("2024-04-01"),
                "period_of_report": "2023-12-31", "form": "DEF 14A",
            }])

        class _Resp:
            text = "<html>proxy</html>"

        monkeypatch.setattr(mod, "LLMExtractor", _FakeExtractor)
        monkeypatch.setattr(mod, "list_filings", _fake_list_filings)
        monkeypatch.setattr(mod, "sec_get", lambda url, **k: _Resp())
        monkeypatch.setattr(mod, "load_cik_mapping", lambda _ctx: pd.DataFrame(
            {"ticker": [TICKER], "cik": ["0000000001"], "company_name": ["Z"]}))
        monkeypatch.setattr(mod, "_is_up_to_date", lambda _ctx, _n: False)

        mod.fetch_def14a_llm(ctx, tickers=[TICKER])

        # (1) the LLM cutoff was the latest as_of already in SQL (2022-04-01)
        assert captured["since"] is not None
        assert pd.Timestamp(captured["since"]).normalize() == pd.Timestamp("2022-04-01")

        # (2) both the old (untouched) and the new row are in the table
        back = ctx.store.load("def14a_llm")
        rows = back[back["ticker"] == TICKER]
        accs = set(rows["accession_number"])
        assert accs == {OLD_ACC, NEW_ACC}, accs

        print("\n=== SANITY CHECK: DEF 14A year-incremental ===")
        print(f"  ticker had 2022 filing stored -> list_filings called with "
              f"since={pd.Timestamp(captured['since']).date()} (only newer years hit the LLM)")
        print(f"  after run, table holds both accessions: {sorted(accs)}. Validated.")
    finally:
        _cleanup()


def _flatten_row(ticker: str, acc: str, as_of: "pd.Timestamp") -> dict:
    """Minimal def14a_llm row for seeding the incremental test."""
    return {
        "ticker": ticker, "as_of": as_of, "period": as_of,
        "accession_number": acc, "company_name": "Z", "fiscal_year_extract": as_of.year - 1,
        "n_directors": 3, "avg_director_age": 60.0, "avg_board_tenure": 8.0,
        "pct_independent_directors": 0.66, "n_officers": 2,
        "ceo_name_proxy": "Old CEO", "ceo_salary": 800_000.0, "ceo_bonus": None,
        "ceo_stock_awards": None, "ceo_option_awards": None, "ceo_total_comp": 1_000_000.0,
        "def14a_json": "{}",
    }


# --------------------------------------------------------------------------- #
# _flatten surfaces the expanded governance / comp / ownership signals          #
# --------------------------------------------------------------------------- #
def test_flatten_surfaces_all_signals():
    """_flatten emits ceo_age and every new aggregate/governance column with the
    right values (the user's explicit ask: ensure ceo_age is included)."""
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"),
                        "period_of_report": "2023-12-31",
                        "accession_number": "0000-24-000001"})
    row = _flatten("ACME", filing, _make_expected())

    # CEO — including age (top-level) and the duality / founder flags
    assert row["ceo_age"] == 58
    assert row["ceo_is_board_chair"] == 1.0
    assert row["ceo_is_founder"] == 1.0
    assert row["ceo_since_year"] == 2019
    assert row["ceo_non_equity_incentive"] == 100_000
    assert row["ceo_equity_pay_pct"] == pytest.approx((3_200_000 + 250_000) / 4_900_000, abs=1e-3)
    # board composition
    assert row["pct_female_directors"] == pytest.approx(2 / 3, abs=1e-3)     # Alice, Mary
    assert row["avg_other_public_boards"] == pytest.approx((1 + 2 + 0) / 3, abs=1e-3)
    assert row["n_financial_experts"] == 1
    # NEO aggregate + ownership
    assert row["n_neos"] == 2
    assert row["total_neo_comp"] == 4_900_000 + 2_900_000
    assert row["ceo_ownership_pct"] == pytest.approx(0.082)
    assert row["insider_ownership_pct"] == pytest.approx(0.082 + 0.001 + 0.001, abs=1e-4)
    assert row["n_five_percent_holders"] == 1                                # Vanguard
    # governance provisions (booleans -> numeric flags)
    assert row["classified_board"] == 0.0 and row["dual_class_shares"] == 0.0
    assert row["say_on_pay_support_pct"] == pytest.approx(0.95)
    assert row["ceo_pay_ratio"] == 250.0
    assert row["median_employee_pay"] == 60_000.0
    assert row["auditor_name"] == "Ernst & Young LLP"
    assert row["auditor_fees"] == 5_000_000.0

    print("\n=== SANITY CHECK: _flatten expanded signals ===")
    print(f"  ceo_age={row['ceo_age']} duality={row['ceo_is_board_chair']} "
          f"pay_ratio={row['ceo_pay_ratio']:.0f}:1 equity_pay={row['ceo_equity_pay_pct']:.2f}")
    print(f"  board: %female={row['pct_female_directors']:.2f} avg_other_boards="
          f"{row['avg_other_public_boards']:.2f} fin_experts={row['n_financial_experts']}")
    print(f"  ownership: insider={row['insider_ownership_pct']:.3f} ceo={row['ceo_ownership_pct']:.3f} "
          f"5%+ holders={row['n_five_percent_holders']}; auditor={row['auditor_name']}. Validated.")


def test_flatten_ceo_age_fallback_to_directors():
    """When ceo_age isn't given top-level, _flatten recovers it from the CEO's
    entry in the directors list (the CEO is a director nominee)."""
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"),
                        "period_of_report": None, "accession_number": "acc-x"})
    extract = Def14AExtract(
        company_name="FallbackCo", ceo_name="Bob Stone",   # ceo_age deliberately omitted
        directors=[DirectorInfo(name="Bob Stone", age=61, is_independent=False),
                   DirectorInfo(name="Jane Roe", age=55, is_independent=True)],
    )
    row = _flatten("FBK", filing, extract)
    assert row["ceo_age"] == 61          # recovered from the directors list
    print("\n=== SANITY CHECK: ceo_age fallback ===")
    print(f"  ceo_age omitted top-level -> recovered {row['ceo_age']} from directors. Validated.")
