"""Focused request, historical reach and daily domain-repair contracts."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.context import Context
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a import carve
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_payloads, prepare_def14a_sections
from src.data_extract.utils.structure.def14a.retry import SECTIONS, Section
from src.gpt_extract.transformers.gpt_getter import LLMExtractor


def test_director_pay_schema_has_no_calendar_cutoff() -> None:
    description = Def14AExtract.model_fields["director_compensation"].description or ""
    assert "2008" not in description
    print("SANITY: extraction instructions do not suppress disclosed director compensation by calendar year.")


def test_historical_named_director_pay_reaches_the_payload() -> None:
    text = "COMPENSATION OF DIRECTORS\nIn 1995 Anne Smith received aggregate director compensation of $25,000."
    payload = prepare_def14a_sections("<pre>" + text + "</pre>", text)
    assert "=== DIRECTOR COMPENSATION TABLE ===" in payload
    assert "Anne Smith" in payload and "$25,000" in payload
    print("SANITY: explicit named 1995 director compensation reaches its focused section without a modern table requirement.")


@pytest.mark.parametrize("section", SECTIONS, ids=lambda section: section.key)
def test_each_section_resolves_its_own_prompt_schema_and_budget(section: Section) -> None:
    class CaptureProvider:
        name, model, structured = "stub", "stub", True
        seen: tuple[type[BaseModel], str, str] | None = None

        def parse(self, schema: type[BaseModel], system: str, user: str) -> tuple[BaseModel, dict[str, int]]:
            self.seen = schema, system, user
            payload = {"percent_unit": "percent"} if section.key in ("governance", "ownership") else {}
            return schema.model_validate(payload), {"input_tokens": 5, "output_tokens": 1, "cached_input_tokens": 0}

    config = OmegaConf.load(Path(__file__).resolve().parents[3] / "configs/gpt.yml")
    context = cast(Context, SimpleNamespace(store=None, log=logging.getLogger("section-test"), config_dir=Path(".")))
    extractor = LLMExtractor(context, config, action=section.action)
    provider = CaptureProvider()
    extractor.extract(section.schema, "SOURCE " * 100_000, provider=provider, action=section.action)
    assert provider.seen is not None
    schema, system, user = provider.seen
    assert schema is section.schema and schema is not Def14AExtract
    assert section.key.replace("_", " ") in system
    assert section.schema.__name__ in system
    assert "{_format}" not in system and "{query}" not in user
    assert extractor.max_chars[section.action] == 130_000
    assert len(user) < 135_000
    print(
        f"SANITY: {section.action} resolves its own focused prompt, enforced {schema.__name__} and existing 130000-character ceiling; zero real calls."
    )


def test_domain_payloads_keep_unrelated_tables_out(monkeypatch: pytest.MonkeyPatch) -> None:
    combined = "\n\n".join(
        "=== " + label + " ===\n" + value
        for label, value in (
            ("SUMMARY COMPENSATION TABLE", "executive-only"),
            ("DIRECTOR COMPENSATION TABLE", "director-pay-only"),
            ("AUDIT FEE TABLE", "audit-only"),
            ("FIVE PERCENT HOLDERS", "holders-only"),
            ("INSIDER OWNERSHIP", "insiders-only"),
            ("DIRECTOR NOMINEES", "board-only"),
        )
    )
    calls: list[int] = []

    def one_parse(html: str, text: str) -> str:
        calls.append(1)
        return combined

    monkeypatch.setattr(carve, "prepare_def14a_sections", one_parse)
    payloads = prepare_def14a_payloads("", "")
    assert calls == [1]
    assert "executive-only" in payloads["executive_pay"] and "director-pay-only" not in payloads["executive_pay"]
    assert "audit-only" in payloads["audit"] and "executive-only" not in payloads["audit"]
    assert "holders-only" in payloads["ownership"] and "insiders-only" in payloads["ownership"]
    assert "board-only" in payloads["governance"] and "audit-only" not in payloads["governance"]
    assert "director-pay-only" in payloads["director_pay"] and "board-only" not in payloads["director_pay"]
    print("SANITY: one filing parse feeds five domain payloads; compensation, audit, ownership and board tables stay with their requests.")


def test_fee_scale_context_survives_outside_the_table() -> None:
    text = "Audit Fees\nThe following fees paid to Example LLP are in millions.\nAudit Fees 57.6\nTax Fees 1.2"
    payload = prepare_def14a_payloads("", text)["audit"]
    assert "in millions" in payload and "57.6" in payload
    assert "SOURCE UNITS" in payload
    print("SANITY: the fee request retains the adjacent source scale rather than asking the model to infer it from magnitude.")


def test_director_table_retains_its_source_year_and_all_rows() -> None:
    html = """<h2>Director Compensation—2024</h2>
    <p>The table shows compensation earned or paid during 2024.</p>
    <table><tr><th>Name</th><th>Fees Earned or Paid in Cash ($)</th><th>Stock Awards ($)</th><th>Total ($)</th></tr>
    <tr><td>Wanda Austin</td><td>75,000</td><td>275,033</td><td>375,966</td></tr>
    <tr><td>James Bell</td><td>50,000</td><td>0</td><td>61,038</td></tr>
    <tr><td>Sue Wagner</td><td>135,000</td><td>275,033</td><td>416,275</td></tr></table>"""
    text = "Director Compensation—2024\nThe table shows compensation earned or paid during 2024.\n"
    text += "Name Fees Earned or Paid in Cash ($) Stock Awards ($) Total ($)\n"
    text += "Wanda Austin 75,000 275,033 375,966\nJames Bell 50,000 0 61,038\nSue Wagner 135,000 275,033 416,275"

    payload = prepare_def14a_payloads(html, text)["director_pay"]

    assert "compensation earned or paid during 2024" in payload
    assert "Wanda Austin\t75,000\t275,033\t375,966" in payload
    assert "James Bell\t50,000\t0\t61,038" in payload
    assert "Sue Wagner\t135,000\t275,033\t416,275" in payload
    print("SANITY: Apple source-shaped director pay retains the disclosed fiscal year and every complete TSV row without year inference.")


def test_ownership_footnote_table_keeps_the_neighboring_beneficial_totals() -> None:
    html = """<h2>Stock Ownership as of April 11, 2025</h2>
    <table><tr><th>Name of Beneficial Owner</th><th>Direct or Indirect with Sole Power</th><th>Total</th><th>Percent of Class</th></tr>
    <tr><td>Doug McMillon</td><td>1,206,828</td><td>2,260,734</td><td>*</td></tr>
    <tr><td>John Furner</td><td>433,007</td><td>433,007</td><td>*</td></tr>
    <tr><td>Directors and Executive Officers as a Group</td><td>5,530,232</td><td>8,126,426</td><td>*</td></tr></table>
    <p>These amounts include Shares that the following persons hold in the 401(k) Plan:</p>
    <table><tr><th>Name</th><th>Shares Held in the 401(k) Plan</th></tr>
    <tr><td>Doug McMillon</td><td>5,746</td></tr><tr><td>John Furner</td><td>5,612</td></tr>
    <tr><td>Directors and Executive Officers as a Group</td><td>11,358</td></tr></table>"""
    text = "Stock Ownership as of April 11, 2025\nName of Beneficial Owner Direct or Indirect with Sole Power Total Percent of Class\n"
    text += "Doug McMillon 1,206,828 2,260,734 *\nJohn Furner 433,007 433,007 *\n"
    text += "Directors and Executive Officers as a Group 5,530,232 8,126,426 *\n"
    text += "These amounts include Shares that the following persons hold in the 401(k) Plan:\n"
    text += "Name Shares Held in the 401(k) Plan\nDoug McMillon 5,746\nJohn Furner 5,612\nDirectors and Executive Officers as a Group 11,358"

    payload = prepare_def14a_payloads(html, text)["ownership"]

    assert "April 11, 2025" in payload and "2,260,734" in payload and "8,126,426" in payload
    assert "These amounts include Shares" in payload
    assert "Doug McMillon\t5,746" in payload and "Shares Held in the 401(k) Plan" in payload
    print("SANITY: Walmart source-shaped 401(k) selection retains the main beneficial totals, ownership date and footnote scope together.")


def test_table_context_requires_joint_row_evidence_and_preserves_ownership_basis() -> None:
    table = "Name\tBeneficial ownership\tAdditional underlying stock units\tTotal\nJames Dimon\t6,464,764\t721,800\t7,186,564"
    decoy = "Unrelated disclosure repeats 6,464,764. " + "other disclosure " * 1_000
    source = decoy + "Security ownership as of February 28, 2025.\n"
    source += "James Dimon 6,464,764 721,800 7,186,564\nAdditional underlying units include deferred compensation outside beneficial ownership."

    context = carve._table_context(source, table, "ownership")

    assert "Unrelated disclosure" not in context
    assert "Security ownership as of February 28, 2025" in context
    assert "outside beneficial ownership" in context
    assert "6,464,764 721,800 7,186,564" in context
    print("SANITY: a repeated amount alone cannot displace the joint person/value row; ownership column-basis definitions stay adjacent.")


def test_table_context_does_not_bridge_distant_partial_row_matches() -> None:
    table = "Name\tOptions exercisable\tShares held in 401(k)\nJames W. Breyer\t5,512\t0\n"
    table += "Directors and Officers as a Group (24 persons)\t4,735,329\t88,246"
    source = "James W. Breyer received an award of 5,512 units.\n" + "unrelated disclosure " * 10_000
    source += "Holdings of officers and directors as of March 30, 2007.\n"
    source += "James W. Breyer 5,512 0\nDirectors and Officers as a Group (24 persons) 4,735,329 88,246"

    context = carve._table_context(source, table, "ownership")

    assert len(context) < 20_000
    assert "received an award" not in context
    assert "as of March 30, 2007" in context and "4,735,329 88,246" in context
    print("SANITY: the strongest complete ownership row keeps context local when another person's option count also appears far away.")
