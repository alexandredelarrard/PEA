"""Focused request, historical reach and daily domain-repair contracts."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.context import Context
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a import carve
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_payloads, prepare_def14a_sections
from src.data_extract.utils.structure.def14a.evidence import prose_has_gap, table_has_gap
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


@pytest.mark.parametrize(
    ("source", "independent", "expected"),
    [
        ("Name\tAge\tDirector Since\tIndependent\nJane Example\t54\t2018\tYes", None, True),
        ("Name\tAge\tDirector Since\tIndependence\nJane Example\t54\t2018\tNo", None, True),
        ("Jane Example is independent.", None, True),
        ("Jane Example is not an independent director.", None, True),
        ("Name\tIndependent\nJane Example\tYes", False, False),
        ("Name\tIndependent\nJane Example\tNo", False, False),
        ("Jane Example is not independent.", False, False),
        ("Name\tIndependent\nJane Example\t-", None, False),
        ("Name\tIndependent\nJane Example\t?", None, False),
        ("Name\tIndependent\nJane Example\t1", None, False),
        ("Name\tStatus\nJane Example\tYes", None, False),
        ("Name\tIndependent", None, False),
        ("Jane Example may be independent.", None, False),
        ("Jane Example is independent?", None, False),
        ("Jane Example is an independent consultant.", None, False),
    ],
)
def test_independence_repair_requires_a_named_explicit_disclosure(source: str, independent: bool | None, expected: bool) -> None:
    section = next(section for section in SECTIONS if section.key == "governance")
    saved = pd.DataFrame([dict(name="Jane Example", age=54, tenure_years=8, is_independent=independent)])
    parent = dict(ceo_name_proxy="Jane Example", ceo_age=54, ceo_since_year=2018, board_size=1, ceo_pay_ratio=100, say_on_pay_support_pct=0.95)

    detected = table_has_gap(section, source, saved, parent) or prose_has_gap(section, source, saved, parent)
    assert detected is expected
    print(f"SANITY: named independence evidence gives repair={expected}; saved False and ambiguous or absent disclosures remain complete/unproved.")
