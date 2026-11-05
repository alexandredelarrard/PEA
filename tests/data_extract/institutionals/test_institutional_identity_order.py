"""Institutional extraction consumes the identity built upstream; it never rebuilds it."""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from typing import Any, cast

from src.data_extract.transformers import step_extract_edgar, step_extract_fundamentals
from src.data_extract.transformers import step_extract_institutionals as module
from src.data_extract.transformers.step_extract_institutionals import (
    StepExtractInstitutionals,
)

#: names that build, download for or propagate the identity; none belongs in a domain step.
IDENTITY_STAGE_NAMES = (
    "scan_form345_cache",
    "build_symbol_tenure",
    "symbol_rows",
    "propagate_identity",
    "download_insider_transactions",
    "download_notes_periods",
    "refresh=True",
)


def test_institutionals_consume_the_upstream_identity_in_one_resolver(monkeypatch):
    order: list[str] = []
    identity = object()

    def mark(name: str):
        def call(*args, **kwargs):
            order.append(name)
            return None

        return call

    for name in (
        "fetch_13f",
        "upsert_roster_snapshot",
        "fetch_13f_managers",
    ):
        monkeypatch.setattr(module, name, mark(name))

    def load(*args, **kwargs):
        assert not kwargs.get("refresh"), "the identity is refreshed once, by the identity stage"
        order.append("identity_load")
        return identity

    monkeypatch.setattr(module, "load_identity", load)

    def short(*args, **kwargs):
        assert kwargs["identity"] is identity
        order.append("regsho")

    monkeypatch.setattr(module, "fetch_short_interest", short)

    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    context = SimpleNamespace(config=config, config_dir="./configs", log=logging.getLogger(__name__))
    step = cast(Any, object.__new__(StepExtractInstitutionals))
    step._context = context
    step.config = config
    step.run(["AAA"])

    assert order == ["fetch_13f", "upsert_roster_snapshot", "fetch_13f_managers", "identity_load", "regsho"], order
    assert not hasattr(module, "run_edgar_fetch"), "the EDGAR walks belong to StepExtractEdgar"

    print("\n=== SANITY CHECK: institutional identity consumption ===")
    print(f"  {len(order)} calls: 13F -> roster -> managers -> identity -> RegSHO")
    print("  OK: no identity build inside the step; zips/FTD are StepExtractZips', insider EDGAR/13D/13G/8-K StepExtractEdgar's")


def test_domain_steps_contain_no_identity_stage():
    steps = (module, step_extract_fundamentals, step_extract_edgar)
    found = {step.__name__: [name for name in IDENTITY_STAGE_NAMES if name in inspect.getsource(step)] for step in steps}
    assert all(not names for names in found.values()), found

    print("\n=== SANITY CHECK: consume-only domain steps (AC-011, AC-038) ===")
    print(f"  {len(steps)} step modules scanned for {len(IDENTITY_STAGE_NAMES)} identity-stage names: none found")
    print("  OK: institutionals, fundamentals and EDGAR only read the lineage built and propagated upstream")
