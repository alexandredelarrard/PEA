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

#: names that build, download for or sync the identity; none belongs in a domain step.
IDENTITY_STAGE_NAMES = (
    "scan_form345_cache",
    "build_symbol_tenure",
    "symbol_rows",
    "identity_sync",
    "restamp_insider_lineage",
    "download_insider_transactions",
    "download_notes_periods",
    "refresh=True",
)


def test_institutionals_build_no_identity_and_store_regsho_as_filed(monkeypatch):
    order: list[str] = []

    def mark(name: str):
        def call(*args, **kwargs):
            assert "identity" not in kwargs, f"{name} takes no resolver: RegSHO is stored as filed"
            order.append(name)
            return None

        return call

    for name in ("fetch_13f", "upsert_roster_snapshot", "fetch_short_interest"):
        monkeypatch.setattr(module, name, mark(name))

    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    context = SimpleNamespace(config=config, config_dir="./configs", log=logging.getLogger(__name__))
    step = cast(Any, object.__new__(StepExtractInstitutionals))
    step._context = context
    step.config = config
    step.run(["AAA"])

    assert order == ["fetch_13f", "upsert_roster_snapshot", "fetch_short_interest"], order
    assert not hasattr(module, "run_edgar_fetch"), "the EDGAR walks belong to StepExtractEdgar"
    assert not hasattr(module, "load_identity"), "the RegSHO tape needs no resolver"

    print("\n=== SANITY CHECK: institutional identity consumption ===")
    print(f"  {len(order)} calls: 13F -> roster -> managers -> RegSHO (as filed, no resolver)")
    print("  OK: no identity build inside the step; zips/FTD are StepExtractZips', insider EDGAR/13D/13G/8-K StepExtractEdgar's")


def test_domain_steps_contain_no_identity_stage():
    steps = (module, step_extract_fundamentals, step_extract_edgar)
    found = {step.__name__: [name for name in IDENTITY_STAGE_NAMES if name in inspect.getsource(step)] for step in steps}
    assert all(not names for names in found.values()), found

    print("\n=== SANITY CHECK: consume-only domain steps (AC-011, AC-038) ===")
    print(f"  {len(steps)} step modules scanned for {len(IDENTITY_STAGE_NAMES)} identity-stage names: none found")
    print("  OK: institutionals, fundamentals and EDGAR only read the lineage built and synced upstream")
