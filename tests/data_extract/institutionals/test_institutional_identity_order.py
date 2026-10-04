"""Institutional extraction consumes the identity built upstream; it never rebuilds it."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast

from src.data_extract.transformers import step_extract_fundamentals, step_extract_structure
from src.data_extract.transformers import step_extract_institutionals as module
from src.data_extract.transformers.step_extract_institutionals import (
    StepExtractInstitutionals,
)

#: names that build, download for or propagate the identity; none belongs in a domain step.
IDENTITY_STAGE_NAMES = (
    "scan_form345_cache",
    "build_symbol_tenure",
    "build_entity_lineage",
    "propagate_identity",
    "download_insider_transactions",
    "download_financial_notes",
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
        "fetch_insider_transactions",
        "fetch_insider_edgar",
    ):
        monkeypatch.setattr(module, name, mark(name))
    # 13D, 13G and 8-K run through the shared driver, one EdgarFetch declaration each
    monkeypatch.setattr(module, "run_edgar_fetch", lambda *args, **kwargs: order.append(kwargs["fetch"].desc))

    def load(*args, **kwargs):
        assert not kwargs.get("refresh"), "the identity is refreshed once, by the identity stage"
        order.append("identity_load")
        return identity

    monkeypatch.setattr(module, "load_identity", load)

    def short(*args, **kwargs):
        assert kwargs["identity"] is identity
        order.append("regsho")

    def fails(*args, **kwargs):
        assert kwargs["identity"] is identity
        order.append("ftd")

    monkeypatch.setattr(module, "fetch_short_interest", short)
    monkeypatch.setattr(module, "fetch_fails_to_deliver", fails)

    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    context = SimpleNamespace(config=config, config_dir="./configs")
    step = cast(Any, object.__new__(StepExtractInstitutionals))
    step._context = context
    step.config = config
    step.run(["AAA"])

    assert order[:5] == ["fetch_13f", "upsert_roster_snapshot", "fetch_13f_managers", "fetch_insider_transactions", "fetch_insider_edgar"]
    assert order[-3:] == ["identity_load", "regsho", "ftd"]
    assert len(order) == 11, order

    print("\n=== SANITY CHECK: institutional identity consumption ===")
    print(f"  {len(order)} calls: 13F -> roster -> managers -> insider parse -> insider live -> 13D -> 13G -> 8-K -> identity -> RegSHO -> FTD")
    print("  OK: no identity build inside the step; both symbol tapes share the one resolver the identity stage refreshed")


def test_domain_steps_contain_no_identity_stage():
    steps = (module, step_extract_fundamentals, step_extract_structure)
    found = {step.__name__: [name for name in IDENTITY_STAGE_NAMES if name in inspect.getsource(step)] for step in steps}
    assert all(not names for names in found.values()), found

    print("\n=== SANITY CHECK: consume-only domain steps (AC-011, AC-038) ===")
    print(f"  {len(steps)} step modules scanned for {len(IDENTITY_STAGE_NAMES)} identity-stage names: none found")
    print("  OK: institutionals, fundamentals and structure only read the lineage built and propagated upstream")
