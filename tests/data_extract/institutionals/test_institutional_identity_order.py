"""Institutional extraction refreshes identity immediately before symbol-only tapes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.data_extract.transformers import step_extract_institutionals as module
from src.data_extract.transformers.step_extract_institutionals import (
    StepExtractInstitutionals,
)


def test_identity_refresh_runs_after_insiders_and_before_regsho_ftd(monkeypatch):
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
        "fetch_13d_edgar",
        "fetch_13g_edgar",
        "fetch_8k_edgar",
    ):
        monkeypatch.setattr(module, name, mark(name))
    monkeypatch.setattr(module, "cache_dir", lambda *args: Path("cache"))
    monkeypatch.setattr(module, "build_symbol_tenure", mark("symbol_tenure"))
    monkeypatch.setattr(module, "build_entity_lineage", mark("entity_lineage"))

    def refresh(*args, **kwargs):
        assert kwargs == {"refresh": True}
        order.append("identity_refresh")
        return identity

    monkeypatch.setattr(module, "load_identity", refresh)

    def short(*args, **kwargs):
        assert kwargs["identity"] is identity
        order.append("regsho")

    def fails(*args, **kwargs):
        assert kwargs["identity"] is identity
        order.append("ftd")

    monkeypatch.setattr(module, "fetch_short_interest", short)
    monkeypatch.setattr(module, "fetch_fails_to_deliver", fails)

    config = SimpleNamespace(
        data_extract=SimpleNamespace(years_history=15),
        local=SimpleNamespace(paths=SimpleNamespace(insider_transactions="sec_insider_transactions")),
    )
    context = SimpleNamespace(config=config, config_dir="./configs")
    step = object.__new__(StepExtractInstitutionals)
    step._context = context
    step.config = config
    step.run(["AAA"])

    assert order.index("fetch_insider_transactions") < order.index("symbol_tenure")
    assert order[-5:] == ["symbol_tenure", "entity_lineage", "identity_refresh", "regsho", "ftd"]

    print("\n=== SANITY CHECK: institutional identity refresh order ===")
    print("  insider cache -> symbol_tenure -> entity_lineage -> refresh -> RegSHO -> FTD")
    print("  OK: both symbol-only consumers share the same newly refreshed resolver")
