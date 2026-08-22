"""
form_registry.py
------------------
Form-dispatch registry: a data-first lookup table (mirrors `schema_registry.
TableSpec`'s shape, not `STRATEGY_REGISTRY`'s call interface) mapping a
normalized SEC form-dispatch key to its discovery mechanism, extraction
handler, and destination table.

Deliberately NOT a generic `run_handler(name, context, tickers)` dispatcher: the
five handlers have genuinely different call signatures (`fetch_13f` walks ALL
filers by filing date, so its `tickers` is only a universe filter and defaults to
the whole universe; `fetch_def14a_llm` takes an extra `model` kwarg) -- forcing a
uniform interface would mean
touching 4 already-working, DAG-scheduled pipelines for no functional gain.
This registry exists to centralize ROUTING/documentation (which form -> which
handler -> which table), consulted by tests and future orchestration work, not
to replace each pipeline's existing call site.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.constants.constants import (
    DEF14A_FORMS, SEC_8K_FORMS, SEC_13D_FORMS, SEC_13F_FORMS,
)
from src.data_extract.utils.prices.fetch_13f import fetch_13f
from src.data_extract.utils.structure.fetch_8k_edgar import fetch_8k_edgar
from src.data_extract.utils.structure.fetch_13d_edgar import fetch_13d_edgar
from src.data_extract.utils.structure.fetch_def14a_edgar import fetch_def14a_edgar
from src.data_extract.utils.structure.fetch_def14a_llm import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_filing_text import (
    FILING_TEXT_FORMS, fetch_filing_text)


@dataclass(frozen=True)
class FormHandlerSpec:
    name: str
    sec_forms: tuple[str, ...]
    discovery: str            # "per_cik_accession" | "all_filers_by_date"
    table: str
    handler: Callable
    call_shape: str
    years_config_key: str | None = None
    step_chain_wired: bool = False
    notes: str = ""


# NOTE: the "fundamentals" entry is absent while the fundamentals stack is being rebuilt
# (see reports/planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md). Phase 3
# re-adds it pointing at `fetch_fundamentals_sec.fetch_fundamentals_sec`.
FORM_REGISTRY: dict[str, FormHandlerSpec] = {
    "sec_8k": FormHandlerSpec(
        name="sec_8k", sec_forms=tuple(SEC_8K_FORMS),
        discovery="per_cik_accession", table="sec_8k",
        handler=fetch_8k_edgar, call_shape="(context, tickers, years_history)",
        step_chain_wired=True,
        notes="edgartools per-filing retrieval (fetch_8k_edgar.py), replacing the "
             "submissions-JSON-only fetch_8k_items.py -- adds has_earnings/has_press_release "
             "from the typed CurrentReport object alongside the item codes. Renamed from "
             "sec_8k_items at the DB level. Grain is (ticker, accession_number, item): one "
             "row PER ITEM CODE, since an 8-K reports 1..n items"),
    "sec_13d": FormHandlerSpec(
        name="sec_13d", sec_forms=tuple(SEC_13D_FORMS),
        discovery="per_cik_accession", table="sec_13d",
        handler=fetch_13d_edgar, call_shape="(context, tickers, years_history)",
        step_chain_wired=True,
        notes="edgartools per-filing retrieval (fetch_13d_edgar.py) reading the typed "
             "Schedule13D object, replacing fetch_13d.py's event/date-only extraction -- adds "
             "reporting-person name/CIK/voting-power + CUSIP + amendment metadata. Grain "
             "changed to (ticker, accession_number, rp_seq): one row PER REPORTING PERSON, "
             "since a single 13D can have multiple co-filers"),
    "def_14": FormHandlerSpec(
        name="def_14", sec_forms=tuple(DEF14A_FORMS),
        discovery="per_cik_accession", table="def14a_llm",
        handler=fetch_def14a_llm,
        call_shape="(context, tickers, model=config.data_extract.llm_model)",
        step_chain_wired=True,
        notes="logical key 'def_14' maps to the EXISTING def14a_llm table -- kept per "
             "the task's own instruction ('keep the def14a_llm table and process as a "
             "complementary one'), not renamed"),
    "def14a_edgar": FormHandlerSpec(
        name="def14a_edgar", sec_forms=tuple(DEF14A_FORMS),
        discovery="per_cik_accession", table="sec_def14a",
        handler=fetch_def14a_edgar, call_shape="(context, tickers, years_history)",
        step_chain_wired=True,
        notes="deterministic complement to def_14's LLM pass: edgartools' typed "
             "ProxyStatement -> sec_def14a + four detail tables, zero LLM cost"),
    "filing_text": FormHandlerSpec(
        name="filing_text", sec_forms=tuple(FILING_TEXT_FORMS),
        discovery="per_cik_accession", table="sec_filing_text",
        handler=fetch_filing_text, call_shape="(context, tickers, years_history)",
        step_chain_wired=True,
        notes="10-K Item 1A + Item 7 and 10-Q Item 2 narrative text, one row per "
             "(ticker, accession, section), for the embedding/drift feature layer"),
    "sec13f_hr": FormHandlerSpec(
        name="sec13f_hr", sec_forms=tuple(SEC_13F_FORMS),
        discovery="all_filers_by_date", table="sec13f_hr",
        handler=fetch_13f, call_shape="(context, tickers=None)  # ALL filers, by filing date",
        step_chain_wired=False,
        notes="renamed from institutional_holdings (DB-level ALTER TABLE RENAME). Discovery "
             "moved off SEC's quarterly bulk data sets (published weeks after a quarter "
             "closes) onto edgartools by filing date; the all-filers grain and the "
             "(cik, period, ticker, cusip) PK are unchanged, so there is no accession column"),
}
