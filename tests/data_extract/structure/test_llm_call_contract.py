"""
test_llm_call_contract.py  (tests/data_extract/structure/)
-------------------------------------------------------------
What the two fetchers ask the LLM for, now that neither owns a client or a prompt.

The point of these is the SEAM, not the extraction: the fetchers must hand the pool the
carved text and the right schema under the right per-action budget, and a single filing's
failure must not cost its ticker the rest.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.schemas.vote_schema import Item507Extract
from src.data_extract.utils.structure.votes import fetch as votes_mod
from src.data_extract.utils.structure.def14a import fetch as def14a_mod
from src.gpt_extract.transformers.gpt_getter import LLMExtractor

_GPT_YML = Path(__file__).resolve().parents[3] / "configs" / "gpt.yml"


def _config():
    return OmegaConf.load(_GPT_YML)


def _context(store=None):
    return types.SimpleNamespace(store=store, log=logging.getLogger("t"), config_dir=Path("."))


class _RecordingProvider:
    """Captures the (schema, system, user) triple every call is made with."""

    name, model, structured = "stub", "stub-model", True

    def __init__(self, calls: list, fail_on: str | None = None):
        self.calls = calls
        self.fail_on = fail_on

    def parse(self, schema, system, user):
        self.calls.append({"schema": schema, "system": system, "user": user})
        if self.fail_on and self.fail_on in user:
            raise RuntimeError("provider blew up on this filing")
        parsed = Def14AExtract() if schema is Def14AExtract else Item507Extract()
        return parsed, {"input_tokens": 5, "output_tokens": 1, "cached_input_tokens": 2}


def _extractor(action: str, calls: list, fail_on: str | None = None, store=None):
    ext = LLMExtractor(_context(store), _config(), action=action, threads=2)
    ext.initialize_client = lambda methode=None, key_index=None: _RecordingProvider(
        calls, fail_on)
    return ext


# --------------------------------------------------------------------------- #
# The action contract                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "action, schema, budget, marker",
    [("def14a", Def14AExtract, 130_000, "SUMMARY COMPENSATION TABLE"),
     ("sec8k_votes", Item507Extract, 40_000, "Item 5.07")],
)
def test_each_action_resolves_its_prompt_schema_and_budget(action, schema, budget, marker):
    calls: list = []
    ext = _extractor(action, calls)
    ext.extract(schema, "x" * 500_000, provider=_RecordingProvider(calls), action=action)

    call = calls[-1]
    assert call["schema"] is schema
    assert marker in call["system"], f"{action} did not load its own system prompt"
    assert len(call["user"]) < budget + 5_000        # template + the truncated payload

    print(f"\n=== SANITY: {action} call contract ===")
    print(f"  schema={schema.__name__}, prompt marker {marker!r} present, "
          f"payload truncated to the {budget:,}-char budget. Validated.")


def test_the_def14a_payload_is_the_carve_not_the_raw_html(monkeypatch):
    """The pool must receive `=== LABEL ===` blocks. Sending raw HTML would be a ~10x bill
    and a worse extraction."""
    html = ("<html><body>" + "<p>boilerplate</p>" * 200
            + "<p>SUMMARY COMPENSATION TABLE</p><table><tr><td>Name</td>"
              "<td>Salary</td></tr><tr><td>Jane Roe</td><td>$1,000,000</td></tr></table>"
            + "</body></html>")
    monkeypatch.setattr(def14a_mod, "_fetch_filing_html", lambda context, filing: html)

    filing = pd.Series({"accession_number": "0000-00-000001", "doc_url": "http://x/p.htm",
                        "filing_date": pd.Timestamp("2024-04-01"),
                        "period_of_report": "2023-12-31", "form": "DEF 14A"})
    payload = def14a_mod._payload_for(_context(), "ZZ", filing)

    # The anchor carve itself is covered by the section tests in test_def14a_llm.py; what
    # this pins is that the POOL receives extracted text rather than the raw markup.
    assert payload is not None
    assert "<table" not in payload and "<html" not in payload and "<p>" not in payload
    assert "SUMMARY COMPENSATION TABLE" in payload
    assert len(payload) < len(html)

    print("\n=== SANITY: def14a payload is text, not markup ===")
    print(f"  {len(html):,} chars of HTML -> {len(payload):,} chars of text, zero tags, "
          "compensation anchor retained. Validated.")


def test_a_filing_that_cannot_be_read_never_becomes_a_task(monkeypatch):
    """A fetch failure must cost zero tokens, not a failed paid call."""
    def _boom(context, filing):
        raise RuntimeError("EDGAR said no")

    monkeypatch.setattr(def14a_mod, "_fetch_filing_html", _boom)
    filing = pd.Series({"accession_number": "0000-00-000002",
                        "filing_date": pd.Timestamp("2024-04-01")})

    assert def14a_mod._payload_for(_context(), "ZZ", filing) is None

    print("\n=== SANITY: unreadable filing ===")
    print("  a filing whose HTML cannot be fetched returns None and never reaches the "
          "queue, so it costs nothing. Validated.")


# --------------------------------------------------------------------------- #
# One filing's failure must not cost its ticker the rest                       #
# --------------------------------------------------------------------------- #
def test_a_failed_filing_does_not_abort_its_ticker():
    class _Store:
        def __init__(self):
            self.saves = []

        def save(self, table, df, pk=None):
            self.saves.append((str(table), len(df)))
            return len(df)

    store = _Store()
    calls: list = []
    ext = _extractor("def14a", calls, fail_on="FILING-2", store=store)

    from src.gpt_extract.utils.schemas_gpt import LlmTask
    tasks = [LlmTask(seq=i, payload=f"FILING-{i}", schema=Def14AExtract,
                     table=def14a_mod.Tables.def14a_llm,
                     meta={"ticker": "ZZ", "filing": pd.Series(
                         {"accession_number": f"acc-{i}",
                          "filing_date": pd.Timestamp("2024-04-01"),
                          "period_of_report": "2023-12-31"})})
             for i in range(4)]

    results = ext.run_extraction(tasks, flatten=def14a_mod._result_frames,
                                 group_key=lambda t: str(t.meta["ticker"]))

    assert len(results) == 4
    assert sum(r.ok for r in results) == 3
    assert not results[2].ok and "blew up" in results[2].error
    parent_saves = [n for name, n in store.saves if name == "def14a_llm"]
    assert parent_saves == [3], f"the 3 good filings must still be saved: {store.saves}"

    print("\n=== SANITY: one filing fails, the ticker survives ===")
    print(f"  4 filings, #2 raised -> 3 parsed and {parent_saves[0]} parent row(s) saved; "
          f"the failure is carried as {results[2].error!r}. Validated.")


# --------------------------------------------------------------------------- #
# Prompts live in .md, not in the code                                         #
# --------------------------------------------------------------------------- #
def test_no_prompt_literals_remain_in_data_extract():
    """The mechanical check for 'prompts live in `.md`'.

    Scoped to MODULE-LEVEL constants bound directly to a string, which is exactly the shape
    the retired `_DEF14A_PROMPT` / `_VOTES_PROMPT` had. Field descriptions, CLI help and
    `form_registry` notes are keyword arguments and a compiled regex is a `Call`, so none of
    them is caught -- those are contracts and documentation, not prose for a model.
    """
    import ast

    root = Path(__file__).resolve().parents[3] / "src" / "data_extract"
    offenders: list[str] = []
    modules = sorted(root.rglob("*.py"))
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:                       # module level only
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if (isinstance(value, ast.Constant) and isinstance(value.value, str)
                    and len(value.value) > 200):
                names = [t.id for t in getattr(node, "targets", [node.target])
                         if isinstance(t, ast.Name)]
                offenders.append(f"{path.relative_to(root)}:{node.lineno} "
                                 f"{names} ({len(value.value)} chars)")

    assert not offenders, "prompt-sized module constants left in src/data_extract:\n  " + \
                          "\n  ".join(offenders)

    print("\n=== SANITY: no prompt literals in src/data_extract ===")
    print(f"  swept {len(modules)} modules; no module-level string constant over 200 "
          "chars. Prompts live in prompt_templates/*.md. Validated.")
