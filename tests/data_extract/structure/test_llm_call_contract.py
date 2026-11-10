"""
test_llm_call_contract.py  (tests/data_extract/structure/)
-------------------------------------------------------------
What the two fetchers ask the LLM for, now that neither owns a client or a prompt.

The point of these is the SEAM, not the extraction: the fetchers must hand the pool the
carved text, whole, with the right prompt and schema, and a single filing's
failure must not cost its ticker the rest.
"""

from __future__ import annotations

import ast
import logging
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_extract.utils.structure.def14a import fetch as def14a_mod
from src.data_extract.utils.structure.def14a.def14a_schema import Def14AExtract as _Def14AExtract
from src.data_extract.utils.structure.def14a.flatten import _result_frames
from src.data_extract.utils.structure.votes.vote_schema import Item507Extract as _Item507Extract
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmTask

_GPT_YML = Path(__file__).resolve().parents[3] / "configs" / "gpt.yml"
Def14AExtract: Any = _Def14AExtract
Item507Extract: Any = _Item507Extract


def _config() -> Any:
    return OmegaConf.load(_GPT_YML)


def _context(store=None) -> Any:
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
        parsed = Def14AExtract(ceo_name="Jane CEO") if schema is Def14AExtract else Item507Extract()
        return parsed, {"input_tokens": 5, "output_tokens": 1, "cached_input_tokens": 2}


def _extractor(action: str, calls: list, fail_on: str | None = None, store=None) -> Any:
    ext: Any = LLMExtractor(_context(store), _config(), action=action, threads=2)
    ext.initialize_client = lambda methode=None, key_index=None: _RecordingProvider(calls, fail_on)
    return ext


# --------------------------------------------------------------------------- #
# The action contract                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "action, schema, marker",
    [("def14a_executive_pay", Def14AExtract, "Summary Compensation Table"), ("sec8k_votes", Item507Extract, "Item 5.07")],
)
def test_each_action_resolves_its_prompt_and_schema_and_sends_the_payload_whole(action, schema, marker):
    calls: list = []
    ext = _extractor(action, calls)
    payload = "x" * 500_000
    results = ext.run([LlmTask(seq=0, payload=payload, schema=schema)])

    call = calls[-1]
    assert results[0].ok and len(calls) == 1
    assert call["schema"] is schema
    assert marker in call["system"], f"{action} did not load its own system prompt"
    assert payload in call["user"] and len(call["user"]) < len(payload) + 5_000  # template + the whole payload

    print(f"\n=== SANITY: {action} call contract ===")
    print(
        f"  schema={schema.__name__}, prompt marker {marker!r} present, the whole {len(payload):,}-char payload sent (the step cuts, not the extractor). Validated."
    )


# --------------------------------------------------------------------------- #
# One filing's failure must not cost its ticker the rest                       #
# --------------------------------------------------------------------------- #
def test_a_failed_filing_does_not_abort_its_ticker():
    class _Store:
        def __init__(self):
            self.saves = []

        def save(self, table, df, pk=None):
            self.saves.append((str(table), df.copy()))
            return len(df)

    store = _Store()
    calls: list = []
    ext = _extractor("def14a_executive_pay", calls, fail_on="FILING-2", store=store)

    tasks = [
        LlmTask(
            seq=i,
            payload=f"FILING-{i}",
            schema=Def14AExtract,
            table=def14a_mod.Tables.def14a_llm,
            meta={
                "ticker": "ZZ",
                "filing": pd.Series({"accession_number": f"acc-{i}", "filing_date": pd.Timestamp("2024-04-01"), "period_of_report": "2023-12-31"}),
            },
        )
        for i in range(4)
    ]

    results = ext.run_extraction(tasks, flatten=_result_frames, group_key=lambda t: str(t.meta["ticker"]))

    assert len(results) == 4
    assert sum(r.ok for r in results) == 3
    assert results[2].error is not None
    assert not results[2].ok and "blew up" in results[2].error
    parent_saves = [frame for name, frame in store.saves if name == "def14a_llm"]
    assert [len(frame) for frame in parent_saves] == [3], f"the 3 good filings must still be saved: {store.saves}"
    assert all("acc-2" not in set(frame.get("accession_number", pd.Series(dtype=str))) for _, frame in store.saves), (
        "the exact failed accession must be absent from every parent/child save"
    )

    print("\n=== SANITY: one filing fails, the ticker survives ===")
    print(
        f"  4 filings, #2 raised -> 3 parsed and {len(parent_saves[0])} parent row(s) saved; "
        f"acc-2 is absent from every save and the failure is carried as {results[2].error!r}. Validated."
    )


# --------------------------------------------------------------------------- #
# Prompts live in .md, not in the code                                         #
# --------------------------------------------------------------------------- #
def _is_field_description_only(tree: ast.Module, name: str) -> bool:
    """Exempt a constant only when every read is direct Pydantic Field metadata."""
    field_imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pydantic"
        and node.level == 0
        and any(alias.name == "Field" and alias.asname in (None, "Field") for alias in node.names)
        for node in tree.body
    )
    if not field_imported:
        return False
    reads = {node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)}
    descriptions = {
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Field"
        for keyword in node.keywords
        if keyword.arg == "description" and isinstance(keyword.value, ast.Name)
    }
    return bool(reads) and reads <= descriptions


@pytest.mark.parametrize(
    "field_import, usage, expected",
    [
        ("from pydantic import Field", "class Answer:\n    first = Field(description=TEXT)\n    second = Field(description=TEXT)", True),
        ("from pydantic import Field", "def prompt():\n    return TEXT", False),
        ("from pydantic import Field", "class Answer:\n    field = Field(description=TEXT)\nprompt = TEXT", False),
        ("from pydantic import Field", "", False),
        ("from pydantic import Field", "class Answer:\n    field = Field(default=TEXT)", False),
        ("from other_library import Field", "class Answer:\n    field = Field(description=TEXT)", False),
        ("", "class Answer:\n    field = Field(description=TEXT)", False),
    ],
    ids=["description-only", "prompt", "mixed", "unused", "other-keyword", "non-pydantic", "unresolved-field"],
)
def test_field_description_exemption_requires_all_reads(field_import: str, usage: str, expected: bool) -> None:
    tree = ast.parse(f"{field_import}\nTEXT = {'Known-truth extraction instructions. ' * 10!r}\n{usage}")

    assert _is_field_description_only(tree, "TEXT") is expected

    print(f"\nSANITY: description-only exemption={expected}; prompt, mixed, unused and unrelated uses remain guarded.")


def test_no_prompt_literals_remain_in_data_extract():
    """The mechanical check for 'prompts live in `.md`'.

    Scoped to MODULE-LEVEL constants bound directly to a string, which is exactly the shape
    the retired `_DEF14A_PROMPT` / `_VOTES_PROMPT` had. Direct keyword documentation and
    compiled regex calls are outside that shape. A constant used exclusively as direct
    Pydantic Field description metadata is also exempt; any other or mixed use stays guarded.
    """
    root = Path(__file__).resolve().parents[3] / "src" / "data_extract"
    offenders: list[str] = []
    modules = sorted(root.rglob("*.py"))
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and len(value.value) > 200:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if names and all(_is_field_description_only(tree, name) for name in names):
                    continue
                offenders.append(f"{path.relative_to(root)}:{node.lineno} {names} ({len(value.value)} chars)")

    assert not offenders, "prompt-sized module constants left in src/data_extract:\n  " + "\n  ".join(offenders)

    print("\n=== SANITY: no prompt literals in src/data_extract ===")
    print(f"  swept {len(modules)} modules; no prompt-sized constants outside schema descriptions. Prompts live in prompt_templates/*.md. Validated.")
