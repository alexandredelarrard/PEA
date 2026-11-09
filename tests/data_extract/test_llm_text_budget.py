"""The LLM text budgets live in `configs/configs.yml` `data_extract.llm_text_budget`."""

from __future__ import annotations

from src.data_extract.utils.edgar_extract import edgar_driver
from src.data_extract.utils.edgar_extract.fetch_employees_text import EMPLOYEES_TEXT_FETCH
from src.data_extract.utils.fundamentals.fundamentals_employees import _GAP, employee_task
from src.data_store.schema import Tables
from src.utils.config import read_config
from tests.data_extract.edgar_fixtures import TenK

FILLER = "The Company designs and sells measurement instruments to laboratories in many markets. "  # no workforce word
#: A long 10-K: 200 generic workforce mentions, then the workforce sentence.
LONG_10K = "".join(f"Our employees value safety and training, item {i}. {FILLER * 15}" for i in range(200)) + "We had 41,000 employees."


def test_llm_text_budget_is_declared_in_the_extract_config() -> None:
    config = read_config("configs")

    budget = config.data_extract.llm_text_budget

    assert dict(budget) == {"employees": 60000, "def14a": 130000}
    assert all(isinstance(value, int) and value > 0 for value in budget.values())
    assert "employees" not in config.gpt.max_chars
    assert not [key for key in config.gpt.max_chars if key.startswith("def14a")], dict(config.gpt.max_chars)
    print("\n=== SANITY CHECK: data_extract.llm_text_budget ===")
    print(
        f"  employees {budget.employees} chars, def14a {budget.def14a} chars: today's cuts, read from configs.yml; "
        f"gpt.max_chars keeps only {sorted(config.gpt.max_chars)} (no employees, no def14a*). Validated."
    )


def test_a_larger_employees_budget_changes_the_excerpt_from_stored_rows() -> None:
    """AC-008: the stored row is parsed once; each budget rebuilds its own excerpt from it, with no further read."""
    filing = TenK("acc-1", "2025-03-01", LONG_10K, report="2024-12-31")
    stamp = edgar_driver.FilingStamp.of(filing, "0000000001")
    assert EMPLOYEES_TEXT_FETCH.parse is not None
    row = next(EMPLOYEES_TEXT_FETCH.parse("AAA", "0000000001", stamp, edgar_driver.EdgarScope())[Tables.sec_employees_text].itertuples(index=False))

    tasks = {budget: employee_task(0, "AAA", row, budget) for budget in (60_000, 80_000)}

    sizes = {budget: len(task.payload) for budget, task in tasks.items()}
    parts = [part for task in tasks.values() for part in str(task.meta["source_text"]).split(f"\n\n{_GAP}\n\n")]
    assert 60_000 - 1_500 < sizes[60_000] <= 60_000 < sizes[80_000] <= 80_000
    assert all(part in filing.body for part in parts) and "We had 41,000 employees." in str(tasks[60_000].meta["source_text"])
    print("\n=== SANITY CHECK: budget 80000 from the stored row (AC-008) ===")
    print(f"  one stored row of a {len(filing.body):,}-char 10-K: payload {sizes[60_000]:,} chars at 60000, {sizes[80_000]:,} at 80000;")
    print(f"  all {len(parts)} excerpt parts are verbatim document text and the workforce sentence is kept. No walk re-read.")
