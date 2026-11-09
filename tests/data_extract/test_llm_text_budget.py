"""The LLM text budgets live in `configs/configs.yml` `data_extract.llm_text_budget`."""

from __future__ import annotations

from src.utils.config import read_config


def test_llm_text_budget_is_declared_in_the_extract_config() -> None:
    config = read_config("configs")

    budget = config.data_extract.llm_text_budget

    assert dict(budget) == {"employees": 60000, "def14a": 130000}
    assert all(isinstance(value, int) and value > 0 for value in budget.values())
    print("\n=== SANITY CHECK: data_extract.llm_text_budget ===")
    print(f"  employees {budget.employees} chars, def14a {budget.def14a} chars: today's cuts, read from configs.yml. Validated.")
