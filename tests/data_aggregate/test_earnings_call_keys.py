"""Source-key planning uses one projection independent of ticker count."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_aggregate.test_step_cube_text import seeded_case
from tests.fixtures.earnings_call_rows import FIXTURE_PATHS


def test_call_keys_reads_once_for_all_tickers(sqlite_store: DataStore) -> None:
    context, config, store = seeded_case(sqlite_store, [p.stem for p in FIXTURE_PATHS])
    with patch.object(store, "load", wraps=store.load) as reads:
        plan = StepCubeText(context, config).plan_window(pd.bdate_range("2020-01-01", "2027-03-01"), full=True)
    source_reads = [call for call in reads.call_args_list if call.args[0] == Tables.earnings_call_sections]
    assert len(source_reads) == 1 and source_reads[0].kwargs["columns"] == ["ticker", "quarter", "as_of"]
    assert source_reads[0].kwargs["where"] == {"paragraph": 1}
    assert len(plan.source) == len(FIXTURE_PATHS)
    print("SANITY: all real call keys/dates planned in one compact read without transcript text.")


@pytest.mark.parametrize("empty_table", [False, True])
def test_call_keys_handles_no_stored_calls(sqlite_store: DataStore, empty_table: bool) -> None:
    context, config, store = seeded_case(sqlite_store, [FIXTURE_PATHS[0].stem])
    if empty_table:
        store.delete(Tables.earnings_call_sections, {"ticker": store.distinct(Tables.earnings_call_sections, "ticker")})
    else:
        store.drop(Tables.earnings_call_sections)
    plan = StepCubeText(context, config).plan_window(pd.bdate_range("2020-01-01", "2027-03-01"), full=True)
    assert plan.source.empty and not plan.sentiment_work and not plan.embedding_work
    print("SANITY: missing/empty source yields no model work and an explicit empty call frame.")
