"""The empty frame each fundamentals facts producer writes, typed exactly as production types it.

`sql/schema.sql` holds no block for the four fundamentals tables (`managed=False`): `store.ensure_table` creates each
from its producer's frame, so a test needing their DDL derives it from these frames through `src.data_store.ddl`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.constants.constants import SHARADAR_SF1_COLUMNS
from src.data_extract.utils.edgar_extract.edgar_driver import FilingStamp
from src.data_extract.utils.fundamentals import fetch_fundamentals_sec as facts_mod
from src.data_extract.utils.fundamentals_sharadar.client import cast_value_columns, coerce_date_columns
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_filing


def sec_facts_frame(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """`parse_fundamentals` on a filing with no facts: the full typed `sec_fundamentals_facts` layout, zero rows."""
    monkeypatch.setattr(facts_mod, "filing_rows", lambda ticker, stamp, catalogue, gics: [])
    stamp = FilingStamp.of(fake_filing("0000000001-26-000001", 1, "2026-09-10", form="10-Q"), "0000000001")
    return facts_mod.parse_fundamentals("AAA", "0000000001", stamp, None, catalogue=None, gics_by_ticker={})[Tables.sec_fundamentals_facts]  # type: ignore[arg-type]


def sharadar_facts_frame() -> pd.DataFrame:
    """The SF1 response header after the fetch's casts: the full typed `sharadar_fundamentals_facts` layout, zero rows."""
    frame = cast_value_columns(pd.DataFrame(columns=list(SHARADAR_SF1_COLUMNS)))
    return coerce_date_columns(frame, Tables.sharadar_fundamentals_facts.date_type_cols)
