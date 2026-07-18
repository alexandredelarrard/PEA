"""The yfinance forward snapshot ACCRUES point-in-time history (append, not
overwrite) — keyed on (ticker, as_of)."""
from __future__ import annotations

import os

import pandas as pd
import pytest


@pytest.mark.skipif(not os.getenv("DATABASE_URL"),
                    reason="DATABASE_URL not set — needs the Postgres DB")
def test_fundamentals_snapshot_accrues_across_days():
    from sqlalchemy import text
    from src.utils.db import get_engine
    from src.data_store.store import DataStore

    try:
        st = DataStore(get_engine())
        st.exists("fundamentals_snapshot")
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"DB not reachable: {e}")

    def _clean():
        if st.exists("fundamentals_snapshot"):
            with st.engine.begin() as c:
                c.execute(text("DELETE FROM fundamentals_snapshot WHERE ticker IN ('ZZA','ZZB')"))

    _clean()
    try:
        d1 = pd.DataFrame({"ticker": ["ZZA", "ZZB"], "as_of": ["2026-01-05"] * 2,
                           "marketCap": [1e9, 2e9], "forwardPE": [10.0, 20.0]})
        d2 = pd.DataFrame({"ticker": ["ZZA", "ZZB"], "as_of": ["2026-01-06"] * 2,
                           "marketCap": [1.1e9, 2.1e9], "forwardPE": [11.0, 21.0]})

        st.save("fundamentals_snapshot", d1)
        st.save("fundamentals_snapshot", d2)
        st.save("fundamentals_snapshot", d2)      # same-day re-run must be idempotent

        z = st.load("fundamentals_snapshot")
        z = z[z["ticker"].isin(["ZZA", "ZZB"])]
        assert len(z) == 4, f"expected 2 tickers x 2 days = 4 accrued rows, got {len(z)}"
        assert z["as_of"].astype(str).nunique() == 2      # two distinct snapshot days
        # newest day's value preserved
        newest = z.sort_values("as_of").groupby("ticker").tail(1).set_index("ticker")
        assert float(newest.loc["ZZA", "forwardPE"]) == 11.0

        print("\n=== SANITY CHECK: fundamentals_snapshot accrual ===")
        print(f"  2 run-days x 2 tickers -> {len(z)} rows across "
              f"{z['as_of'].astype(str).nunique()} dates (append, not overwrite); "
              f"same-day re-run idempotent. Validated.")
    finally:
        _clean()
