"""Real-data contract for the JCI 2016 Tyco 0.955 share consolidation in the price register."""

from __future__ import annotations

import pandas as pd
import pytest

from src.constants.constants import SHARADAR_ACTION_SPLIT
from src.context import get_config_context
from src.data_store.schema import Tables
from src.utils.price.adjustments import load_bugfix, resolve_share_events
from src.utils.price.levels import security_basis


def test_jci_tyco_consolidation_steps_the_quantity_basis_only() -> None:
    _, context = get_config_context("./configs", use_cache=False, save=False)
    store = context.store
    df_actions = store.load(
        Tables.sharadar_actions, columns=["ticker", "date", "action", "value"], where={"ticker": ["JCI"], "action": [SHARADAR_ACTION_SPLIT]}
    )
    df_splits = store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": ["JCI"]})
    df_vendor = store.load(Tables.sharadar_fundamentals, columns=["ticker", "date", "sharesbas"], where={"ticker": ["JCI"], "dimension": "ARQ"})
    df_bars = store.load(Tables.prices, columns=["ticker", "date", "close_split"], where={"ticker": ["JCI"]}, since="2016-09-02", until="2016-09-06")
    assert df_actions is not None and df_splits is not None and df_vendor is not None and df_bars is not None

    basis = resolve_share_events(df_actions, df_splits, load_bugfix(context.config_dir), shares=df_vendor, tickers=["JCI"])
    sessions = pd.Series(pd.to_datetime(["2016-09-02", "2016-09-06"]))
    factor = security_basis(basis, pd.Series(["JCI", "JCI"]), sessions)
    vendor = basis.factor(pd.Series(["JCI", "JCI"]), sessions, "vendor")
    assert factor.iloc[1] / factor.iloc[0] == pytest.approx(0.955, rel=1e-12)
    assert vendor.iloc[0] == vendor.iloc[1]

    # Issuer-independent FTD prior-day closes: TYC 45.59 on 09-02, JCI 48.90 on 09-06 (first new-basis session).
    close = pd.to_numeric(df_bars.set_index(pd.to_datetime(df_bars["date"]))["close_split"])
    implied = (48.90 / 45.59) / (close.loc["2016-09-06"] / close.loc["2016-09-02"])
    assert implied == pytest.approx(1 / 0.955, rel=5e-4)
    print(
        f"SANITY: JCI F steps x{factor.iloc[1] / factor.iloc[0]:.6f} on 2016-09-06, vendor counts unrestated; as-traded quotes imply 1/{1 / implied:.5f}."
    )
