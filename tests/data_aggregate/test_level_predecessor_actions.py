"""The cube's level factor S(d) inside a predecessor vendor series window reads the window owner's `sharadar_actions`.

JCI's prices before 2016-09-02 are Tyco's, but ticker JCI's own actions there are old Johnson Controls' (2004 x2,
2007-10 x3). Known-truth fixture: live JCI `prices_splits`, live JCI and TYC actions, real Sharadar TYC ARQ rows.
Both S entry points (the cube price step and the validate price panel) are exercised. `FakeStore`, no database.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.utils.common.level_basis import genuine_splits, level_factor
from src.data_store.schema import Tables
from src.validate.utils import prices as vprices
from tests.conftest import FakeStore

LOGGER = "test.level_predecessor_actions"
SPIN_2012 = 2.011667672500503
#: Yahoo's JCI split events (prices_splits, live 2026-10-07): Tyco's splits, the 2012 ADT/Pentair spin, the 2016 consolidation.
JCI_YF = [
    ("JCI", "1995-11-15", 2.0),
    ("JCI", "1997-10-23", 2.0),
    ("JCI", "1999-10-22", 2.0),
    ("JCI", "2007-07-02", 0.25),
    ("JCI", "2012-10-01", SPIN_2012),
    ("JCI", "2016-09-06", 0.955),
]
#: sharadar_actions splits and spinoffs (live 2026-10-07): old JCI's own splits under JCI, Tyco's under TYC.
JCI_ACTIONS = [
    ("JCI", "2004-01-05", "split", 2.0),
    ("JCI", "2007-10-03", "split", 3.0),
    ("JCI", "2016-10-31", "spinoff", 0.1),
    ("TYC", "1999-10-22", "split", 2.0),
    ("TYC", "2007-07-02", "spinoff", 1.0),
    ("TYC", "2007-07-02", "split", 0.25),
    ("TYC", "2012-10-01", "spinoff", 0.5),
    ("TYC", "2012-10-01", "spinoff", 0.23994),
]
#: Real Sharadar TYC ARQ rows (`_cache/tyc_sf1.csv`): (date, sharesbas, marketcap, JCI close_split on `date`).
TYC_ROWS = [
    ("1997-02-13", 156679751, 9244105309, 30.710890),
    ("2003-05-15", 499221544, 33947065026, 35.395603),
    ("2006-05-09", 509150440, 56821189048, 58.090427),
    ("2011-01-27", 473753233, 21195719644, 23.288223),
    ("2012-07-31", 459875217, 25265544422, 28.597565),
    ("2012-11-16", 465717368, 12467253941, 28.031414),
    ("2016-07-29", 426224367, 19423044404, 47.717278),
]
#: An ordinary ticker: one corroborated split and one Yahoo-only spinoff factor.
PLAIN_YF = [("AAA", "2010-06-01", 2.0), ("AAA", "2015-03-02", 0.97)]
PLAIN_ACTIONS = [("AAA", "2010-06-01", "split", 2.0), ("AAA", "2015-03-02", "spinoff", 0.2)]
JCI_OVERRIDE = {
    "ticker": "JCI",
    "vendor_ticker": "TYC",
    "cik": "0000833444",
    "valid_from": None,
    "valid_to": "2016-09-02",
    "source": "0001104659-16-143068",
}
DAYS = pd.DatetimeIndex(
    ["1997-02-13", "2003-05-15", "2006-05-09", "2011-01-27", "2012-07-31", "2012-11-16", "2016-07-29", "2016-10-03", "2017-01-03"]
)
#: S(d) with Tyco's actions: Tyco's own splits cancel, the 2012 spin and the 2016 consolidation are price-only.
TYCO_S = [SPIN_2012 * 0.955] * 5 + [0.955, 0.955, 1.0, 1.0]


def _context(tmp_path: Path, overrides: list[dict]) -> SimpleNamespace:
    (tmp_path / "sec").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sec" / "security_master_manual.json").write_text(json.dumps({"vendor_series_overrides": overrides}), encoding="utf-8")
    store = FakeStore(
        {
            Tables.sharadar_actions: pd.DataFrame(
                [{"ticker": t, "date": pd.Timestamp(d), "action": a, "value": v} for t, d, a, v in JCI_ACTIONS + PLAIN_ACTIONS]
            ),
            Tables.prices_splits: pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "ratio": v} for t, d, v in JCI_YF + PLAIN_YF]),
        }
    )
    return SimpleNamespace(store=store, log=logging.getLogger(LOGGER), config_dir=tmp_path)


def _cube_s(context: SimpleNamespace, tickers: list[str]) -> pd.DataFrame:
    """S(d) through `StepCubePrices._level_factor`, on a close grid that is never null."""
    step = StepCubePrices.__new__(StepCubePrices)
    step._context, step._store, step._log, step._bugfix = context, context.store, context.log, {}
    close = pd.DataFrame(1.0, index=DAYS, columns=tickers)
    return step._level_factor(DAYS, close)


def _validate_s(context: SimpleNamespace, tickers: list[str]) -> pd.DataFrame:
    """S(d) through the validate price panel's `_level_factor_for`."""
    panel = pd.DataFrame([{"ticker": t, "date": d, "close_split": 1.0, "price": 1.0} for t in tickers for d in DAYS])
    flat = vprices._level_factor_for(context, panel, {"ticker": tickers})  # type: ignore[arg-type]
    return panel.assign(S=flat).pivot(index="date", columns="ticker", values="S")


def test_jci_level_factor_reads_tyco_actions_before_the_seam(tmp_path: Path) -> None:
    context = _context(tmp_path, [JCI_OVERRIDE])
    cube, check = _cube_s(context, ["JCI"])["JCI"], _validate_s(context, ["JCI"])["JCI"]
    rows = pd.DataFrame(TYC_ROWS, columns=["date", "sharesbas", "marketcap", "close_split"]).assign(date=lambda f: pd.to_datetime(f["date"]))
    s_rows = cube.reindex(rows["date"]).to_numpy()
    ratio = rows["close_split"].to_numpy() * s_rows * rows["sharesbas"].to_numpy() / rows["marketcap"].to_numpy()
    print("\n=== SANITY CHECK: JCI S(d) from TYC's actions inside the window ===")
    print(pd.DataFrame({"cube_S": cube, "validate_S": check, "expected": TYCO_S}).round(6).to_string())
    print(pd.DataFrame({"date": rows["date"].dt.date, "cube_mc_over_tyc_mc": ratio.round(6)}).to_string(index=False))
    assert cube.tolist() == pytest.approx(TYCO_S, rel=1e-12)
    assert check.tolist() == pytest.approx(TYCO_S, rel=1e-12)
    assert np.abs(ratio - 1).max() < 1e-3
    print("  OK: before 2004, 2004..2007-10 and 2007-10..2012-09 S = 2.0117 x 0.955 (Tyco's splits cancel, old JCI's are")
    print("  gone), 0.955 to the seam, 1.0 after; cube and validate agree; close_split x S x sharesbas = TYC marketcap within 0.1 %.")


def test_a_ticker_without_a_vendor_series_is_unchanged(tmp_path: Path) -> None:
    context = _context(tmp_path, [JCI_OVERRIDE])
    actions = context.store.load(Tables.sharadar_actions, where={"ticker": ["AAA"]})
    yf = context.store.load(Tables.prices_splits, where={"ticker": ["AAA"]})
    raw = level_factor(DAYS, ["AAA"], yf, genuine_splits(actions, yf))["AAA"]
    cube, check = _cube_s(context, ["AAA", "JCI"])["AAA"], _validate_s(context, ["AAA"])["AAA"]
    plain = _cube_s(_context(tmp_path / "none", []), ["JCI"])["JCI"]
    jci_raw = level_factor(DAYS, ["JCI"], *_jci_inputs(context))["JCI"]
    print("\n=== SANITY CHECK: no vendor series, S unchanged ===")
    print(pd.DataFrame({"AAA_raw": raw, "AAA_cube": cube, "AAA_validate": check, "JCI_no_override": plain, "JCI_raw": jci_raw}).round(6).to_string())
    assert cube.equals(raw) and check.equals(raw)
    assert plain.equals(jci_raw)
    print("  OK: an ordinary ticker's S is bit-identical to its own actions' S, and JCI without the override keeps its own actions.")


def _jci_inputs(context: SimpleNamespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = context.store.load(Tables.sharadar_actions, where={"ticker": ["JCI"]})
    yf = context.store.load(Tables.prices_splits, where={"ticker": ["JCI"]})
    return yf, genuine_splits(actions, yf)
