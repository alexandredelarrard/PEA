"""`run_predict` refuses to score on stale inputs (AC-017).

A known-truth fixture on a real SQLite store: 20 universe tickers whose `prices`,
`fundamentals_sharadar` and `fundamentals_history` are fresh as of today, a cube ending on the
last price date, and a stub one-member ensemble so no model is trained. Each case makes one
input stale and checks whether prediction is blocked and `predictions_latest` left unwritten."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_store.schema import Table, Tables
from src.modelling.steps import step_long_short
from src.modelling.steps.step_long_short import StepLongShort
from src.utils.freshness import PREDICTION_INPUTS, StaleInputsError
from tests.modelling.model_fixtures import make_config

TICKERS = [f"T{i:02d}" for i in range(20)]
TODAY = pd.Timestamp.today().normalize()
LAST_PRICE = TODAY - pd.Timedelta(days=1)


class _Member:
    """A stub ensemble member scoring each row by its `f0`."""

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return panel["f0"].to_numpy()


def _seed(store: Any) -> None:
    """Every universe key fresh, and the cube ending on the last price date."""
    store.replace(Tables.sp500_tickers, pd.DataFrame({"ticker": TICKERS}))
    store.replace(Tables.prices, pd.DataFrame({"ticker": TICKERS, "date": LAST_PRICE, "close_split": 10.0}))
    store.replace(
        Tables.sharadar_fundamentals,
        pd.DataFrame({"ticker": TICKERS, "dimension": "ARQ", "date": TODAY - pd.Timedelta(days=30), "reportperiod": TODAY - pd.Timedelta(days=60)}),
    )
    store.replace(Tables.fundamentals_history, pd.DataFrame({"ticker": TICKERS, "as_of": TODAY - pd.Timedelta(days=30), "totalRevenue": 1.0}))
    store.replace(Tables.cube, pd.DataFrame({"date": LAST_PRICE, "ticker": TICKERS, "f0": np.linspace(-1.0, 1.0, len(TICKERS))}))


def _age(store: Any, table: Table, tickers: list[str], days: int) -> None:
    """Move `tickers`' rows in `table` to `days` days before today."""
    frame = store.load(table)
    frame.loc[frame["ticker"].isin(tickers), table.freshness_col] = TODAY - pd.Timedelta(days=days)
    store.replace(table, frame)


def _step(tmp_path: Path, store: Any, monkeypatch: pytest.MonkeyPatch, min_share: float = 0.95) -> StepLongShort:
    cfg = make_config(
        model={"target_type": "rank", "models_dir": str(tmp_path / "models")},
        build_cube={"output": {}},
        data_extract={"prediction_fresh_share": min_share, "redundant_ticks": []},
    )
    meta = {"horizons": [30], "feature_cols": ["f0"], "categorical_cols": [], "model_types": ["stub"], "train_ic_ir": {"30": 1.0}}
    monkeypatch.setattr(step_long_short, "read_metadata", lambda directory: meta)
    monkeypatch.setattr(step_long_short, "load_ensemble", lambda directory, horizons, families: {30: {"stub": _Member()}})
    paths = {"DATA_STORE": tmp_path / "data", "OUTPUT_DIR": tmp_path / "out", "MODELS_DIR": tmp_path / "models"}
    context: Any = SimpleNamespace(
        store=store, save=False, log=logging.getLogger("predict-freshness"), config=cfg, config_dir=Path("configs"), paths=paths
    )
    return StepLongShort(context=context, config=cfg)


def test_stale_prices_block_prediction(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(sqlite_store)
    _age(sqlite_store, Tables.prices, TICKERS[:2], days=10)  # 18 of 20 fresh = 0.90
    step = _step(tmp_path, sqlite_store, monkeypatch)
    with pytest.raises(StaleInputsError, match=r"prices fresh share 0\.900"):
        step.run_predict()
    assert not sqlite_store.exists(Tables.predictions_latest), "a blocked run writes nothing"

    relaxed = _step(tmp_path, sqlite_store, monkeypatch, min_share=0.85)
    out = relaxed.run_predict()
    assert out["ticker"].nunique() == len(TICKERS)

    print("\n=== SANITY CHECK: stale prices block prediction ===")
    print("  2 of 20 tickers' last price 10 days old -> share 0.90 < 0.95 -> StaleInputsError, nothing written")
    print(f"  the same store with prediction_fresh_share 0.85 scores {out['ticker'].nunique()} tickers. Validated.")


@pytest.mark.parametrize("table", PREDICTION_INPUTS, ids=[t.name for t in PREDICTION_INPUTS])
def test_each_prediction_input_blocks_when_stale_per_key(table: Table, tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(sqlite_store)
    _age(sqlite_store, table, TICKERS[:3], days=400)  # 17 of 20 fresh = 0.85; the table-wide max stays fresh
    with pytest.raises(StaleInputsError) as raised:
        _step(tmp_path, sqlite_store, monkeypatch).run_predict()
    assert f"{table.name} fresh share 0.850" in str(raised.value)
    assert not sqlite_store.exists(Tables.predictions_latest)

    print(f"\n=== SANITY CHECK: {table.name} stale per key ===")
    print(f"  3 of 20 keys 400 days old while the table-wide max is fresh -> blocked: {raised.value}. Validated.")


def test_stale_targets_13f_and_insider_do_not_block(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(sqlite_store)
    old = TODAY - pd.Timedelta(days=400)
    sqlite_store.replace(Tables.cube_part_targets, pd.DataFrame({"date": old, "ticker": TICKERS, "target_rank_h30": 0.5}))
    sqlite_store.replace(Tables.sec13f_hr, pd.DataFrame({"cik": "0000000001", "period": old, "ticker": TICKERS, "cusip": "X"}))
    sqlite_store.replace(
        Tables.insider_transactions,
        pd.DataFrame(
            {"accession_number": TICKERS, "security_type": "C", "row_sequence": 1, "ticker": TICKERS, "transaction_date": old, "filing_date": old}
        ),
    )
    out = _step(tmp_path, sqlite_store, monkeypatch).run_predict()
    assert out["date"].max() == LAST_PRICE and out["ticker"].nunique() == len(TICKERS)
    assert len(sqlite_store.load(Tables.predictions_latest)) == len(out)

    print("\n=== SANITY CHECK: only the three inputs gate prediction ===")
    print(f"  targets, sec13f_hr and insider_transactions 400 days old -> {len(out)} rows scored for {LAST_PRICE.date()}. Validated.")


def test_cube_lagging_prices_blocks_prediction(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(sqlite_store)
    lagging = LAST_PRICE - pd.Timedelta(days=2)
    sqlite_store.replace(Tables.cube, pd.DataFrame({"date": lagging, "ticker": TICKERS, "f0": np.linspace(-1.0, 1.0, len(TICKERS))}))
    with pytest.raises(StaleInputsError, match=f"cube ends {lagging.date()} before prices"):
        _step(tmp_path, sqlite_store, monkeypatch).run_predict()
    assert not sqlite_store.exists(Tables.predictions_latest)

    print("\n=== SANITY CHECK: cube behind prices ===")
    print(f"  every input fresh, cube ends {lagging.date()} vs prices {LAST_PRICE.date()} -> blocked. Validated.")
