"""The projected modelling loader (src/modelling/utils/panel.py) on a real SQLite DataStore:
projection, labelled-row push-down, date scope, float32 only on request, None when absent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_store.schema import Tables
from src.modelling.utils.panel import load_frame


def _seed(store) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=4)
    cube = pd.DataFrame(
        {
            "date": np.repeat(dates, 3),
            "ticker": ["AAA", "BBB", "CCC"] * 4,
            "f_a": np.arange(12, dtype="float64") / 7.0,
            "f_unused": 1.0,
            "target_rank_h30": [0.1, 0.5, 0.9] * 3 + [np.nan] * 3,  # newest date: label immature
        }
    )
    store.replace(Tables.cube, cube)
    return cube


def test_load_frame_projects_scopes_and_downcasts(sqlite_store) -> None:
    cube = _seed(sqlite_store)
    train = load_frame(
        sqlite_store, Tables.cube, columns=["date", "ticker", "f_a", "target_rank_h30"], where={"target_rank_h30": sqlite_store.NOT_NULL}
    )
    assert train is not None
    assert list(train.columns) == ["date", "ticker", "f_a", "target_rank_h30"], "only the projection is read"
    assert len(train) == 9, "unlabelled rows are filtered in SQL"
    assert str(train["f_a"].dtype) == "float32" and str(train["target_rank_h30"].dtype) == "float32"

    latest = load_frame(sqlite_store, Tables.cube, columns=["date", "ticker", "f_a"], since=cube["date"].max(), downcast=False)
    assert latest is not None and len(latest) == 3 and str(latest["f_a"].dtype) == "float64", "prediction reads stay float64"
    assert np.array_equal(latest["f_a"].to_numpy(), cube["f_a"].to_numpy()[-3:])
    print("\n=== SANITY CHECK: load_frame ===")
    print(f"  train read: {len(train)} labelled rows, cols {list(train.columns)}, f_a {train['f_a'].dtype}")
    print(f"  predict read: newest date only ({len(latest)} rows, label never loaded), f_a {latest['f_a'].dtype} bit-exact. Validated.")


def test_load_frame_absent_table_is_none_when_optional(sqlite_store) -> None:
    assert load_frame(sqlite_store, Tables.cube, columns=["date", "ticker"]) is None
    print("\n=== SANITY CHECK: absent table ===")
    print("  optional read of a missing cube -> None (the caller decides how loud to be). Validated.")


def test_streamed_preparation_preserves_store_values_and_precision(sqlite_store) -> None:
    cube = _seed(sqlite_store)
    cube.loc[0, "f_a"] = np.nan
    cube["f_a"] = cube["f_a"].astype("str")
    sqlite_store.replace(Tables.cube, cube)
    loaded = load_frame(sqlite_store, Tables.cube, columns=["date", "ticker", "f_a"], feature_cols=["f_a"], stream=True, chunksize=2)
    precise = load_frame(sqlite_store, Tables.cube, columns=["date", "ticker", "f_a"], feature_cols=["f_a"], stream=True, downcast=False)
    assert loaded is not None and precise is not None
    stored = sqlite_store.load(Tables.cube, columns=["date", "ticker", "f_a"])
    assert stored is not None
    expected = pd.to_numeric(stored["f_a"], errors="raise").to_numpy(dtype=float)
    assert loaded["f_a"].dtype == np.dtype("float32") and precise["f_a"].dtype == np.dtype("float64")
    assert np.array_equal(loaded["f_a"], expected.astype("float32"), equal_nan=True)
    assert np.array_equal(precise["f_a"], expected, equal_nan=True)
    print("SANITY: streamed preparation preserves actual stored values and NaNs, with float64 retained when requested.")
