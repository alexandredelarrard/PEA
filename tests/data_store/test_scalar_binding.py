"""Generic dataframe scalar binding keeps Python values and SQL NULLs."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from src.data_store import store as store_module


def test_records_match_the_to_dict_binding() -> None:
    frame = pd.DataFrame(
        {
            "i": np.array([1, 2, 3], dtype="int64"),
            "f": [1.5, np.nan, 3.0],
            "I": pd.array([1, None, 3], dtype="Int64"),
            "B": pd.array([True, None, False], dtype="boolean"),
            "s": pd.Series(["a", None, "c"], dtype="str"),
            "o": pd.Series([np.int64(4), None, "x"], dtype=object),
            "t": pd.to_datetime(pd.Series(["2024-01-01", None, "2024-03-01"], dtype=object)),
            "d": pd.Series([dt.date(2024, 1, 1), None, pd.NaT], dtype=object),
            "ci": pd.Categorical([1, None, 2]),
            "l": pd.Series([[1, 2], None, (3,)], dtype=object),
        }
    )
    expected = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    got = store_module._records(frame)
    pairs = [(a[k], b[k]) for a, b in zip(expected, got, strict=True) for k in frame.columns]
    assert [list(r) for r in got] == [list(r) for r in expected]
    assert all(type(a) is type(b) and (a is None or (isinstance(a, float) and math.isnan(a)) or a == b) for a, b in pairs)
    assert store_module._records(pd.DataFrame(index=range(2))) == []
    print(f"SANITY: _records binds the same Python type and value as to_dict on {len(pairs)} cells over {frame.shape[1]} dtypes; NaN/NaT/NA -> None.")
