"""Dated `security_master` class lines: one reader, one normalisation and one dated CUSIP join."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.string import yahoo_symbol

CLASS_LINE_COLUMNS = ("canonical_company", "market_symbol", "conversion_ratio", "valid_from", "valid_to")
_COLUMNS = (*CLASS_LINE_COLUMNS, "security_id", "source", "source_symbol", "security_class", "lineage_role", "cusip")
_OPEN_START = pd.Timestamp("1900-01-01")


def load_class_lines(store: DataStore) -> pd.DataFrame | None:
    """The adjudicated `ftd` master lines, one row per dated security line."""
    return store.load(Tables.security_master, columns=_COLUMNS, where={"source": "ftd"}, optional=True)


def normalised_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Lines with `company`, Yahoo `symbol`, numeric `ratio` and half-open `[start, end)` days, oldest start first."""
    return lines.assign(
        company=lines["canonical_company"].astype(str),
        symbol=lines["market_symbol"].map(yahoo_symbol),
        ratio=pd.to_numeric(lines["conversion_ratio"], errors="coerce"),
        start=pd.to_datetime(lines["valid_from"]).dt.normalize(),
        end=pd.to_datetime(lines["valid_to"]).dt.normalize(),
    ).sort_values("start", kind="stable")


def cusip_windows(lines: pd.DataFrame | None, cusip_map: pd.DataFrame | None) -> pd.DataFrame:
    """Dated CUSIP -> ticker windows: the master's canonical lines (secondary classes only at ratio 1, so BRK-A
    stays apart) override `cusip_map`, which is an open window used before or without a master line."""
    frames = []
    if cusip_map is not None and not cusip_map.empty:
        df_map = cusip_map.dropna(subset=["cusip", "ticker"]).drop_duplicates("cusip")
        frames.append(pd.DataFrame({"cusip": df_map["cusip"], "ticker": df_map["ticker"], "start": _OPEN_START, "end": pd.NaT}))
    if lines is not None and not lines.empty:
        work = normalised_lines(lines.dropna(subset=["cusip", "valid_from"]))
        canonical = work["lineage_role"].isin(CANONICAL_ROLES) | (work["lineage_role"].eq(SECONDARY_CLASS) & work["ratio"].eq(1.0))
        frames.append(work.loc[canonical, ["cusip", "company", "start", "end"]].rename(columns={"company": "ticker"}))
    if not frames:
        return pd.DataFrame(columns=["cusip", "ticker", "start", "end"])
    df_windows = pd.concat(frames, ignore_index=True).astype({"start": "datetime64[ns]", "end": "datetime64[ns]"})
    return df_windows.sort_values("start", kind="stable").reset_index(drop=True)


def dated_tickers(cusips: pd.Series, dates: pd.Series, windows: pd.DataFrame) -> pd.Series:
    """Per (CUSIP, date), the ticker of the latest window started at or before the date; null once it has ended."""
    out = pd.Series(np.nan, index=cusips.index, dtype=windows["ticker"].dtype)
    keys = pd.DataFrame({"cusip": cusips.to_numpy(), "date": pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]"), "row": np.arange(len(cusips))})
    keys = keys.dropna(subset=["cusip", "date"]).sort_values("date", kind="stable")
    if keys.empty or windows.empty:
        return out
    hits = pd.merge_asof(keys, windows, left_on="date", right_on="start", by="cusip")
    active = hits["ticker"].notna() & (hits["end"].isna() | hits["date"].lt(hits["end"]))
    out.iloc[hits.loc[active, "row"].to_numpy()] = hits.loc[active, "ticker"].to_numpy()
    return out
