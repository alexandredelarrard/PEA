"""Capital inputs and transaction-date market-cap evidence for insider trade signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_trade_capital(
    df_trades: pd.DataFrame,
    df_market_cap: pd.DataFrame | None,
    trading_index: pd.DatetimeIndex,
    share_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach action-date ratios and lineage; future actions stay unsupported for their public version."""
    df_result = df_trades.copy()
    calendar = trading_index.normalize().unique().sort_values().as_unit("ns")
    action = pd.to_datetime(df_result["transaction_day"], errors="coerce").dt.normalize().dt.as_unit("ns")
    public_day = pd.to_datetime(df_result["visible_from"], errors="coerce").dt.normalize().dt.as_unit("ns")
    df_result["cap_session"] = pd.Series(pd.NaT, index=df_result.index, dtype="datetime64[ns]")
    df_result["market_cap_on_action"] = np.nan
    df_result["cap_status"] = "missing_action_date"

    # Session identity comes from the authoritative calendar, including ticker price holes.
    if not calendar.empty:
        positions = calendar.searchsorted(pd.DatetimeIndex(action), side="right") - 1
        has_prior = action.notna().to_numpy() & (positions >= 0)
        selected = pd.Series(calendar.take(np.clip(positions, 0, len(calendar) - 1)), index=df_result.index)
        recent = (action - selected).le(pd.Timedelta(days=7)).to_numpy()
        already_public = action.le(public_day) & selected.le(public_day)
        permitted = has_prior & (action.isin(calendar).to_numpy() | recent) & already_public.to_numpy()
        df_result.loc[action.notna(), "cap_status"] = "no_previous_session"
        df_result.loc[has_prior, "cap_status"] = "stale_non_session"
        df_result.loc[action.notna() & public_day.isna(), "cap_status"] = "missing_public_date"
        df_result.loc[action.gt(public_day), "cap_status"] = "future_action_date"
        df_result.loc[permitted, "cap_session"] = selected.loc[permitted]
        df_result.loc[permitted, "cap_status"] = "unsupported_cap"

        # An invalid exact/immediate-prior cell never searches farther back or forward.
        if df_market_cap is not None and not df_market_cap.empty:
            columns = df_market_cap.columns.get_indexer(pd.Index(df_result["ticker"]))
            supported = permitted & (columns >= 0)
            values = df_market_cap.reindex(index=calendar).to_numpy(dtype=float)
            caps = np.full(len(df_result), np.nan)
            caps[supported] = values[positions[supported], columns[supported]]
            positive = np.isfinite(caps) & (caps > 0)
            df_result["market_cap_on_action"] = np.where(positive, caps, np.nan)
            df_result.loc[positive, "cap_status"] = "supported"

    # Preserve execution dollars for all non-cap consumers and reject invalid clean amounts.
    value = pd.to_numeric(df_result["value"], errors="coerce")
    df_result.loc[df_result["cap_status"].eq("supported") & ~np.isfinite(value), "cap_status"] = "unsupported_execution"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = value / df_result["market_cap_on_action"]
    df_result.loc[df_result["cap_status"].eq("supported") & ~np.isfinite(ratio), "cap_status"] = "unsupported_ratio"
    df_result["value_mcap"] = ratio.where(df_result["cap_status"].eq("supported"))
    df_result["cap_price_date"] = df_result["cap_session"].where(df_result["market_cap_on_action"].notna())
    df_result["cap_share_as_of"] = _share_observation_dates(df_result, share_history)
    return df_result


def _share_observation_dates(df_trades: pd.DataFrame, share_history: pd.DataFrame | None) -> pd.Series:
    """Trace the last nonnull vendor observation public by the selected session."""
    result = pd.Series(pd.NaT, index=df_trades.index, dtype="datetime64[ns]")
    if share_history is None or not {"ticker", "as_of", "sharesOutstanding"}.issubset(share_history):
        return result

    # Match the common projection's null and same-publication deduplication policy.
    df_history = share_history[["ticker", "as_of", "sharesOutstanding"]].copy()
    df_history["as_of"] = pd.to_datetime(df_history["as_of"], errors="coerce").dt.as_unit("ns")
    df_history = df_history.dropna().sort_values(["ticker", "as_of"]).drop_duplicates(["ticker", "as_of"], keep="last")
    df_lookup = df_trades[["ticker", "cap_session"]].assign(_row=np.arange(len(df_trades))).dropna(subset=["cap_session"])
    if df_history.empty or df_lookup.empty:
        return result

    df_joined = pd.merge_asof(
        df_lookup.sort_values("cap_session"),
        df_history.rename(columns={"as_of": "cap_share_as_of"}).sort_values("cap_share_as_of"),
        left_on="cap_session",
        right_on="cap_share_as_of",
        by="ticker",
        direction="backward",
    )
    result.iloc[df_joined["_row"].to_numpy(dtype=int)] = df_joined["cap_share_as_of"].to_numpy()
    return result
