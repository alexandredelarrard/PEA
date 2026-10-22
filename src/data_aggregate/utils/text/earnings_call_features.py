"""Cached earnings-call metrics, issuer identity and point-in-time feature mathematics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import EARNINGS_CALL_FEATURES, EARNINGS_CALL_SIGNAL_SESSIONS
from src.context import Context
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.text.earnings_call_preparation import quarter_number
from src.data_store.schema import Tables
from src.utils.string import pad_cik_series
from src.utils.text.earnings_call_split import CALL_SECTIONS

SENTIMENT_COLUMNS = ("ticker", "quarter", "tag", "as_of", "sent_pos", "sent_neg", "sent_neu", "n_words", "uncertainty_ratio", "model")


def valid_sentiment_rows(sentiment: pd.DataFrame, cache_model: str) -> pd.DataFrame:
    """Validate existing scores and explicit unavailable-section rows without source reads."""
    if sentiment.empty:
        return sentiment.copy()
    out = sentiment[sentiment["model"].eq(cache_model) & sentiment["tag"].isin(CALL_SECTIONS)].copy()
    numeric = ["sent_pos", "sent_neg", "sent_neu", "n_words", "uncertainty_ratio"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    probabilities = out[["sent_pos", "sent_neg", "sent_neu"]]
    words = out["n_words"]
    present = words.gt(0) & words.mod(1).eq(0)
    distribution = probabilities.ge(0).all(axis=1) & probabilities.le(1).all(axis=1)
    distribution &= probabilities.sum(axis=1).sub(1).abs().le(2e-6)
    uncertainty = out["uncertainty_ratio"].between(0, 1)
    absent = words.eq(0) & probabilities.isna().all(axis=1) & out["uncertainty_ratio"].isna()
    keep = out["as_of"].notna() & ((present & distribution & uncertainty) | absent)
    keep &= ~out.duplicated(["ticker", "quarter", "tag"], keep=False)
    return out.loc[keep].copy()


def completed_sentiment_calls(sentiment: pd.DataFrame) -> set[tuple[str, str]]:
    """Both canonical tags, one date and at least 100 actual words establish completion."""
    if sentiment.empty:
        return set()
    groups = sentiment.groupby(["ticker", "quarter"], sort=False)
    complete = groups["tag"].nunique().eq(len(CALL_SECTIONS)) & groups["n_words"].sum().ge(100) & groups["as_of"].nunique().eq(1)
    return set(complete[complete].index)


def sentiment_kpis(sentiment: pd.DataFrame, cache_model: str) -> pd.DataFrame:
    """Actual-word-weighted metrics and coverage from completed saved calls."""
    s = valid_sentiment_rows(sentiment, cache_model)
    complete = completed_sentiment_calls(s)
    if not complete:
        return pd.DataFrame(columns=["ticker", "quarter", "as_of"])
    s = s[pd.MultiIndex.from_frame(s[["ticker", "quarter"]]).isin(complete)].copy()
    s["net"] = s["sent_pos"] - s["sent_neg"]
    # Recover integer word hits from the six-decimal cache when the count is unambiguous.
    recoverable = s["n_words"].gt(0) & s["n_words"].lt(1_000_000)
    s.loc[recoverable, "uncertainty_ratio"] = (
        np.rint(s.loc[recoverable, "uncertainty_ratio"] * s.loc[recoverable, "n_words"]) / s.loc[recoverable, "n_words"]
    )
    keys = ["ticker", "quarter"]
    tone = s.pivot(index=keys, columns="tag", values="net").reindex(columns=CALL_SECTIONS)
    words = s.pivot(index=keys, columns="tag", values="n_words").reindex(columns=CALL_SECTIONS)
    uncertainty = s.pivot(index=keys, columns="tag", values="uncertainty_ratio").reindex(columns=CALL_SECTIONS)
    total = words.sum(axis=1)
    coverage = words.gt(0).dot(np.asarray([1, 2]))
    prep, qa = CALL_SECTIONS
    result = pd.DataFrame(
        {
            "as_of": s.groupby(keys)["as_of"].first(),
            "ec_tone": (tone * words).sum(axis=1) / total,
            "ec_qa_gap": tone[qa] - tone[prep],
            "ec_uncertainty": (uncertainty * words).sum(axis=1) / total,
            "total_words": total,
            "coverage": coverage,
        }
    ).reset_index()
    return result.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def _issuer_history_zscore(per_call: pd.DataFrame, value_col: str) -> pd.Series:
    """Five-year, prior-only issuer z-score; at least four prior calls, sample std."""
    out = pd.Series(np.nan, index=per_call.index, dtype="float64")
    identity = "issuer_id" if "issuer_id" in per_call.columns else "ticker"
    group_keys = [identity, "coverage"] if value_col in ("ec_tone", "ec_uncertainty") and "coverage" in per_call else [identity]
    for _, group in per_call.groupby(group_keys, sort=False):
        ordered = group.sort_values("as_of")
        dates = pd.to_datetime(ordered["as_of"], errors="coerce")
        values = pd.to_numeric(ordered[value_col], errors="coerce")
        for position, row_index in enumerate(ordered.index):
            current_date = dates.iloc[position]
            if pd.isna(current_date):
                continue
            prior = values.iloc[:position]
            prior_dates = dates.iloc[:position]
            prior = prior[(prior_dates >= current_date - pd.DateOffset(years=5)).to_numpy()].dropna()
            if len(prior) < 4:
                continue
            std = prior.std(ddof=1)
            if pd.notna(values.iloc[position]) and std > 0:
                out.loc[row_index] = (values.iloc[position] - prior.mean()) / std
    return out


def load_issuer_identity(context: Context) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Canonical roster, consolidating lineage and trusted symbol ownership for stored calls."""
    roster = context.store.load(Tables.sp500_tickers, columns=["ticker", "cik"], optional=True)
    lineage = context.store.load(Tables.entity_lineage, columns=["cik", "entity_id", "role", "valid_from", "valid_to"], optional=True)
    tenure = context.store.load(
        Tables.symbol_tenure, columns=["symbol", "issuer_cik", "valid_from", "valid_to"], where={"source": ["form345", "manual"]}, optional=True
    )
    return roster, None if lineage is None else lineage.drop_duplicates(ignore_index=True), tenure


def _dated_call_rows(calls: pd.DataFrame, intervals: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
    """Match call dates to half-open issuer or symbol intervals without changing source dates."""
    matched = calls.merge(intervals, left_on=left_on, right_on=right_on, how="inner")
    dates = pd.to_datetime(matched["as_of"], errors="coerce")
    start = pd.to_datetime(matched["valid_from"], errors="coerce")
    end = pd.to_datetime(matched["valid_to"], errors="coerce")
    return matched[dates.ge(start) & (end.isna() | dates.lt(end))]


def attach_issuer_identity(
    per_call: pd.DataFrame,
    roster: pd.DataFrame | None,
    entity_lineage: pd.DataFrame | None,
    symbol_tenure: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach canonical issuer history, excluding unsupported dates of reused issuer aliases."""
    out = per_call.copy()
    out["issuer_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if roster is None or roster.empty or entity_lineage is None or entity_lineage.empty or symbol_tenure is None or symbol_tenure.empty:
        return out.iloc[0:0]
    lineage = entity_lineage.assign(cik=pad_cik_series(entity_lineage["cik"]))[["cik", "entity_id"]].drop_duplicates()
    memberships = lineage.groupby("cik")["entity_id"].agg(["size", "count", "nunique", "first"])
    valid = memberships[(memberships["size"] == memberships["count"]) & memberships["nunique"].eq(1)]
    current = roster.assign(cik=pad_cik_series(roster["cik"]))[["ticker", "cik"]].drop_duplicates()
    current["entity_id"] = current["cik"].map(valid["first"])
    owners = current.groupby("ticker")["entity_id"].agg(["size", "count", "nunique", "first"])
    owners = owners[(owners["size"] == owners["count"]) & owners["nunique"].eq(1)]
    out["issuer_id"] = out["ticker"].map(owners["first"]).astype("string")
    windows = entity_lineage[entity_lineage["role"].eq("cik_window")].copy()
    windows["cik"] = pad_cik_series(windows["cik"])
    calls = out[["issuer_id", "as_of"]].assign(call_position=np.arange(len(out)))
    admitted = _dated_call_rows(calls, windows[["entity_id", "valid_from", "valid_to"]], "issuer_id", "entity_id")["call_position"]
    keep = pd.Series(np.arange(len(out))).isin(admitted) & out["issuer_id"].reset_index(drop=True).notna()
    tenure = symbol_tenure.assign(issuer_cik=pad_cik_series(symbol_tenure["issuer_cik"]))
    window_owners = windows[windows["entity_id"].eq(windows["cik"].map(valid["first"]))][["cik", "entity_id"]].drop_duplicates()
    tenure = tenure.merge(window_owners, left_on="issuer_cik", right_on="cik", how="left")
    aliases = tenure.dropna(subset=["entity_id"])[["entity_id", "symbol"]].drop_duplicates()
    keep &= out["issuer_id"].reset_index(drop=True).isin(aliases["entity_id"])
    ownership = aliases.merge(tenure[["symbol", "entity_id"]], on="symbol", suffixes=("", "_owner"))
    contested = ownership.loc[ownership["entity_id_owner"].isna() | ownership["entity_id"].ne(ownership["entity_id_owner"]), "entity_id"]
    supported = _dated_call_rows(calls, tenure, "issuer_id", "entity_id")[["call_position", "issuer_id", "as_of", "symbol", "cik"]]
    supported = _dated_call_rows(supported, windows[["cik", "valid_from", "valid_to"]], "cik", "cik")
    dated_owners = _dated_call_rows(supported[["call_position", "issuer_id", "as_of", "symbol"]].drop_duplicates(), tenure, "symbol", "symbol")
    ambiguous = dated_owners.loc[
        dated_owners["entity_id"].isna() | dated_owners["issuer_id"].ne(dated_owners["entity_id"]), ["call_position", "symbol"]
    ]
    supported = supported.merge(ambiguous.drop_duplicates(), on=["call_position", "symbol"], how="left", indicator=True)
    supported = supported.loc[supported["_merge"].eq("left_only"), "call_position"]
    keep &= ~out["issuer_id"].reset_index(drop=True).isin(contested) | pd.Series(np.arange(len(out))).isin(supported)
    return out.iloc[np.flatnonzero(keep)].copy()


def _daily_frame(
    per_call: pd.DataFrame, value_col: str, idx: pd.DatetimeIndex, availability_lag_days: int, since: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Emit only output sessions, preserving each call's original 66-session interval."""
    calendar = pd.DatetimeIndex(idx).normalize()
    first = 0 if since is None else int(calendar.searchsorted(since))
    output = calendar[first:]
    if per_call.empty or output.empty:
        return pd.DataFrame(index=output)
    tickers = per_call["ticker"].astype(str).drop_duplicates().tolist()
    frame = pd.DataFrame(np.nan, index=output, columns=tickers, dtype="float64")
    ordered = per_call.assign(as_of=pd.to_datetime(per_call["as_of"], errors="coerce")).sort_values(["ticker", "as_of"])
    for row in ordered.itertuples(index=False):
        if pd.isna(row.as_of):
            continue
        eligible = cast(pd.Timestamp, row.as_of).normalize() + pd.Timedelta(days=availability_lag_days)
        start = int(calendar.searchsorted(eligible, side="left"))
        stop = min(start + EARNINGS_CALL_SIGNAL_SESSIONS, len(calendar))
        if stop > first and start < len(calendar):
            frame.loc[calendar[max(start, first) : stop], str(row.ticker)] = getattr(row, value_col)
    return frame


def feature_history_scope(
    calls: pd.DataFrame, calendar: pd.DatetimeIndex, since: pd.Timestamp | None, lag_days: int
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Select active emitters, five-year observations and actual fiscal predecessors."""
    if calls.empty:
        return calls.copy(), set()
    ordered = calls.dropna(subset=["as_of"]).sort_values(["issuer_id", "as_of"]).reset_index(drop=True)
    first = 0 if since is None else int(calendar.searchsorted(since))
    eligible = pd.to_datetime(ordered["as_of"]).dt.normalize() + pd.Timedelta(days=lag_days)
    starts = calendar.searchsorted(eligible)
    active = (starts < len(calendar)) & (starts + EARNINGS_CALL_SIGNAL_SESSIONS > first)
    emitters = ordered.loc[active]
    if emitters.empty:
        return ordered.iloc[0:0], set()
    history_from = pd.Timestamp(emitters["as_of"].min()) - pd.DateOffset(years=5)
    ordered["_position"] = np.arange(len(ordered))
    grouped = ordered.groupby("issuer_id", sort=False)
    previous = grouped["_position"].shift()
    consecutive = ordered["quarter"].map(quarter_number).sub(grouped["quarter"].shift().map(quarter_number)).eq(1)
    predecessor_positions = previous[active & consecutive].dropna().astype(int)
    comparisons = pd.concat([emitters[["ticker", "quarter"]], ordered.loc[predecessor_positions, ["ticker", "quarter"]]]).drop_duplicates()
    keep = pd.to_datetime(ordered["as_of"]).between(history_from, calendar[-1]) | ordered["_position"].isin(predecessor_positions)
    return ordered.loc[keep].drop(columns="_position"), set(map(tuple, comparisons.to_numpy()))


def build_earnings_call_feature_panel(
    per_call: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    availability: pd.DataFrame | None = None,
    *,
    availability_lag_days: int,
    since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the exact 12-column raw/issuer-history daily features from computed per-call KPIs."""
    if per_call.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    per_call = prepare_earnings_call_kpis(per_call)
    return _feature_panel_from_prepared(per_call, trading_index, availability, availability_lag_days, since)


def prepare_earnings_call_kpis(per_call: pd.DataFrame) -> pd.DataFrame:
    """Add the exact raw/history KPI contract on the per-call grain."""
    history_bases = ["ec_tone", "ec_qa_gap", "ec_uncertainty", "ec_qa_coherence_mean"]
    raw_kpis = [
        "ec_tone",
        "ec_qa_gap",
        "ec_uncertainty",
        "ec_qa_coherence_mean",
        "ec_tone_delta",
        "ec_length_delta",
        "ec_qa_qq_distance",
        "ec_prep_qq_distance",
    ]
    per_call = per_call.copy()
    for col in raw_kpis:
        if col not in per_call.columns:
            per_call[col] = np.nan
    for col in history_bases:
        per_call[f"{col}_vs_hist"] = _issuer_history_zscore(per_call, col)
    identity = "issuer_id" if "issuer_id" in per_call.columns else "ticker"
    if "total_words" not in per_call.columns:
        return per_call
    ordered = per_call.sort_values([identity, "as_of"])
    grouped = ordered.groupby(identity, sort=False)
    previous_quarter = grouped["quarter"].shift(1)
    consecutive = (ordered["quarter"].map(quarter_number) - previous_quarter.map(quarter_number)) == 1
    matching = consecutive & ordered["coverage"].eq(grouped["coverage"].shift()) if "coverage" in ordered else consecutive
    per_call.loc[ordered.index, "ec_tone_delta"] = grouped["ec_tone"].diff().where(matching)
    previous_words = grouped["total_words"].shift(1)
    length_delta = cast(pd.Series, np.log(ordered["total_words"] / previous_words.where(previous_words > 0))).where(matching)
    per_call.loc[ordered.index, "ec_length_delta"] = length_delta.replace([np.inf, -np.inf], np.nan)
    for column in ("ec_qa_qq_distance", "ec_prep_qq_distance"):
        per_call.loc[ordered.index, column] = ordered[column].where(consecutive)
    return per_call


def _feature_panel_from_prepared(
    per_call: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    availability: pd.DataFrame | None,
    availability_lag_days: int,
    since: pd.Timestamp | None,
) -> pd.DataFrame:
    fields: dict[str, pd.DataFrame] = {}
    for col in EARNINGS_CALL_FEATURES:
        if col not in per_call.columns:
            continue
        sub = per_call[["as_of", "ticker", col]]
        if sub[col].notna().sum() == 0:
            continue
        frame = _daily_frame(sub, col, trading_index, availability_lag_days, since)
        if frame is not None and not frame.empty and frame.notna().any().any():
            fields[col] = frame
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    panel = build_peer_relative_panel(fields, {}, emission={name: "raw" for name in fields}, availability=availability)
    for name in EARNINGS_CALL_FEATURES:
        column = f"f_{name}"
        if column not in panel.columns:
            panel[column] = pd.Series(np.nan, index=panel.index, dtype="float32")
    return panel[["date", "ticker", *[f"f_{name}" for name in EARNINGS_CALL_FEATURES]]]
