"""
earnings_call_features.py  (src/data_aggregate/utils/earnings_call_features.py)
-------------------------------------------------------------------------------
Turn the raw earnings-call paragraphs (`earnings_call_sections`, one row per ticker·quarter·
paragraph) into point-in-time raw and issuer-history features. Each call is cut into its
prepared_remarks / qa texts by `src/utils/earnings_call_split.split_call`; a call whose split
status is not `ok`, or whose texts fail `assess_earnings_call_sections`, yields no features.

Two stages:

  1. score_earnings_calls(context)  — the EXPENSIVE, cached, incremental NLP pass.
     For every not-yet-scored (ticker, quarter, tag) it runs the local FinBERT-tone
     model (GPU if available) to get {pos, neg, neu} tone probabilities, plus cheap
     text metrics (word count, Loughran-McDonald uncertainty ratio), and upserts them
     to the `earnings_call_sentiment` cache — PER TICKER, so an interrupted run never
     loses GPU work. Skipped cleanly if torch/transformers are unavailable.

  2. build_earnings_call_feature_panel(...) — the cheap per-build derivation. From the
     cached per-call scores it builds raw and prior-only issuer-history KPIs, aligns
     them to the daily trading calendar after the configured calendar-day lag, and expires each call
     after 66 trading sessions. Missing calls stay null; genuine zero signals stay zero.

Smart KPIs (eligible on the first session on/after the buffered call date):
    ec_tone            length-weighted call tone  P(pos) − P(neg)      (level)
    ec_tone_delta      Δ tone vs the prior consecutive quarter's call  (tone momentum)
    ec_qa_gap          Q&A tone − prepared-remarks tone   (candor: scripted optimism
                       far above unscripted answers is a bearish tell)
    ec_uncertainty     length-weighted hedging ratio (LM uncertainty words)
    ec_length_delta    log(total words this call / prior consecutive   (disclosure Δ)
                       quarter's call); both deltas are null when that quarter has no call
    ec_qa_coherence_mean   mean question/answer embedding cosine
    ec_qa_qq_distance      1 − cosine(Q&A embedding vs prior consecutive quarter)
    ec_prep_qq_distance    1 − cosine(prepared embedding vs prior consecutive quarter)
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import EARNINGS_CALL_FEATURES, EARNINGS_CALL_SIGNAL_SESSIONS
from src.context import Context
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.text.earnings_call_embeddings import PARAGRAPH_COLS, load_paragraphs, split_calls
from src.data_store.schema import Tables
from src.utils.earnings_call_split import split_call
from src.utils.nlp_sentiment import (
    EARNINGS_CALL_SENTIMENT_CACHE_MODEL,
    EARNINGS_CALL_SENTIMENT_INVALID_HANDLED_MODEL,
    EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL,
    SentimentEngine,
    get_sentiment_engine,
)
from src.utils.string import pad_cik_series
from src.utils.text_metrics import EarningsCallQuality, assess_earnings_call_sections, uncertainty_ratio, word_count

# SENTIMENT KPIs — from the local FinBERT-tone + Loughran-McDonald pass (`score_earnings_calls`).


# --------------------------------------------------------------------------- #
# Stage 1: incremental, cached FinBERT scoring                                  #
# --------------------------------------------------------------------------- #
def _valid_calls(paragraphs: pd.DataFrame) -> list[tuple[str, str, object, EarningsCallQuality]]:
    """(ticker, quarter, as_of, quality) for every call of a paragraph frame whose split status is
    `ok` and whose prepared_remarks / qa texts pass the shared quality gate; other calls are
    skipped, exactly as malformed section rows were."""
    calls = []
    for ticker, quarter, as_of, split in split_calls(paragraphs):
        if split.status != "ok":
            continue
        quality = assess_earnings_call_sections({"prepared_remarks": split.prepared_remarks, "qa": split.qa})
        if quality.valid:
            calls.append((ticker, quarter, as_of, quality))
    return calls


def score_earnings_calls(context: Context) -> pd.Timestamp | None:
    """Load source/cache once, then split, clean and score each missing call; save per ticker."""
    store, log = context.store, context.log

    tags = ("prepared_remarks", "qa")
    keys = ["ticker", "quarter", "tag"]

    # load tables
    df_paragraphs = store.load(
        Tables.earnings_call_sections,
        columns=PARAGRAPH_COLS,
        order_by=["ticker", "quarter", "paragraph"],
        optional=True,
    )
    df_cache = store.load(
        Tables.earnings_call_sentiment,
        columns=["ticker", "quarter", "tag", "as_of", "model"],
        where={
            "model": [
                EARNINGS_CALL_SENTIMENT_CACHE_MODEL,
                EARNINGS_CALL_SENTIMENT_INVALID_HANDLED_MODEL,
                EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL,
            ]
        },
        optional=True,
    )
    if df_paragraphs is None:
        log.warning("No earnings_call_sections -> sentiment scoring skipped (run extract-earnings-calls).")
        return None
    if df_cache is None:
        df_cache = pd.DataFrame(columns=[*keys, "as_of", "model"])

    pending = df_cache["model"].eq(EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL)
    pending_dates = pd.to_datetime(df_cache.loc[pending, "as_of"], errors="coerce").dropna()
    earliest = pd.Timestamp(pending_dates.min()) if not pending_dates.empty else None
    df_calls = df_paragraphs.loc[df_paragraphs["paragraph"].eq(1), ["ticker", "quarter"]].drop_duplicates()
    if df_calls.empty:
        return None

    # get the missing sentiments to do
    completed = pd.MultiIndex.from_frame(df_cache.loc[~pending, keys])
    df_missing = df_calls.merge(pd.DataFrame({"tag": tags}), how="cross")
    df_missing = df_missing.loc[~pd.MultiIndex.from_frame(df_missing[keys]).isin(completed)]
    if df_missing.empty:
        log.info("Earnings-call sentiment cache already complete.")
        return earliest

    missing = set(map(tuple, df_missing[keys].to_numpy()))
    df_paragraphs = df_paragraphs.merge(df_missing[["ticker", "quarter"]].drop_duplicates(), on=["ticker", "quarter"], how="inner")
    engine: SentimentEngine | None = None

    n_new = 0
    for ticker, df_ticker in df_paragraphs.groupby("ticker", sort=False):
        parts: list[pd.DataFrame] = []
        for quarter, df_call in df_ticker.groupby("quarter", sort=False):
            df_call = df_call.sort_values("paragraph", kind="stable")
            paragraphs = cast(list[dict[str, object]], df_call[["paragraph", "speaker", "content"]].to_dict("records"))
            split = split_call(paragraphs)
            if split.status != "ok":
                continue
            quality = assess_earnings_call_sections({"prepared_remarks": split.prepared_remarks, "qa": split.qa})
            if not quality.valid:
                continue
            dates = df_call["as_of"].dropna()
            as_of = dates.iloc[0] if not dates.empty else None
            call_tags = [tag for tag in tags if (ticker, quarter, tag) in missing and tag in quality.cleaned_sections]
            texts = [quality.cleaned_sections[tag] for tag in call_tags]

            if engine is None:
                engine = get_sentiment_engine()
            if engine is None:
                log.warning("Sentiment model unavailable -> earnings-call sections left unscored; earnings-call features will be skipped.")
                return earliest
            probabilities = engine.score_texts(texts)
            df_scored = pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "quarter": quarter,
                        "tag": tag,
                        "as_of": as_of,
                        "sent_pos": round(float(probability["pos"]), 6),
                        "sent_neg": round(float(probability["neg"]), 6),
                        "sent_neu": round(float(probability["neu"]), 6),
                        "n_words": int(word_count(text)),
                        "uncertainty_ratio": round(float(uncertainty_ratio(text)), 6),
                        "model": EARNINGS_CALL_SENTIMENT_CACHE_MODEL,
                    }
                    for tag, text, probability in zip(call_tags, texts, probabilities, strict=False)
                    if probability is not None
                ]
            )
            if not df_scored.empty:
                parts.append(df_scored)
        if not parts:
            continue
        df_scored = pd.concat(parts, ignore_index=True)
        store.save(Tables.earnings_call_sentiment, df_scored)
        n_new += len(df_scored)
        changed = pd.to_datetime(df_scored["as_of"], errors="coerce").min()
        if pd.notna(changed):
            earliest = changed if earliest is None else min(earliest, changed)
    log.info("Earnings-call sentiment: +%d newly scored rows -> '%s'.", n_new, Tables.earnings_call_sentiment)
    return earliest


def _scorable_keys(context: Context, keys: pd.DataFrame) -> set[tuple[str, str, str]]:
    """(ticker, quarter, tag) of `keys` that `score_earnings_calls` would score: the call is
    stored, splits `ok`, passes the quality gate and has a cleaned text for the tag."""
    paragraphs = load_paragraphs(context.store, sorted(keys["ticker"].astype(str).unique()))
    if paragraphs is None or paragraphs.empty:
        return set()
    calls = keys[["ticker", "quarter"]].astype(str).drop_duplicates()
    paragraphs = paragraphs.merge(calls, on=["ticker", "quarter"], how="inner")
    return {(ticker, quarter, tag) for ticker, quarter, _as_of, quality in _valid_calls(paragraphs) for tag in quality.cleaned_sections}


def acknowledge_earnings_call_invalidations(context: Context) -> int:
    """Mark refresh markers handled after the cube write succeeds, except those of calls that
    are still to be scored.

    A crash before acknowledgement leaves the marker pending, so the next run repeats the same
    inclusive repair instead of losing the historical correction. A scored section overwrites
    its marker (same key), so a marker still pending for a scorable call means scoring did not
    run (no sentiment engine): it stays pending, because `invalid-handled` counts as done and
    would drop the call from scoring for good. Markers of calls that no longer exist or never
    split into valid text are handled.
    """
    pending = context.store.load(
        Tables.earnings_call_sentiment,
        where={"model": EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL},
        optional=True,
    )
    if pending is None or pending.empty:
        return 0
    pending = pending[pd.to_datetime(pending["as_of"], errors="coerce").notna()].copy()
    if pending.empty:
        return 0
    scorable = _scorable_keys(context, pending)
    keys = zip(pending["ticker"].astype(str), pending["quarter"].astype(str), pending["tag"].astype(str), strict=True)
    pending = pending[[key not in scorable for key in keys]].copy()
    if pending.empty:
        return 0
    pending["model"] = EARNINGS_CALL_SENTIMENT_INVALID_HANDLED_MODEL
    return context.store.save(Tables.earnings_call_sentiment, pending)


# --------------------------------------------------------------------------- #
# Stage 2: smart KPIs + point-in-time daily alignment                           #
# --------------------------------------------------------------------------- #
def _validated_sentiment_cache(sentiment: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    """Keep complete, valid calls and refresh cheap metrics from the canonical source paragraphs
    (`sections`: `earnings_call_sections` rows, split per call)."""
    tags = ("prepared_remarks", "qa")
    if "model" not in sentiment or sentiment.empty:
        return sentiment.iloc[0:0].copy()
    sentiment = sentiment[sentiment["model"] == EARNINGS_CALL_SENTIMENT_CACHE_MODEL].copy()
    metrics = []
    for ticker, quarter, _as_of, quality in _valid_calls(sections):
        for tag in tags:
            cleaned = quality.cleaned_sections[tag]
            metrics.append(
                {
                    "ticker": ticker,
                    "quarter": quarter,
                    "tag": tag,
                    "n_words_current": word_count(cleaned),
                    "uncertainty_current": uncertainty_ratio(cleaned),
                }
            )
    if not metrics:
        return sentiment.iloc[0:0].copy()
    current = pd.DataFrame(metrics)
    out = sentiment.merge(current, on=["ticker", "quarter", "tag"], how="inner")
    complete = out.groupby(["ticker", "quarter"], sort=False)["tag"].nunique()
    complete = complete[complete == len(tags)].index
    keys = pd.MultiIndex.from_frame(out[["ticker", "quarter"]])
    out = out[keys.isin(complete)].copy()
    out["n_words"] = out.pop("n_words_current")
    out["uncertainty_ratio"] = out.pop("uncertainty_current")
    return out


def _per_call_kpis(sentiment: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-(ticker,quarter,tag) cache into one row per (ticker, quarter)
    with the smart KPIs. If source paragraphs are supplied, calls that do not split `ok`,
    fail the quality gate or are incomplete in the cache are excluded, and cheap
    word/uncertainty metrics are refreshed from the cleaned split text."""

    s = _validated_sentiment_cache(sentiment, sections)
    if s.empty:
        return pd.DataFrame(columns=["ticker", "quarter", "as_of"])
    s["net"] = s["sent_pos"].astype(float) - s["sent_neg"].astype(float)
    s["n_words"] = pd.to_numeric(s["n_words"], errors="coerce").fillna(0.0)
    s["uncertainty_ratio"] = pd.to_numeric(s["uncertainty_ratio"], errors="coerce")

    # per (ticker, quarter): tag-wise tone/words/uncertainty, then length-weighted call
    idx = ["ticker", "quarter"]
    as_of = s.groupby(idx)["as_of"].first()
    tone_tag = s.pivot_table(index=idx, columns="tag", values="net", aggfunc="mean")
    words_tag = s.pivot_table(index=idx, columns="tag", values="n_words", aggfunc="sum")
    unc_tag = s.pivot_table(index=idx, columns="tag", values="uncertainty_ratio", aggfunc="mean")

    w = words_tag.reindex(columns=tone_tag.columns).fillna(0.0)
    tot_w = w.sum(axis=1)
    ec_tone = (tone_tag * w).sum(axis=1) / tot_w.where(tot_w > 0)  # length-weighted tone
    ec_unc = (unc_tag.reindex(columns=w.columns) * w).sum(axis=1) / tot_w.where(tot_w > 0)
    prep, qa = "prepared_remarks", "qa"
    ec_qa_gap = tone_tag[qa] - tone_tag[prep] if qa in tone_tag.columns and prep in tone_tag.columns else np.nan

    per_q = pd.DataFrame(
        {
            "as_of": pd.to_datetime(as_of),
            "ec_tone": ec_tone,
            "ec_qa_gap": ec_qa_gap,
            "ec_uncertainty": ec_unc,
            "total_words": tot_w,
        }
    ).reset_index()
    return per_q.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def _quarter_number(value: object) -> float:
    text = str(value)
    if len(text) == 6 and text[4] == "Q" and text[:4].isdigit() and text[5] in "1234":
        return int(text[:4]) * 4 + int(text[5]) - 1
    return np.nan


def _issuer_history_zscore(per_call: pd.DataFrame, value_col: str) -> pd.Series:
    """Five-year, prior-only issuer z-score; at least four prior calls, sample std."""
    out = pd.Series(np.nan, index=per_call.index, dtype="float64")
    identity = "issuer_id" if "issuer_id" in per_call.columns else "ticker"
    for _, group in per_call.groupby(identity, sort=False):
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


def _daily_frame(per_call: pd.DataFrame, value_col: str, idx: pd.DatetimeIndex, availability_lag_days: int) -> pd.DataFrame:
    """Place each KPI after release for 66 sessions on a complete trading calendar.

    ``idx`` must span the source calls; a tail-only index would restart an expired call.
    ``StepCubeText`` therefore computes on the full calendar and slices only at write time.
    """
    if per_call.empty or idx.empty:
        return pd.DataFrame(index=idx)
    calendar = pd.DatetimeIndex(idx).normalize()
    tickers = per_call["ticker"].astype(str).drop_duplicates().tolist()
    frame = pd.DataFrame(np.nan, index=calendar, columns=tickers, dtype="float64")
    ordered = per_call.assign(as_of=pd.to_datetime(per_call["as_of"], errors="coerce")).sort_values(["ticker", "as_of"])
    for row in ordered.itertuples(index=False):
        if pd.isna(row.as_of):
            continue
        eligible = cast(pd.Timestamp, row.as_of).normalize() + pd.Timedelta(days=availability_lag_days)
        start = int(calendar.searchsorted(eligible, side="left"))
        if start >= len(calendar):
            continue
        stop = min(start + EARNINGS_CALL_SIGNAL_SESSIONS, len(calendar))
        frame.loc[calendar[start:stop], str(row.ticker)] = getattr(row, value_col)
    return frame


def sentiment_kpis(context: Context) -> pd.DataFrame | None:
    """Load source and current cache once, then derive per-call KPIs grouped by ticker."""
    store = context.store
    df_cache = store.load(
        Tables.earnings_call_sentiment,
        columns=["ticker", "quarter", "tag", "as_of", "sent_pos", "sent_neg", "n_words", "uncertainty_ratio", "model"],
        where={"model": EARNINGS_CALL_SENTIMENT_CACHE_MODEL},
        optional=True,
    )
    df_paragraphs = store.load(
        Tables.earnings_call_sections,
        columns=PARAGRAPH_COLS,
        where={"ticker": df_cache["ticker"].unique().tolist() if df_cache is not None else []},
        optional=True,
    )
    if df_cache is None or df_paragraphs is None:
        return None
    by_ticker = df_paragraphs.groupby("ticker", sort=False)
    parts: list[pd.DataFrame] = []
    for ticker, df_scores in df_cache.groupby("ticker", sort=False):
        if ticker not in by_ticker.groups:
            continue
        df_kpis = _per_call_kpis(df_scores, by_ticker.get_group(ticker))
        if not df_kpis.empty:
            parts.append(df_kpis)
    return pd.concat(parts, ignore_index=True) if parts else None


def build_earnings_call_feature_panel(
    per_call: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    availability: pd.DataFrame | None = None,
    *,
    availability_lag_days: int,
) -> pd.DataFrame:
    """Build the exact 12-column raw/issuer-history daily features from computed per-call KPIs."""
    if per_call.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    per_call = prepare_earnings_call_kpis(per_call)
    return _feature_panel_from_prepared(per_call, trading_index, availability, availability_lag_days)


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
    consecutive = (ordered["quarter"].map(_quarter_number) - previous_quarter.map(_quarter_number)) == 1
    per_call.loc[ordered.index, "ec_tone_delta"] = grouped["ec_tone"].diff().where(consecutive)
    previous_words = grouped["total_words"].shift(1)
    length_delta = cast(pd.Series, np.log(ordered["total_words"] / previous_words.where(previous_words > 0))).where(consecutive)
    per_call.loc[ordered.index, "ec_length_delta"] = length_delta.replace([np.inf, -np.inf], np.nan)
    for column in ("ec_qa_qq_distance", "ec_prep_qq_distance"):
        per_call.loc[ordered.index, column] = ordered[column].where(consecutive)
    return per_call


def _feature_panel_from_prepared(
    per_call: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    availability: pd.DataFrame | None,
    availability_lag_days: int,
) -> pd.DataFrame:
    fields: dict[str, pd.DataFrame] = {}
    for col in EARNINGS_CALL_FEATURES:
        if col not in per_call.columns:
            continue
        sub = per_call[["as_of", "ticker", col]]
        if sub[col].notna().sum() == 0:
            continue
        frame = _daily_frame(sub, col, trading_index, availability_lag_days)
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
