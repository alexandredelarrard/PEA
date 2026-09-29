"""
earnings_call_features.py  (src/data_aggregate/utils/earnings_call_features.py)
-------------------------------------------------------------------------------
Turn the parsed earnings-call SECTIONS (`earnings_call_sections`: prepared_remarks /
qa, one row per ticker·quarter·tag) into point-in-time raw and issuer-history features.

Two stages:

  1. score_earnings_calls(context)  — the EXPENSIVE, cached, incremental NLP pass.
     For every not-yet-scored (ticker, quarter, tag) it runs the local FinBERT-tone
     model (GPU if available) to get {pos, neg, neu} tone probabilities, plus cheap
     text metrics (word count, Loughran-McDonald uncertainty ratio), and upserts them
     to the `earnings_call_sentiment` cache — PER TICKER, so an interrupted run never
     loses GPU work. Skipped cleanly if torch/transformers are unavailable.

  2. build_earnings_call_feature_panel(...) — the cheap per-build derivation. From the
     cached per-call scores it builds raw and prior-only issuer-history KPIs, aligns
     them to the daily trading calendar with a one-session lag, and expires each call
     after 66 trading sessions. Missing calls stay null; genuine zero signals stay zero.

Smart KPIs (all leak-free; a call at date d only affects features on d+1 onward):
    ec_tone            length-weighted call tone  P(pos) − P(neg)      (level)
    ec_tone_delta      Δ tone vs the PRIOR call                        (tone momentum)
    ec_qa_gap          Q&A tone − prepared-remarks tone   (candor: scripted optimism
                       far above unscripted answers is a bearish tell)
    ec_uncertainty     length-weighted hedging ratio (LM uncertainty words)
    ec_length_delta    log(total words this call / prior call)         (disclosure Δ)
    ec_qa_coherence_mean   mean question/answer embedding cosine
    ec_qa_qq_distance      1 − cosine(Q&A embedding vs prior consecutive quarter)
    ec_prep_qq_distance    1 − cosine(prepared embedding vs prior consecutive quarter)
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import EARNINGS_CALL_FEATURES, EARNINGS_CALL_SCORED_TAGS, EARNINGS_CALL_SIGNAL_SESSIONS, FINBERT_TONE_MODEL
from src.context import Context
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.text.earnings_call_embeddings import build_embedding_kpis
from src.data_store.schema import Tables
from src.utils.nlp_sentiment import get_sentiment_engine
from src.utils.text_metrics import assess_earnings_call_sections, uncertainty_ratio, word_count

_SECTION_COLS = ["ticker", "quarter", "tag", "as_of", "text"]  # never SELECT * : `text` is huge


# --------------------------------------------------------------------------- #
# Stage 1: incremental, cached FinBERT scoring                                  #
# --------------------------------------------------------------------------- #
def _score_rows(engine, rows: pd.DataFrame) -> pd.DataFrame:
    """Score a frame of section rows (ticker, quarter, tag, as_of, text) -> per-call
    cache rows (tone probs + word count + uncertainty ratio). Pure given `engine`."""
    probs = engine.score_texts(rows["text"].tolist())
    out = []
    for r, p in zip(rows.itertuples(index=False), probs, strict=False):
        if p is None:  # blank section -> no tone
            continue
        out.append(
            {
                "ticker": r.ticker,
                "quarter": r.quarter,
                "tag": r.tag,
                "as_of": r.as_of,
                "sent_pos": round(float(p["pos"]), 6),
                "sent_neg": round(float(p["neg"]), 6),
                "sent_neu": round(float(p["neu"]), 6),
                "n_words": int(word_count(cast(str, r.text))),
                "uncertainty_ratio": round(float(uncertainty_ratio(cast(str, r.text))), 6),
                "model": FINBERT_TONE_MODEL,
            }
        )
    return pd.DataFrame(out)


def _yield_sections_to_score(context: Context, todo_keys: pd.DataFrame, sections: pd.DataFrame | None, tags: tuple[str, ...]):
    """GENERATOR yielding (ticker, rows_to_score) ONE TICKER AT A TIME — reads the transcript text
    per ticker (ticker+tag pushed down server-side, or sliced from a provided `sections` frame),
    keeping only the sections still absent from the cache. Bounded memory: never the whole table."""
    by_tkr: dict[str, set[tuple[str, str]]] = {}
    for tkr, q, tag in todo_keys[["ticker", "quarter", "tag"]].to_numpy():
        by_tkr.setdefault(tkr, set()).add((q, tag))
    for tkr, pairs in by_tkr.items():
        if sections is not None:
            g = sections[(sections["ticker"] == tkr) & (sections["tag"].isin(tags))]
        else:
            g = context.store.load(Tables.earnings_call_sections, _SECTION_COLS, where={"ticker": tkr, "tag": list(tags)}, optional=True)
            if g is None:
                continue
        g = g[[(q, tag) in pairs for q, tag in zip(g["quarter"], g["tag"], strict=False)]]
        valid_calls = []
        for _, call in g.groupby("quarter", sort=False):
            quality = assess_earnings_call_sections(dict(zip(call["tag"].astype(str), call["text"], strict=False)))
            if not quality.valid:
                continue
            call = call.copy()
            call["text"] = call["tag"].astype(str).map(quality.cleaned_sections)
            valid_calls.append(call)
        if valid_calls:
            yield tkr, pd.concat(valid_calls, ignore_index=True)


def score_earnings_calls(
    context: Context, sections: pd.DataFrame | None = None, tags: tuple[str, ...] = EARNINGS_CALL_SCORED_TAGS
) -> pd.Timestamp | None:
    """Ensure every high-signal section has a cached FinBERT tone score. MEMORY-SAFE + incremental:
    the REMAINING sections are found by comparing two key-column-only reads (section keys vs. the
    already-scored keys — never the transcript text), then each remaining ticker's text is read,
    scored and SAVED ONE TICKER AT A TIME (a generator, nothing accumulated). An interrupted GPU run
    keeps its progress; a re-run scores nothing. Returns None — the KPIs stream the cache back per
    ticker (`sentiment_kpis_streamed`)."""
    store, log = context.store, context.log
    # section keys (no text) vs. already-scored keys (no probs) -> only what's left to score
    keys = (
        sections[sections["tag"].isin(tags)]
        if sections is not None
        else store.load(Tables.earnings_call_sections, ["ticker", "quarter", "tag"], where={"tag": list(tags)}, optional=True)
    )
    if keys is None or keys.empty:
        log.warning("No earnings_call_sections -> sentiment scoring skipped (run fetch_earnings_calls).")
        return None
    sec_keys = keys[["ticker", "quarter", "tag"]].drop_duplicates()
    done_df = store.load(Tables.earnings_call_sentiment, ["ticker", "quarter", "tag"], optional=True)
    done = set() if done_df is None else set(map(tuple, done_df.drop_duplicates().to_numpy()))
    todo_keys = sec_keys[[tuple(k) not in done for k in sec_keys.to_numpy()]]
    if todo_keys.empty:
        log.info("Earnings-call sentiment cache already complete.")
        return None

    engine = get_sentiment_engine(log)
    if engine is None:  # torch/transformers/model unavailable
        log.warning("Sentiment model unavailable -> %d sections left unscored; earnings-call features will be skipped.", len(todo_keys))
        return None

    log.info("Scoring %d earnings-call sections on %s (FinBERT-tone)...", len(todo_keys), engine.device)
    n_new = 0
    earliest: pd.Timestamp | None = None
    for _tkr, grp in _yield_sections_to_score(context, todo_keys, sections, tags):
        scored = _score_rows(engine, grp)
        if not scored.empty:
            store.save(Tables.earnings_call_sentiment, scored)  # iterative per-ticker upsert
            n_new += len(scored)
            changed = pd.to_datetime(scored["as_of"], errors="coerce").min()
            if pd.notna(changed):
                earliest = changed if earliest is None else min(earliest, changed)
    log.info("Earnings-call sentiment: +%d newly scored rows -> '%s'.", n_new, Tables.earnings_call_sentiment)
    return earliest


# --------------------------------------------------------------------------- #
# Stage 2: smart KPIs + point-in-time daily alignment                           #
# --------------------------------------------------------------------------- #
def _validated_sentiment_cache(sentiment: pd.DataFrame, sections: pd.DataFrame | None) -> pd.DataFrame:
    """Keep complete, valid calls and refresh cheap metrics from canonical source text."""
    if sections is None:
        return sentiment.copy()
    metrics = []
    for (ticker, quarter), call in sections.groupby(["ticker", "quarter"], sort=False):
        quality = assess_earnings_call_sections(dict(zip(call["tag"].astype(str), call["text"], strict=False)))
        if not quality.valid:
            continue
        for tag in EARNINGS_CALL_SCORED_TAGS:
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
    complete = complete[complete == len(EARNINGS_CALL_SCORED_TAGS)].index
    keys = pd.MultiIndex.from_frame(out[["ticker", "quarter"]])
    out = out[keys.isin(complete)].copy()
    out["n_words"] = out.pop("n_words_current")
    out["uncertainty_ratio"] = out.pop("uncertainty_current")
    return out


def _per_call_kpis(sentiment: pd.DataFrame, sections: pd.DataFrame | None) -> pd.DataFrame:
    """Collapse the per-(ticker,quarter,tag) cache into one row per (ticker, quarter)
    with the smart KPIs. If source sections are supplied, malformed/incomplete calls
    are excluded and cheap word/uncertainty metrics are refreshed from cleaned text."""

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


def attach_issuer_identity(
    per_call: pd.DataFrame,
    symbol_tenure: pd.DataFrame | None,
    entity_lineage: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach the point-in-time economic issuer used by the history normalization."""
    out = per_call.copy()
    out["issuer_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if symbol_tenure is None or symbol_tenure.empty:
        return out.iloc[0:0]
    tenure = symbol_tenure.copy()
    tenure["valid_from"] = pd.to_datetime(tenure["valid_from"], errors="coerce")
    tenure["valid_to"] = pd.to_datetime(tenure["valid_to"], errors="coerce")
    entity: dict[str, str] = {}
    if entity_lineage is not None and not entity_lineage.empty:
        for cik, group in entity_lineage.groupby(entity_lineage["cik"].astype(str), sort=False):
            identifiers = group["entity_id"].dropna().astype(str).unique()
            if len(identifiers) == 1:
                entity[str(cik)] = str(identifiers[0])
    by_symbol = {str(symbol): group for symbol, group in tenure.groupby(tenure["symbol"].astype(str), sort=False)}
    for row_index, row in out.iterrows():
        date = pd.to_datetime(row["as_of"], errors="coerce")
        candidates = by_symbol.get(str(row["ticker"]))
        if candidates is None:
            continue
        candidates = candidates[candidates["valid_from"] <= date]
        candidates = candidates[candidates["valid_to"].isna() | (date < candidates["valid_to"])]
        if candidates.empty:
            continue
        identifiers = {entity.get(str(cik)) for cik in candidates["issuer_cik"]}
        identifiers.discard(None)
        if len(identifiers) == 1:
            out.at[row_index, "issuer_id"] = identifiers.pop()
    return out[out["issuer_id"].notna()].copy()


def _daily_frame(per_call: pd.DataFrame, value_col: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Place each KPI on the first trading session after release for exactly 66 sessions."""
    if per_call.empty or idx.empty:
        return pd.DataFrame(index=idx)
    calendar = pd.DatetimeIndex(idx).normalize()
    tickers = per_call["ticker"].astype(str).drop_duplicates().tolist()
    frame = pd.DataFrame(np.nan, index=calendar, columns=tickers, dtype="float64")
    ordered = per_call.assign(as_of=pd.to_datetime(per_call["as_of"], errors="coerce")).sort_values(["ticker", "as_of"])
    for row in ordered.itertuples(index=False):
        if pd.isna(row.as_of):
            continue
        start = int(calendar.searchsorted(pd.Timestamp(row.as_of).normalize(), side="right"))
        if start >= len(calendar):
            continue
        stop = min(start + EARNINGS_CALL_SIGNAL_SESSIONS, len(calendar))
        frame.loc[calendar[start:stop], str(row.ticker)] = getattr(row, value_col)
    return frame


# SENTIMENT KPIs — from the local FinBERT-tone + Loughran-McDonald pass (`score_earnings_calls`).
_RAW_KPI_COLS = [
    "ec_tone",
    "ec_qa_gap",
    "ec_uncertainty",
    "ec_qa_coherence_mean",
    "ec_tone_delta",
    "ec_length_delta",
    "ec_qa_qq_distance",
    "ec_prep_qq_distance",
]
_HISTORY_BASES = ["ec_tone", "ec_qa_gap", "ec_uncertainty", "ec_qa_coherence_mean"]
_KPI_COLS = list(EARNINGS_CALL_FEATURES)


def sentiment_kpis_streamed(context: Context) -> pd.DataFrame | None:
    """Derive the per-call SENTIMENT KPIs from the cache PER TICKER (bounded memory: one ticker's
    sentiment rows + source text at a time, never the whole sections/sentiment tables).
    Returns the per-call KPI frame, or None if the cache is empty."""
    store = context.store
    parts = []
    tickers = list(store.distinct(Tables.earnings_call_sentiment, "ticker"))
    for start in range(0, len(tickers), 25):
        batch = tickers[start : start + 25]
        scored = store.load(Tables.earnings_call_sentiment, where={"ticker": batch}, optional=True)
        sections = store.load(
            Tables.earnings_call_sections,
            _SECTION_COLS,
            where={"ticker": batch, "tag": list(EARNINGS_CALL_SCORED_TAGS)},
            optional=True,
        )
        if scored is None or sections is None:
            continue
        for ticker, s in scored.groupby("ticker", sort=False):
            ticker_sections = sections[sections["ticker"] == ticker]
            kpis = _per_call_kpis(s, ticker_sections)
            if not kpis.empty:
                parts.append(kpis)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def build_earnings_call_feature_panel(
    sentiment: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    sections: pd.DataFrame | None = None,
    embeddings: pd.DataFrame | None = None,
    per_call: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the exact 12-column raw/issuer-history daily feature contract.

    Empty if the sentiment cache is unavailable. ``per_call`` may be supplied from
    the bounded-memory stream to skip re-deriving cached call KPIs.
    """
    if per_call is None:
        if sentiment is None or sentiment.empty or "sent_pos" not in sentiment.columns:
            return pd.DataFrame(columns=["date", "ticker"])
        per_call = _per_call_kpis(sentiment, sections)
    if per_call is None or per_call.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    ekpi = build_embedding_kpis(embeddings)
    if ekpi is not None and not ekpi.empty:
        per_call = per_call.merge(ekpi, on=["ticker", "quarter"], how="left")
    per_call = prepare_earnings_call_kpis(per_call)
    return _feature_panel_from_prepared(per_call, peer_dict, trading_index, availability)


def prepare_earnings_call_kpis(per_call: pd.DataFrame) -> pd.DataFrame:
    """Add the exact raw/history KPI contract on the per-call grain."""
    per_call = per_call.copy()
    for col in _RAW_KPI_COLS:
        if col not in per_call.columns:
            per_call[col] = np.nan
    for col in _HISTORY_BASES:
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
    length_delta = np.log(ordered["total_words"] / previous_words.where(previous_words > 0)).where(consecutive)
    per_call.loc[ordered.index, "ec_length_delta"] = length_delta.replace([np.inf, -np.inf], np.nan)
    prior_ticker = grouped["ticker"].shift(1)
    comparable_embedding_predecessor = consecutive & ordered["ticker"].eq(prior_ticker)
    for column in ("ec_qa_qq_distance", "ec_prep_qq_distance"):
        per_call.loc[ordered.index, column] = ordered[column].where(comparable_embedding_predecessor)
    return per_call


def _feature_panel_from_prepared(
    per_call: pd.DataFrame,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    availability: pd.DataFrame | None,
) -> pd.DataFrame:
    fields: dict[str, pd.DataFrame] = {}
    for col in _KPI_COLS:
        if col not in per_call.columns:
            continue
        sub = per_call.loc[per_call[col].notna(), ["as_of", "ticker", col]]
        if sub.empty:
            continue
        frame = _daily_frame(sub, col, trading_index)
        if frame is not None and not frame.empty and frame.notna().any().any():
            fields[col] = frame
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    panel = build_peer_relative_panel(fields, peer_dict, emission={name: "raw" for name in fields}, availability=availability)
    for name in _KPI_COLS:
        column = f"f_{name}"
        if column not in panel.columns:
            panel[column] = pd.Series(np.nan, index=panel.index, dtype="float32")
    return panel[["date", "ticker", *[f"f_{name}" for name in _KPI_COLS]]]
