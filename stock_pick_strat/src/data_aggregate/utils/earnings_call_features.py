"""
earnings_call_features.py  (src/data_aggregate/utils/earnings_call_features.py)
-------------------------------------------------------------------------------
Turn the parsed earnings-call SECTIONS (`earnings_call_sections`: prepared_remarks /
qa, one row per ticker·quarter·tag) into point-in-time, peer-relative model features.

Two stages:

  1. score_earnings_calls(context)  — the EXPENSIVE, cached, incremental NLP pass.
     For every not-yet-scored (ticker, quarter, tag) it runs the local FinBERT-tone
     model (GPU if available) to get {pos, neg, neu} tone probabilities, plus cheap
     text metrics (word count, Loughran-McDonald uncertainty ratio), and upserts them
     to the `earnings_call_sentiment` cache — PER TICKER, so an interrupted run never
     loses GPU work. Skipped cleanly if torch/transformers are unavailable.

  2. build_earnings_call_feature_panel(...) — the CHEAP per-build derivation. From the
     cached per-call scores (+ the section text for the vocabulary metric) it builds
     the "smart" KPIs that characterize how a call's language changes, then aligns them
     to the daily trading calendar (stamped on the call date, +1-day lag for the
     transcript-publication delay, forward-filled until the next call) and expresses
     each as `f_ec_<kpi>_{xs,vs_peers}` via the shared peer-relative builder.

Smart KPIs (all leak-free; a call at date d only affects features on d+1 onward):
    ec_tone            length-weighted call tone  P(pos) − P(neg)      (level)
    ec_tone_delta      Δ tone vs the PRIOR call                        (tone momentum)
    ec_qa_gap          Q&A tone − prepared-remarks tone   (candor: scripted optimism
                       far above unscripted answers is a bearish tell)
    ec_uncertainty     length-weighted hedging ratio (LM uncertainty words)
    ec_length_delta    log(total words this call / prior call)         (disclosure Δ)
    ec_vocab_novelty   1 − cosine(prepared-remarks bag-of-words vs prior call)
                       (a new narrative / strategy shift)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import (
    EARNINGS_CALL_SCORED_TAGS,
    EARNINGS_CALL_SECTIONS_TABLE,
    EARNINGS_CALL_SENTIMENT_TABLE,
    FINBERT_TONE_MODEL,
)
from src.context import Context
from src.data_aggregate.utils.earnings_call_embeddings import build_embedding_kpis
from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel
from src.utils.nlp_sentiment import get_sentiment_engine
from src.utils.text_metrics import content_frequency, cosine_similarity, uncertainty_ratio, word_count

_TRANSCRIPT_LAG_DAYS = 1        # transcript public the trading day AFTER the call
_FFILL_LIMIT = 190             # ~9 months: bridge a skipped quarter, but let stale calls die
_VOCAB_TAG = "prepared_remarks"  # scripted narrative — where a strategy shift shows up


# --------------------------------------------------------------------------- #
# Stage 1: incremental, cached FinBERT scoring                                  #
# --------------------------------------------------------------------------- #
def _score_rows(engine, rows: pd.DataFrame) -> pd.DataFrame:
    """Score a frame of section rows (ticker, quarter, tag, as_of, text) -> per-call
    cache rows (tone probs + word count + uncertainty ratio). Pure given `engine`."""
    probs = engine.score_texts(rows["text"].tolist())
    out = []
    for r, p in zip(rows.itertuples(index=False), probs):
        if p is None:                               # blank section -> no tone
            continue
        out.append({
            "ticker": r.ticker, "quarter": r.quarter, "tag": r.tag, "as_of": r.as_of,
            "sent_pos": round(float(p["pos"]), 6),
            "sent_neg": round(float(p["neg"]), 6),
            "sent_neu": round(float(p["neu"]), 6),
            "n_words": int(word_count(r.text)),
            "uncertainty_ratio": round(float(uncertainty_ratio(r.text)), 6),
            "model": FINBERT_TONE_MODEL,
        })
    return pd.DataFrame(out)


def score_earnings_calls(context: Context,
                         sections: pd.DataFrame | None = None,
                         tags: tuple[str, ...] = EARNINGS_CALL_SCORED_TAGS
                         ) -> pd.DataFrame | None:
    """Ensure every high-signal section has a cached FinBERT tone score; return the
    full `earnings_call_sentiment` frame (None if there is nothing to score / no model).

    Incremental: only (ticker, quarter, tag) absent from the cache are scored, and the
    cache is upserted PER TICKER so an interrupted GPU run keeps its progress."""
    store = context.store
    log = context.log
    if sections is None:
        sections = store.load(EARNINGS_CALL_SECTIONS_TABLE)
    if sections is None or sections.empty:
        log.warning("No earnings_call_sections -> sentiment scoring skipped "
                    "(run fetch_earnings_calls).")
        return None

    sections = sections[sections["tag"].isin(tags)].copy()
    existing = store.load(EARNINGS_CALL_SENTIMENT_TABLE)
    done = (set() if existing is None or existing.empty
            else set(map(tuple, existing[["ticker", "quarter", "tag"]].to_numpy())))
    key = list(zip(sections["ticker"], sections["quarter"], sections["tag"]))
    todo = sections[[k not in done for k in key]]
    if todo.empty:
        log.info("Earnings-call sentiment cache already complete (%d rows).",
                 0 if existing is None else len(existing))
        return existing

    engine = get_sentiment_engine(log)
    if engine is None:                              # torch/transformers/model unavailable
        log.warning("Sentiment model unavailable -> %d calls left unscored; "
                    "earnings-call features will be skipped.", len(todo))
        return existing if existing is not None and not existing.empty else None

    log.info("Scoring %d earnings-call sections on %s (FinBERT-tone)...",
             len(todo), engine.device)
    n_new = 0
    for tkr, grp in todo.groupby("ticker", sort=False):
        scored = _score_rows(engine, grp)
        if not scored.empty:
            store.save(EARNINGS_CALL_SENTIMENT_TABLE, scored)   # upsert per ticker
            n_new += len(scored)
    log.info("Earnings-call sentiment: +%d newly scored rows -> '%s'.",
             n_new, EARNINGS_CALL_SENTIMENT_TABLE)
    return store.load(EARNINGS_CALL_SENTIMENT_TABLE)


# --------------------------------------------------------------------------- #
# Stage 2: smart KPIs + point-in-time daily alignment                           #
# --------------------------------------------------------------------------- #
def _per_call_kpis(sentiment: pd.DataFrame, sections: pd.DataFrame | None) -> pd.DataFrame:
    """Collapse the per-(ticker,quarter,tag) cache into one row per (ticker, quarter)
    with the smart KPIs. Cross-call KPIs (deltas, novelty) are computed per ticker in
    call order. Returns columns: ticker, as_of, ec_tone, ec_tone_delta, ec_qa_gap,
    ec_uncertainty, ec_length_delta, ec_vocab_novelty."""
    s = sentiment.copy()
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
    ec_tone = (tone_tag * w).sum(axis=1) / tot_w.where(tot_w > 0)          # length-weighted tone
    ec_unc = (unc_tag.reindex(columns=w.columns) * w).sum(axis=1) / tot_w.where(tot_w > 0)
    prep, qa = "prepared_remarks", "qa"
    ec_qa_gap = (tone_tag[qa] - tone_tag[prep]
                 if qa in tone_tag.columns and prep in tone_tag.columns else np.nan)

    per_q = pd.DataFrame({
        "as_of": pd.to_datetime(as_of), "ec_tone": ec_tone,
        "ec_qa_gap": ec_qa_gap, "ec_uncertainty": ec_unc,
        "total_words": tot_w,
        # per-section tone levels (helpers -> quarter-to-quarter tone deltas below)
        "qa_tone_lvl": tone_tag[qa] if qa in tone_tag.columns else np.nan,
        "prep_tone_lvl": tone_tag[prep] if prep in tone_tag.columns else np.nan,
    }).reset_index()

    nov = _vocab_novelty(sections) if sections is not None else None
    if nov is not None:
        per_q = per_q.merge(nov, on=idx, how="left")
    else:
        per_q["ec_vocab_novelty"] = np.nan

    # cross-call deltas in CALL ORDER (by call date), per ticker
    per_q = per_q.sort_values(["ticker", "as_of"]).reset_index(drop=True)
    g = per_q.groupby("ticker", sort=False)
    per_q["ec_tone_delta"] = g["ec_tone"].diff()
    prev_words = g["total_words"].shift(1)
    per_q["ec_length_delta"] = np.log(per_q["total_words"] / prev_words.where(prev_words > 0))
    per_q["ec_length_delta"] = per_q["ec_length_delta"].replace([np.inf, -np.inf], np.nan)
    # quarter-to-quarter tone distance, PER SECTION (qa vs qa, prepared vs prepared)
    per_q["ec_qa_tone_delta"] = g["qa_tone_lvl"].diff()
    per_q["ec_prep_tone_delta"] = g["prep_tone_lvl"].diff()
    return per_q.drop(columns=["qa_tone_lvl", "prep_tone_lvl"])


def _vocab_novelty(sections: pd.DataFrame, tag: str = _VOCAB_TAG) -> pd.DataFrame | None:
    """Per (ticker, quarter): 1 − cosine(bag-of-words this call vs the prior call), on
    the `tag` section (scripted narrative). None if the tag is absent."""
    sec = sections[sections["tag"] == tag][["ticker", "quarter", "as_of", "text"]].copy()
    if sec.empty:
        return None
    sec["as_of"] = pd.to_datetime(sec["as_of"])
    sec = sec.sort_values(["ticker", "as_of"])
    rows = []
    for tkr, grp in sec.groupby("ticker", sort=False):
        prev_cf = None
        for r in grp.itertuples(index=False):
            cf = content_frequency(r.text)
            nov = np.nan if prev_cf is None else 1.0 - cosine_similarity(cf, prev_cf)
            rows.append({"ticker": tkr, "quarter": r.quarter, "ec_vocab_novelty": nov})
            prev_cf = cf
    return pd.DataFrame(rows)


def _daily_frame(per_call: pd.DataFrame, value_col: str,
                 idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide [date × ticker] point-in-time frame for one KPI: stamp the value on the
    call date, forward-fill until the next call (bounded), and lag one trading day so
    it is only visible AFTER the call (transcript-publication delay)."""
    piv = per_call.pivot_table(index="as_of", columns="ticker", values=value_col, aggfunc="last")
    if piv.empty:
        return piv
    piv.index = pd.to_datetime(piv.index).normalize()
    piv = piv[~piv.index.duplicated(keep="last")].sort_index()
    piv = (piv.reindex(piv.index.union(idx)).ffill(limit=_FFILL_LIMIT)
           .reindex(idx).shift(_TRANSCRIPT_LAG_DAYS))
    return piv


_KPI_COLS = ["ec_tone", "ec_tone_delta", "ec_qa_gap", "ec_uncertainty",
             "ec_length_delta", "ec_vocab_novelty",
             "ec_qa_tone_delta", "ec_prep_tone_delta",   # per-section quarter-to-quarter tone drift
             # OpenAI-embedding KPIs (present only when the embedding cache is passed):
             "ec_qa_coherence_mean", "ec_qa_coherence_std", "ec_n_qa",
             "ec_qa_qq_sim", "ec_prep_qq_sim"]


def build_earnings_call_feature_panel(
    sentiment: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    sections: pd.DataFrame | None = None,
    embeddings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format earnings-call feature panel (`f_ec_<kpi>_vs_peers`, `f_ec_<kpi>_xs`).
    Empty if the sentiment cache is unavailable/empty. When `embeddings` (the
    `earning_calls_embedding` cache) is provided, the Q&A-coherence + quarter-to-quarter
    embedding-drift KPIs are merged in and expressed on the same daily, peer-relative basis."""
    if sentiment is None or sentiment.empty or "sent_pos" not in sentiment.columns:
        return pd.DataFrame(columns=["date", "ticker"])
    per_call = _per_call_kpis(sentiment, sections)
    if per_call.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    ekpi = build_embedding_kpis(embeddings)
    if ekpi is not None and not ekpi.empty:
        per_call = per_call.merge(ekpi, on=["ticker", "quarter"], how="left")

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
    return build_peer_relative_panel(fields, peer_dict)
