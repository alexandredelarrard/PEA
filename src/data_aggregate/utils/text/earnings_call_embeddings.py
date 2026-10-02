"""
earnings_call_embeddings.py  (src/data_aggregate/utils/earnings_call_embeddings.py)
-----------------------------------------------------------------------------------
OpenAI-embedding layer for earnings calls, on top of the raw paragraphs of `earnings_call_sections`.

Two stages, mirroring the FinBERT sentiment pipeline:

  1. embed_earnings_calls(context)  — the EXPENSIVE, cached, incremental OpenAI pass. For every
     not-yet-embedded (ticker, quarter) it reads the call's source paragraphs, splits them with
     `src/utils/earnings_call_split.split_call` (speaker turns come from the source `speaker`
     field, so no header is parsed; operator / IR-flow / pure-courtesy turns dropped; courtesy
     preambles stripped) and embeds each `CallSplit.turns` text, writing ONE ROW PER TURN to
     `earning_calls_embedding`:
         ticker, quarter, seq (turn order in the call), section (qa / prepared_remarks),
         tag (question / answer / prepared), exchange_idx (which Q&A pair; -1 for prepared),
         answer_idx (0 for the question, 1..k for the 1st..last answer turn; -1 for prepared),
         person (the speaker -- the analyst on question rows, the manager on answer rows),
         text (the cleaned turn), as_of (the call date), embedding (the turn's OpenAI vector),
         model (the cache tag "<OpenAI model>:<EARNINGS_CALL_EMBEDDING_CACHE_VERSION>"), run_at.
     A call is complete only under the current cache tag, so a turn-text change (a version bump)
     re-embeds every call through this same incremental pass.
     Storing per turn keeps every question/answer embedding we pay for — auditable and reusable —
     at NO extra API cost vs a pooled design (same text is embedded either way). Incremental
     & per-call upsert, so an interrupted (billed) run never loses work and re-runs make ZERO calls.

  2. build_embedding_kpis(embeddings) — the CHEAP per-build derivation of point-in-time KPIs
     DERIVED from the turn rows (nothing is precomputed at store time except the vectors):
         * ec_qa_coherence_mean — per exchange, the AVERAGE cosine of the question vs EACH of
           its answer turns; the quarter KPI is the mean of those exchange averages
         * ec_qa_qq_sim   — cosine(this quarter's POOLED q&a turns, prior quarter's)   (narrative
         * ec_prep_qq_sim — cosine(this quarter's POOLED prepared turns, prior quarter's)  drift)
     These merge into the exact raw/history earnings-call feature panel.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    EARNINGS_CALL_EMBEDDING_CACHE_MODEL,
    EARNINGS_CALL_EMBEDDING_CACHE_VERSION,
    EARNINGS_CALL_EMBEDDING_MODEL,
    EARNINGS_CALL_TAG_ANSWER,
    EARNINGS_CALL_TAG_QUESTION,
)
from src.context import Context
from src.data_store.schema import Tables
from src.data_store.store import DataStore

# `gpt_extract` is a shared service, like `src/utils/` -- the one sanctioned cross-import
# between src/ subfolders. It owns the single OpenAI client factory and the measured
# 28,000-char cap; the alternative was a second copy of both living here.
from src.gpt_extract import cosine, embed_texts, openai_api_key
from src.utils.earnings_call_split import CallSplit, split_call

_QA_TAG, _PREP_TAG = "qa", "prepared_remarks"

# ---- source paragraphs -> split calls ----------------------------------------- #
# `earnings_call_sections` is one row per source paragraph; `content` is the bulk of the table,
# so every text read names its columns and is scoped to one ticker (or one batch of tickers).
PARAGRAPH_COLS = ["ticker", "quarter", "paragraph", "as_of", "speaker", "content"]
_SPLIT_COLS = ["paragraph", "speaker", "content"]


def call_keys(store: DataStore, paragraphs: pd.DataFrame | None = None) -> pd.DataFrame:
    """(ticker, quarter) of every stored call. Read as DISTINCT quarters per ticker, never the
    text and never one row per paragraph; sliced from `paragraphs` when a frame is supplied."""
    if paragraphs is not None:
        return paragraphs[["ticker", "quarter"]].drop_duplicates().reset_index(drop=True)
    frames = [
        pd.DataFrame({"ticker": ticker, "quarter": store.distinct(Tables.earnings_call_sections, "quarter", where={"ticker": ticker})})
        for ticker in store.distinct(Tables.earnings_call_sections, "ticker")
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "quarter"])


def load_paragraphs(store: DataStore, tickers: str | list[str]) -> pd.DataFrame | None:
    """The projected paragraph rows of `tickers`, ordered by call and paragraph; None if absent."""
    return store.load(
        Tables.earnings_call_sections,
        PARAGRAPH_COLS,
        where={"ticker": tickers},
        order_by=["ticker", "quarter", "paragraph"],
        optional=True,
    )


def split_calls(paragraphs: pd.DataFrame) -> Iterator[tuple[str, str, object, CallSplit]]:
    """(ticker, quarter, as_of, CallSplit) per call of a paragraph frame. The call's `as_of` is
    its paragraphs' `as_of` (the real call date, one per call)."""
    ordered = paragraphs.sort_values(["ticker", "quarter", "paragraph"], kind="stable")
    for (ticker, quarter), call in ordered.groupby(["ticker", "quarter"], sort=False):
        dates = call["as_of"].dropna()
        as_of = dates.iloc[0] if len(dates) else None
        yield str(ticker), str(quarter), as_of, split_call(call[_SPLIT_COLS].to_dict("records"))


def _drop_stale_turns(store, log, counts: dict[tuple[str, str], int]) -> None:
    """After a FORCE re-embed, drop rows whose `seq` no longer exists — a re-parse (e.g. after a
    splitter change) can yield FEWER turns than a prior run left cached; upsert-by-PK would leave
    those tail rows orphaned. Reads the KEY columns of the re-embedded calls only and deletes the
    stale tail per call; the previous version pulled the whole table (every 1536-dim vector) into
    RAM to rewrite it via `replace`."""
    keys = store.load(
        Tables.earning_calls_embedding,
        ["ticker", "quarter", "seq"],
        where={"ticker": [t for t, _ in counts], "quarter": [q for _, q in counts]},
        optional=True,
    )
    if keys is None:
        return
    stale: dict[tuple[str, str], list[int]] = {}
    for tkr, quarter, seq in keys.to_numpy():
        call = (tkr, quarter)
        if call in counts and int(seq) >= counts[call]:
            stale.setdefault(call, []).append(int(seq))
    n = 0
    for (tkr, quarter), seqs in stale.items():
        n += store.delete(Tables.earning_calls_embedding, {"ticker": tkr, "quarter": quarter, "seq": seqs})
    if n:
        log.info("Force re-embed reconcile: dropped %d stale turn rows.", n)


# columns the KPI derivation needs — everything BUT the bulky `text` (and the audit columns): a
# projected read keeps the per-ticker KPI streaming from pulling the turn text it never uses.
_KPI_LOAD_COLS = ["ticker", "quarter", "as_of", "section", "tag", "exchange_idx", "embedding", "model"]


def _embedded_calls(store, cache_model: str) -> set[tuple[str, str]]:
    """(ticker, quarter) already embedded — reads ONLY the two key columns (never the 1536-dim
    vectors), so the done-set check costs almost nothing even on a million-row cache."""
    if cache_model not in set(store.distinct(Tables.earning_calls_embedding, "model")):
        return set()
    df = store.load(
        Tables.earning_calls_embedding,
        ["ticker", "quarter"],
        where={"model": cache_model},
        optional=True,
    )
    if df is None:
        return set()
    return set(map(tuple, df.drop_duplicates().to_numpy()))


def _yield_call_turns(
    context: Context, remaining: list[tuple[str, str]], paragraphs: pd.DataFrame | None = None
) -> Iterator[tuple[str, str, list[dict], object]]:
    """GENERATOR yielding (ticker, quarter, turns, as_of) for each remaining call, reading the
    paragraphs ONE TICKER AT A TIME (bounded memory: never the whole table) or slicing a provided
    paragraph frame. `turns` is `CallSplit.turns`, whatever the split status: the embedding pass
    has no quality gate, a call's prepared turns still anchor the next quarter's drift."""
    by_tkr: dict[str, set[str]] = {}
    for tkr, q in remaining:
        by_tkr.setdefault(tkr, set()).add(q)
    for tkr, quarters in by_tkr.items():
        g = paragraphs[paragraphs["ticker"] == tkr] if paragraphs is not None else load_paragraphs(context.store, tkr)
        if g is None:
            continue
        for ticker, quarter, as_of, split in split_calls(g[g["quarter"].isin(quarters)]):
            yield ticker, quarter, split.turns, as_of


def embed_earnings_calls(
    context: Context,
    sections: pd.DataFrame | None = None,
    model: str = EARNINGS_CALL_EMBEDDING_MODEL,
    force: bool = False,
    client: Any | None = None,
) -> pd.Timestamp | None:
    """Ensure every call's speaker turns are embedded + cached in `earning_calls_embedding` (one row
    per turn). MEMORY-SAFE + incremental: first the REMAINING calls are found by comparing two
    key-column-only reads (stored calls vs. already-embedded — never the vectors or the text), then each
    remaining call's paragraphs are read, split, embedded and SAVED ONE CALL AT A TIME (a generator,
    so nothing is accumulated and flushed at the end). An interrupted (billed) run keeps every saved
    call and a re-run makes ZERO calls. `sections` optionally supplies `earnings_call_sections`
    paragraph rows instead of the store; `client` injects a stub embedder for tests. Returns the
    earliest call date (re)embedded, or None."""
    store, log = context.store, context.log
    if client is None and not openai_api_key():
        log.warning("OPENAI/OPEN_AI_API_KEY not set -> earnings-call embedding skipped.")
        return None

    universe = [(str(t), str(q)) for t, q in call_keys(store, sections).itertuples(index=False, name=None)]
    if not universe:
        log.warning("No earnings_call_sections -> embedding skipped (run extract-earnings-calls).")
        return None
    cache_model = f"{model}:{EARNINGS_CALL_EMBEDDING_CACHE_VERSION}"
    done = set() if force else _embedded_calls(store, cache_model)
    remaining = [k for k in universe if tuple(k) not in done]
    if not remaining:
        log.info("Earnings-call embedding cache already complete (%d calls).", len(universe))
        return None

    log.info("Embedding %d earnings calls per-turn (OpenAI %s)...", len(remaining), model)
    n_new, counts = 0, {}
    earliest: pd.Timestamp | None = None
    for tkr, q, turns, aod in tqdm(_yield_call_turns(context, remaining, sections), "EC embeddings", total=len(remaining)):
        run_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        counts[(tkr, q)] = len(turns)
        changed = pd.to_datetime(aod, errors="coerce")
        if pd.notna(changed):
            earliest = changed if earliest is None else min(earliest, changed)
        if not turns:
            continue
        vectors = embed_texts([t["text"] for t in turns], model=model, client=client)
        rows = [
            {
                "ticker": tkr,
                "quarter": q,
                "seq": i,
                "section": t["section"],
                "tag": t["tag"],
                "exchange_idx": int(t["exchange_idx"]),
                "answer_idx": int(t["answer_idx"]),
                "person": t["person"],
                "text": t["text"],
                "as_of": aod,
                "embedding": [float(x) for x in vectors[i]],
                "model": cache_model,
                "run_at": run_at,
            }
            for i, t in enumerate(turns)
        ]
        store.save(Tables.earning_calls_embedding, pd.DataFrame(rows))  # iterative per-call upsert
        n_new += len(rows)
    if counts:  # parser-version migrations may yield FEWER turns -> drop orphaned legacy tails
        _drop_stale_turns(store, log, counts)
    log.info("Earnings-call embeddings: +%d turn rows -> '%s'.", n_new, Tables.earning_calls_embedding)
    return earliest


def embedding_kpis_streamed(
    context: Context,
    call_identity: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Derive embedding KPIs in bounded issuer-sized batches.

    The caller supplies the already validated call-to-issuer map, so predecessor calls under an
    old symbol remain comparable to the first call under a new symbol. The ticker-only fallback
    is retained for isolated tests and unseeded stores.
    """
    store = context.store
    kparts = []
    groups: list[tuple[str | None, list[str]]] = []
    if call_identity is not None and not call_identity.empty:
        groups = [
            (str(issuer_id), sorted(group["ticker"].astype(str).unique())) for issuer_id, group in call_identity.groupby("issuer_id", sort=False)
        ]
    if not groups:
        groups = [(None, [str(ticker)]) for ticker in store.distinct(Tables.earning_calls_embedding, "ticker")]

    for issuer_id, tickers in groups:
        emb = store.load(Tables.earning_calls_embedding, _KPI_LOAD_COLS, where={"ticker": tickers}, optional=True)
        if emb is None:
            continue
        emb = emb[emb["model"] == EARNINGS_CALL_EMBEDDING_CACHE_MODEL]
        if emb.empty:
            continue
        if issuer_id is not None:
            calls = call_identity[call_identity["issuer_id"].astype(str).eq(issuer_id)]
            if calls.empty:
                continue
            emb = emb.merge(calls[["ticker", "quarter", "issuer_id"]], on=["ticker", "quarter"], how="inner")
        k = build_embedding_kpis(emb)
        if k is not None and not k.empty:
            kparts.append(k)
    if not kparts:
        return None
    return pd.concat(kparts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Stage 2: cheap per-build KPIs (coherence + quarter-to-quarter drift)         #
# --------------------------------------------------------------------------- #
def _pooled_section_vectors(turns: pd.DataFrame, section: str) -> pd.DataFrame:
    """Mean-pool a call's turn embeddings for `section` -> one vector per call."""
    identity = "issuer_id" if "issuer_id" in turns.columns else "ticker"
    columns = ["ticker", "quarter", "as_of", "embedding"]
    columns += [column for column in ("model", "issuer_id") if column in turns.columns]
    s = turns[turns["section"] == section][columns]
    if s.empty:
        return pd.DataFrame(columns=["ticker", "quarter", "as_of", "vec"])
    out = []
    group_columns = ["issuer_id", "ticker", "quarter"] if identity == "issuer_id" else ["ticker", "quarter"]
    for key, g in s.groupby(group_columns, sort=False):
        if identity == "issuer_id":
            issuer, tkr, q = key
        else:
            tkr, q = key
            issuer = tkr
        vecs = [np.asarray(v, dtype="float64") for v in g["embedding"]]
        out.append(
            {
                identity: issuer,
                "ticker": tkr,
                "quarter": q,
                "as_of": g["as_of"].iloc[0],
                "model": g["model"].iloc[0] if "model" in g else None,
                "vec": np.mean(vecs, axis=0),
            }
        )
    return pd.DataFrame(out)


def _qq_distance(turns: pd.DataFrame, section: str, name: str) -> pd.DataFrame:
    """Cosine distance to the immediately preceding fiscal quarter only."""
    pooled = _pooled_section_vectors(turns, section)
    if pooled.empty:
        return pd.DataFrame(columns=["ticker", "quarter", name])
    pooled["as_of"] = pd.to_datetime(pooled["as_of"])
    identity = "issuer_id" if "issuer_id" in pooled.columns else "ticker"
    pooled = pooled.sort_values([identity, "as_of"])
    out = []
    for _, grp in pooled.groupby(identity, sort=False):
        previous = None
        for r in grp.itertuples(index=False):
            distance = np.nan
            if previous is not None:
                consecutive = _quarter_number(r.quarter) - _quarter_number(previous.quarter) == 1
                comparable = r.model == previous.model and len(r.vec) == len(previous.vec)
                if consecutive and comparable:
                    distance = round(1.0 - cosine(cast(np.ndarray, r.vec), cast(np.ndarray, previous.vec)), 6)
            out.append({"ticker": r.ticker, "quarter": r.quarter, name: distance})
            previous = r
    return pd.DataFrame(out)


def _quarter_number(value: object) -> float:
    text = str(value)
    if len(text) == 6 and text[4] == "Q" and text[:4].isdigit() and text[5] in "1234":
        return int(text[:4]) * 4 + int(text[5]) - 1
    return np.nan


def _comparable_embedding_calls(turns: pd.DataFrame) -> pd.DataFrame:
    """Keep calls with one model, one positive finite vector dimension, and no zero vectors."""
    valid: list[tuple[object, object]] = []
    for key, group in turns.groupby(["ticker", "quarter"], sort=False):
        vectors = [np.asarray(value, dtype="float64") for value in group["embedding"]]
        model_values = group["model"] if "model" in group else pd.Series(dtype="object")
        models = model_values.dropna().astype(str).nunique()
        models_ok = len(model_values) == len(group) and model_values.notna().all() and models == 1
        dimensions = {vector.size for vector in vectors}
        vectors_ok = bool(vectors) and all(vector.size > 0 and np.isfinite(vector).all() and np.linalg.norm(vector) > 0 for vector in vectors)
        if models_ok and len(dimensions) == 1 and vectors_ok:
            valid.append(key)
    if not valid:
        return turns.iloc[0:0].copy()
    keys = pd.MultiIndex.from_tuples(valid, names=["ticker", "quarter"])
    current = pd.MultiIndex.from_frame(turns[["ticker", "quarter"]])
    return turns[current.isin(keys)].copy()


def _qa_coherence(turns: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, quarter):
    * ec_qa_coherence_mean — for each exchange, the AVERAGE cosine of the question vector
      (mean of its question turns) vs EACH of its answer turns individually (how directly EVERY
      answering exec addresses the question); the quarter KPI is the mean of those
      per-exchange averages. At least two mapped exchanges are required."""
    cols = ["ticker", "quarter", "as_of", "ec_qa_coherence_mean"]
    qa = turns[turns["section"] == _QA_TAG]
    if qa.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (tkr, q), g in qa.groupby(["ticker", "quarter"], sort=False):
        per_ex = []  # one average cosine per exchange
        for _, ge in g.groupby("exchange_idx", sort=False):
            qv = [np.asarray(v, "float64") for v in ge.loc[ge["tag"] == EARNINGS_CALL_TAG_QUESTION, "embedding"]]
            av = [np.asarray(v, "float64") for v in ge.loc[ge["tag"] == EARNINGS_CALL_TAG_ANSWER, "embedding"]]
            if not qv or not av:
                continue
            qvec = np.mean(qv, axis=0)
            per_ex.append(float(np.mean([cosine(qvec, a) for a in av])))  # Q vs EACH answer, averaged
        rows.append(
            {
                "ticker": tkr,
                "quarter": q,
                "as_of": g["as_of"].iloc[0],
                "ec_qa_coherence_mean": round(float(np.mean(per_ex)), 6) if len(per_ex) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_embedding_kpis(embeddings: pd.DataFrame | None) -> pd.DataFrame | None:
    """Per (ticker, quarter) earnings-call embedding KPIs derived from the turn rows:
    Q&A coherence plus consecutive-quarter Q&A/prepared-text distances."""
    if embeddings is None or embeddings.empty:
        return None
    keys = embeddings[["ticker", "quarter"]].drop_duplicates()
    comparable = _comparable_embedding_calls(embeddings)
    kpi = keys.merge(_qa_coherence(comparable), on=["ticker", "quarter"], how="left")
    for section, name in ((_QA_TAG, "ec_qa_qq_distance"), (_PREP_TAG, "ec_prep_qq_distance")):
        kpi = kpi.merge(_qq_distance(comparable, section, name), on=["ticker", "quarter"], how="left")
    kpi = kpi.drop(columns=["as_of"], errors="ignore")
    return kpi
