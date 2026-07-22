"""
earnings_call_embeddings.py  (src/data_aggregate/utils/earnings_call_embeddings.py)
-----------------------------------------------------------------------------------
OpenAI-embedding layer for earnings calls, on top of the parsed `earnings_call_sections`.

Two stages, mirroring the FinBERT sentiment pipeline:

  1. embed_earnings_calls(context)  — the EXPENSIVE, cached, incremental OpenAI pass. For every
     not-yet-embedded (ticker, quarter) it: splits the Q&A into (question, answer) exchanges,
     embeds each question and answer, and derives
         * n_qa                     — number of analyst Q&A exchanges
         * qa_cos_mean / qa_cos_std — mean & std of cosine(question, answer): how DIRECTLY /
           consistently management answers what was asked (low / erratic = evasive-answer tell)
     It also stores the section-level mean-pooled embeddings (qa, prepared_remarks) so the cheap
     build step can measure quarter-to-quarter drift. Rows -> `earning_calls_embedding` (one per
     ticker·quarter·section) with the model name + run timestamp. Incremental & per-call upsert,
     so an interrupted (billed) run never loses work and re-runs make ZERO API calls.

  2. build_embedding_kpis(embeddings) — the CHEAP per-build derivation of point-in-time KPIs:
         * ec_qa_coherence_mean / _std, ec_n_qa                       (from stage 1)
         * ec_qa_qq_sim   — cosine(this quarter's Q&A embedding, prior quarter's)   (narrative
         * ec_prep_qq_sim — cosine(this quarter's prepared-remarks embedding, prior)  drift: low
           = a story that changed quarter-on-quarter; high = boilerplate repetition)
     These merge into the earnings-call feature panel and become `f_ec_*_{xs,vs_peers}`.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    EARNINGS_CALL_EMBED_MODEL,
    EARNINGS_CALL_EMBEDDING_TABLE,
    EARNINGS_CALL_SECTIONS_TABLE,
)
from src.context import Context
from src.utils.openai_embeddings import cosine, embed_texts, openai_api_key

_QA_TAG, _PREP_TAG = "qa", "prepared_remarks"
_OPERATOR = re.compile(r"^\s*operator\b", re.I)
# a Motley-Fool/HF speaker header line: "Name -- Role/Firm" (kept short so prose lines don't match)
_SPEAKER = re.compile(r"^\s*([A-Z][\w.'\-]+(?:\s+[A-Z][\w.'\-]+){0,4})\s+--\s+(.+?)\s*$")
# role/affiliation words that mark the speaker as a sell-side ANALYST (asks questions)
_ANALYST = re.compile(r"analyst|research|securities|capital|\bbank\b|partners|equity|& ?co|"
                      r"markets|advisors|asset manage|investment", re.I)
_MIN_TURN = 25                                   # chars: ignore "thanks"/"good morning" fragments


def split_qa_pairs(qa_text: str, min_len: int = _MIN_TURN) -> list[tuple[str, str]]:
    """Split a Q&A section blob into (question, answer) exchanges. Speaker turns are detected from
    'Name -- Role' headers (and 'Operator'); each analyst turn is the QUESTION and the following
    management turn(s), up to the next analyst/operator, are its ANSWER. Robust to missing
    structure: returns [] if no analyst/answer turns are found."""
    if not qa_text:
        return []
    turns: list[tuple[str, str]] = []            # (kind, text); kind in {op, q, a}
    kind, buf = None, []

    def _flush():
        if kind and buf:
            t = " ".join(buf).strip()
            if t:
                turns.append((kind, t))

    for line in qa_text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if _OPERATOR.match(s):
            _flush(); kind, buf = "op", []
            continue
        m = _SPEAKER.match(s)
        if m and len(s) < 120:                   # a new speaker header -> new turn
            _flush()
            kind = "q" if _ANALYST.search(m.group(2)) else "a"
            buf = []
            continue
        buf.append(s)
    _flush()

    pairs, i = [], 0
    while i < len(turns):
        if turns[i][0] == "q":
            q = turns[i][1]; ans, j = [], i + 1
            while j < len(turns) and turns[j][0] == "a":
                ans.append(turns[j][1]); j += 1
            a = " ".join(ans).strip()
            if len(q) >= min_len and len(a) >= min_len:
                pairs.append((q, a))
            i = j
        else:
            i += 1
    return pairs


def _call_row(ticker: str, quarter: str, as_of, section: str, emb: np.ndarray, model: str,
              run_at: str, n_qa=None, cos_mean=None, cos_std=None) -> dict:
    return {"ticker": ticker, "quarter": quarter, "section": section, "as_of": as_of,
            "embedding": [float(x) for x in emb], "n_qa": n_qa, "qa_cos_mean": cos_mean,
            "qa_cos_std": cos_std, "model": model, "run_at": run_at}


def embed_earnings_calls(context: Context, sections: pd.DataFrame | None = None,
                         model: str = EARNINGS_CALL_EMBED_MODEL, force: bool = False,
                         client=None) -> pd.DataFrame | None:
    """Ensure every call has cached OpenAI embeddings + Q&A-coherence stats in
    `earning_calls_embedding`. Incremental (skips ticker·quarter already embedded), per-call upsert.
    `client` lets tests inject a stub embedder (no network / no spend). Returns the full table."""
    
    store, log = context.store, context.log
    if sections is None:
        sections = store.load(EARNINGS_CALL_SECTIONS_TABLE)
    if sections is None or sections.empty:
        log.warning("No earnings_call_sections -> embedding skipped (run fetch_earnings_calls).")
        return None
    if client is None and not openai_api_key():
        log.warning("OPENAI/OPEN_AI_API_KEY not set -> earnings-call embedding skipped.")
        return store.load(EARNINGS_CALL_EMBEDDING_TABLE)

    sec = sections[sections["tag"].isin([_QA_TAG, _PREP_TAG])].copy()
    text = sec.pivot_table(index=["ticker", "quarter"], columns="tag", values="text", aggfunc="first")
    as_of = sec.groupby(["ticker", "quarter"])["as_of"].first()

    existing = store.load(EARNINGS_CALL_EMBEDDING_TABLE)
    done = (set() if existing is None or existing.empty
            else set(map(tuple, existing[["ticker", "quarter"]].drop_duplicates().to_numpy())))
    calls = [k for k in text.index if force or tuple(k) not in done]
    if not calls:
        log.info("Earnings-call embedding cache already complete (%d rows).",
                 0 if existing is None else len(existing))
        return existing

    log.info("Embedding %d earnings calls (OpenAI %s)...", len(calls), model)
    n_new = 0
    for (tkr, q) in tqdm(calls, "EC EMbeddings calls"):
        run_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        qa_text = text.loc[(tkr, q), _QA_TAG] if _QA_TAG in text.columns else None
        prep_text = text.loc[(tkr, q), _PREP_TAG] if _PREP_TAG in text.columns else None
        aod = as_of.get((tkr, q))
        pairs = split_qa_pairs(qa_text) if isinstance(qa_text, str) else []
        texts = [p[0] for p in pairs] + [p[1] for p in pairs]
        if isinstance(prep_text, str) and prep_text.strip():
            texts.append(prep_text)
        if not texts:
            continue
        V = embed_texts(texts, model=model, client=client)
        rows = []
        n = len(pairs)
        if n:
            qv, av = V[:n], V[n:2 * n]
            cos = np.array([cosine(qv[i], av[i]) for i in range(n)])
            qa_emb = np.vstack([qv, av]).mean(axis=0)          # section-level pooled embedding
            rows.append(_call_row(tkr, q, aod, _QA_TAG, qa_emb, model, run_at,
                                  n_qa=int(n), cos_mean=round(float(cos.mean()), 6),
                                  cos_std=round(float(cos.std()), 6)))
        if isinstance(prep_text, str) and prep_text.strip():
            rows.append(_call_row(tkr, q, aod, _PREP_TAG, V[-1], model, run_at))

        if rows:
            store.save(EARNINGS_CALL_EMBEDDING_TABLE, pd.DataFrame(rows))   # per-call upsert
            n_new += len(rows)
    log.info("Earnings-call embeddings: +%d rows -> '%s'.", n_new, EARNINGS_CALL_EMBEDDING_TABLE)
    return store.load(EARNINGS_CALL_EMBEDDING_TABLE)


# --------------------------------------------------------------------------- #
# Stage 2: cheap per-build KPIs (coherence + quarter-to-quarter drift)         #
# --------------------------------------------------------------------------- #
def _qq_similarity(emb: pd.DataFrame, section: str, name: str) -> pd.DataFrame:
    """Per (ticker, quarter): cosine(this quarter's section embedding, the PRIOR quarter's), in
    call order. NaN on a ticker's first call."""
    s = emb[emb["section"] == section][["ticker", "quarter", "as_of", "embedding"]].copy()
    if s.empty:
        return pd.DataFrame(columns=["ticker", "quarter", name])
    s["as_of"] = pd.to_datetime(s["as_of"])
    s = s.sort_values(["ticker", "as_of"])
    out = []
    for tkr, grp in s.groupby("ticker", sort=False):
        prev = None
        for r in grp.itertuples(index=False):
            v = np.asarray(r.embedding, dtype="float64")
            out.append({"ticker": tkr, "quarter": r.quarter,
                        name: np.nan if prev is None else round(cosine(v, prev), 6)})
            prev = v
    return pd.DataFrame(out)


def build_embedding_kpis(embeddings: pd.DataFrame | None) -> pd.DataFrame | None:
    """Per (ticker, quarter) earnings-call embedding KPIs: ec_qa_coherence_mean/_std, ec_n_qa,
    ec_qa_qq_sim, ec_prep_qq_sim. None if the embedding cache is empty."""
    if embeddings is None or embeddings.empty:
        return None
    qa = embeddings[embeddings["section"] == _QA_TAG][
        ["ticker", "quarter", "n_qa", "qa_cos_mean", "qa_cos_std"]].copy()
    kpi = qa.rename(columns={"qa_cos_mean": "ec_qa_coherence_mean",
                             "qa_cos_std": "ec_qa_coherence_std", "n_qa": "ec_n_qa"})
    for section, name in ((_QA_TAG, "ec_qa_qq_sim"), (_PREP_TAG, "ec_prep_qq_sim")):
        kpi = kpi.merge(_qq_similarity(embeddings, section, name), on=["ticker", "quarter"], how="outer")
    for c in ("ec_n_qa", "ec_qa_coherence_mean", "ec_qa_coherence_std"):
        kpi[c] = pd.to_numeric(kpi[c], errors="coerce")
    return kpi
