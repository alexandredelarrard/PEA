"""
earnings_call_embeddings.py  (src/data_aggregate/utils/earnings_call_embeddings.py)
-----------------------------------------------------------------------------------
OpenAI-embedding layer for earnings calls, on top of the parsed `earnings_call_sections`.

Two stages, mirroring the FinBERT sentiment pipeline:

  1. embed_earnings_calls(context)  — the EXPENSIVE, cached, incremental OpenAI pass. For every
     not-yet-embedded (ticker, quarter) it splits each scored section into SPEAKER TURNS and
     embeds each turn's text, writing ONE ROW PER TURN to `earning_calls_embedding`:
         ticker, quarter, seq (turn order in the call), section (qa / prepared_remarks),
         tag (question / answer / prepared), exchange_idx (links a question to its answer turns;
         -1 for prepared), person (the speaker), text (the raw turn), as_of (the call date),
         embedding (the turn's OpenAI vector), model, run_at.
     Storing per turn keeps every question/answer embedding we pay for — auditable and reusable —
     at NO extra API cost vs the old pooled design (same text is embedded either way). Incremental
     & per-call upsert, so an interrupted (billed) run never loses work and re-runs make ZERO calls.

  2. build_embedding_kpis(embeddings) — the CHEAP per-build derivation of point-in-time KPIs
     DERIVED from the turn rows (nothing is precomputed at store time except the vectors):
         * ec_qa_coherence_mean / _std — per exchange, cosine(question turn, mean of its answer
           turns) = how DIRECTLY / consistently management answers; mean & std over exchanges
         * ec_n_qa                     — number of analyst Q&A exchanges
         * ec_qa_qq_sim   — cosine(this quarter's POOLED q&a turns, prior quarter's)   (narrative
         * ec_prep_qq_sim — cosine(this quarter's POOLED prepared turns, prior quarter's)  drift)
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
    EARNINGS_CALL_TAG_ANSWER,
    EARNINGS_CALL_TAG_PREPARED,
    EARNINGS_CALL_TAG_QUESTION,
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


def split_turns(text: str, section: str, min_len: int = _MIN_TURN) -> list[dict]:
    """Split a section blob into SPEAKER TURNS -> [{section, tag, person, text, exchange_idx}].

      * qa: an analyst speaker header starts a QUESTION turn AND a new exchange; a management
        header (or an operator hand-off) starts an ANSWER turn of the current exchange. exchange_idx
        increments on each question so a question pairs with the answer turn(s) that follow it.
      * prepared_remarks: every speaker turn is a 'prepared' turn (exchange_idx = -1).

    Robust to missing speaker structure (returns one turn with person=None). Turns shorter than
    `min_len` (courtesy fragments) are dropped."""
    if not text:
        return []
    is_qa = section == _QA_TAG
    raw: list[tuple[str, str | None, str]] = []       # (kind, person, text)
    # prepared remarks are often prose with no "Name -- Role" header -> seed kind so the leading
    # (headerless) block is still captured as one turn; qa needs an explicit question header.
    kind: str | None = None if is_qa else "prep"
    person: str | None = None
    buf: list[str] = []

    def _flush():
        if kind and buf:
            t = " ".join(buf).strip()
            if t:
                raw.append((kind, person, t))

    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if is_qa and _OPERATOR.match(s):
            _flush(); kind, person, buf = "op", "Operator", []
            continue
        m = _SPEAKER.match(s)
        if m and len(s) < 120:                        # a new speaker header -> new turn
            _flush()
            person = m.group(1).strip()
            kind = ("prep" if not is_qa
                    else ("q" if _ANALYST.search(m.group(2)) else "a"))
            buf = []
            continue
        buf.append(s)
    _flush()

    out: list[dict] = []
    ex = -1
    for knd, per, txt in raw:
        if knd == "op" or len(txt) < min_len:
            continue
        if not is_qa:
            out.append({"section": section, "tag": EARNINGS_CALL_TAG_PREPARED,
                        "person": per, "text": txt, "exchange_idx": -1})
            continue
        if knd == "q":
            ex += 1
            tag = EARNINGS_CALL_TAG_QUESTION
        else:
            tag = EARNINGS_CALL_TAG_ANSWER
        out.append({"section": section, "tag": tag, "person": per, "text": txt,
                    "exchange_idx": ex if ex >= 0 else 0})
    return out


def split_qa_pairs(qa_text: str, min_len: int = _MIN_TURN) -> list[tuple[str, str]]:
    """Backward-compat helper: pair each analyst question with the concatenated management answer
    turns of the same exchange, derived from `split_turns` so the two stay consistent. [] with no
    structure."""
    by_ex: dict[int, dict[str, list[str]]] = {}
    for t in split_turns(qa_text, _QA_TAG, min_len):
        slot = by_ex.setdefault(t["exchange_idx"], {"q": [], "a": []})
        slot["q" if t["tag"] == EARNINGS_CALL_TAG_QUESTION else "a"].append(t["text"])
    pairs: list[tuple[str, str]] = []
    for ex in sorted(by_ex):
        q = " ".join(by_ex[ex]["q"]).strip()
        a = " ".join(by_ex[ex]["a"]).strip()
        if len(q) >= min_len and len(a) >= min_len:
            pairs.append((q, a))
    return pairs


def embed_earnings_calls(context: Context, sections: pd.DataFrame | None = None,
                         model: str = EARNINGS_CALL_EMBED_MODEL, force: bool = False,
                         client=None) -> pd.DataFrame | None:
    """Ensure every call's speaker turns are embedded + cached in `earning_calls_embedding`
    (one row per turn). Incremental (skips ticker·quarter already embedded), per-call upsert.
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
        log.info("Earnings-call embedding cache already complete (%d turn rows).",
                 0 if existing is None else len(existing))
        return existing

    log.info("Embedding %d earnings calls per-turn (OpenAI %s)...", len(calls), model)
    n_new = 0
    for (tkr, q) in tqdm(calls, "EC EMbeddings calls"):
        run_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        qa_text = text.loc[(tkr, q), _QA_TAG] if _QA_TAG in text.columns else None
        prep_text = text.loc[(tkr, q), _PREP_TAG] if _PREP_TAG in text.columns else None
        aod = as_of.get((tkr, q))
        turns: list[dict] = []
        if isinstance(prep_text, str) and prep_text.strip():
            turns += split_turns(prep_text, _PREP_TAG)
        if isinstance(qa_text, str) and qa_text.strip():
            turns += split_turns(qa_text, _QA_TAG)
        if not turns:
            continue
        V = embed_texts([t["text"] for t in turns], model=model, client=client)
        rows = [{"ticker": tkr, "quarter": q, "seq": i, "section": t["section"], "tag": t["tag"],
                 "exchange_idx": int(t["exchange_idx"]), "person": t["person"], "text": t["text"],
                 "as_of": aod, "embedding": [float(x) for x in V[i]], "model": model,
                 "run_at": run_at}
                for i, t in enumerate(turns)]
        store.save(EARNINGS_CALL_EMBEDDING_TABLE, pd.DataFrame(rows))    # per-call upsert
        n_new += len(rows)
    log.info("Earnings-call embeddings: +%d turn rows -> '%s'.", n_new, EARNINGS_CALL_EMBEDDING_TABLE)
    return store.load(EARNINGS_CALL_EMBEDDING_TABLE)


# --------------------------------------------------------------------------- #
# Stage 2: cheap per-build KPIs (coherence + quarter-to-quarter drift)         #
# --------------------------------------------------------------------------- #
def _pooled_section_vectors(turns: pd.DataFrame, section: str) -> pd.DataFrame:
    """Mean-pool a call's turn embeddings for `section` -> one vector per (ticker, quarter)."""
    s = turns[turns["section"] == section][["ticker", "quarter", "as_of", "embedding"]]
    if s.empty:
        return pd.DataFrame(columns=["ticker", "quarter", "as_of", "vec"])
    out = []
    for (tkr, q), g in s.groupby(["ticker", "quarter"], sort=False):
        vecs = [np.asarray(v, dtype="float64") for v in g["embedding"]]
        out.append({"ticker": tkr, "quarter": q, "as_of": g["as_of"].iloc[0],
                    "vec": np.mean(vecs, axis=0)})
    return pd.DataFrame(out)


def _qq_similarity(turns: pd.DataFrame, section: str, name: str) -> pd.DataFrame:
    """Per (ticker, quarter): cosine(this call's POOLED section vector, the PRIOR call's), in
    call order. NaN on a ticker's first call."""
    P = _pooled_section_vectors(turns, section)
    if P.empty:
        return pd.DataFrame(columns=["ticker", "quarter", name])
    P["as_of"] = pd.to_datetime(P["as_of"])
    P = P.sort_values(["ticker", "as_of"])
    out = []
    for tkr, grp in P.groupby("ticker", sort=False):
        prev = None
        for r in grp.itertuples(index=False):
            v = r.vec
            out.append({"ticker": tkr, "quarter": r.quarter,
                        name: np.nan if prev is None else round(cosine(v, prev), 6)})
            prev = v
    return pd.DataFrame(out)


def _qa_coherence(turns: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, quarter): mean/std over exchanges of cosine(question turn, mean of its answer
    turns), plus ec_n_qa (# exchanges that have a question)."""
    qa = turns[turns["section"] == _QA_TAG]
    if qa.empty:
        return pd.DataFrame(columns=["ticker", "quarter", "ec_qa_coherence_mean",
                                     "ec_qa_coherence_std", "ec_n_qa"])
    rows = []
    for (tkr, q), g in qa.groupby(["ticker", "quarter"], sort=False):
        cos_list = []
        for _, ge in g.groupby("exchange_idx", sort=False):
            qv = [np.asarray(v, "float64")
                  for v in ge.loc[ge["tag"] == EARNINGS_CALL_TAG_QUESTION, "embedding"]]
            av = [np.asarray(v, "float64")
                  for v in ge.loc[ge["tag"] == EARNINGS_CALL_TAG_ANSWER, "embedding"]]
            if not qv or not av:
                continue
            cos_list.append(cosine(np.mean(qv, axis=0), np.mean(av, axis=0)))
        n_qa = int(g.loc[g["tag"] == EARNINGS_CALL_TAG_QUESTION, "exchange_idx"].nunique())
        rows.append({"ticker": tkr, "quarter": q,
                     "ec_qa_coherence_mean": round(float(np.mean(cos_list)), 6) if cos_list else np.nan,
                     "ec_qa_coherence_std": round(float(np.std(cos_list)), 6) if cos_list else np.nan,
                     "ec_n_qa": n_qa})
    return pd.DataFrame(rows)


def build_embedding_kpis(embeddings: pd.DataFrame | None) -> pd.DataFrame | None:
    """Per (ticker, quarter) earnings-call embedding KPIs derived from the turn rows:
    ec_qa_coherence_mean/_std, ec_n_qa, ec_qa_qq_sim, ec_prep_qq_sim. None if the cache is empty."""
    if embeddings is None or embeddings.empty:
        return None
    kpi = _qa_coherence(embeddings)
    for section, name in ((_QA_TAG, "ec_qa_qq_sim"), (_PREP_TAG, "ec_prep_qq_sim")):
        kpi = kpi.merge(_qq_similarity(embeddings, section, name), on=["ticker", "quarter"], how="outer")
    for c in ("ec_n_qa", "ec_qa_coherence_mean", "ec_qa_coherence_std"):
        if c in kpi.columns:
            kpi[c] = pd.to_numeric(kpi[c], errors="coerce")
    return kpi
