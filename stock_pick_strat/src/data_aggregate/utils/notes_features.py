"""
notes_features.py  (src/data_aggregate/utils/notes_features.py)
---------------------------------------------------------------
Turn the raw SEC footnote NARRATIVE (`notes_text`: high-signal 10-K/10-Q TextBlocks — commitments &
contingencies, legal matters, revenue recognition, significant accounting policies / use-of-estimates,
concentration risk, going concern) into point-in-time, peer-relative RISK / COMPLIANCE features for
the cube. Mirrors the earnings-call embedding layer.

Stage 1  embed_notes(context)          EXPENSIVE, cached, incremental OpenAI pass. For every
    not-yet-embedded (ticker, filing `adsh`, TextBlock `tag`) it CLEANS the note (de-escapes HTML
    entities, drops cross-references / note headers, collapses whitespace), CHUNKS it, embeds the
    chunks and mean-pools them into ONE vector, and upserts a row to `notes_embedding`
    (ticker, cik, adsh, tag, theme, as_of=filed, ddate, fy, fp, txtlen, n_chunks, embedding, model, run_at).

Stage 2  build_notes_kpis(...)         CHEAP per-build derivation from the cached vectors + text:
    A. notes_drift          filing-to-filing narrative drift = 1 - cos(vec_t, vec_{t-1}) per tag
                            (a stable footnote scores ~0; a rewrite scores high -> escalation/ change).
    B. risk-anchor          [TODO] cosine of the note vs named risk/compliance ANCHOR concepts.
    C. tone / lexicon       [TODO] FinBERT tone + Loughran-McDonald litigious / uncertainty density.
    D. length dynamics      [TODO] log(txtlen_t / txtlen_{t-1}) + first-appearance flag.
    E. peer-relative        [TODO] express every KPI vs peers via build_peer_relative_panel.

Themes (NOTES_THEME_TAGS) map the raw tags to the tracked risk/compliance areas: litigation,
going_concern, revenue_rec, critical_estimates, concentration.
"""
from __future__ import annotations

import datetime as dt
import html
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    NOTES_EMBED_MODEL,
    NOTES_EMBEDDING_TABLE,
    NOTES_RISK_ANCHORS,
    NOTES_TEXT_TABLE,
    NOTES_THEME_TAGS,
)
from src.context import Context
from src.utils.openai_embeddings import cosine, embed_texts, openai_api_key
from src.utils.text_metrics import litigious_ratio, uncertainty_ratio, word_count

# tag -> theme lookup (built once from the constant)
_TAG_THEME: dict[str, str] = {tag: theme for theme, tags in NOTES_THEME_TAGS.items() for tag in tags}


def _themed(embeddings: pd.DataFrame | None) -> pd.DataFrame | None:
    """Keep only rows whose tag is CURRENTLY mapped to a theme and (re)derive `theme` from the tag,
    so the NOTES_THEME_TAGS mapping is authoritative at build time (a tag dropped from a theme, e.g.
    UseOfEstimates, is excluded even if older cache rows carry the stale theme)."""
    if embeddings is None or embeddings.empty:
        return None
    e = embeddings[embeddings["tag"].isin(_TAG_THEME)].copy()
    if e.empty:
        return None
    e["theme"] = e["tag"].map(_TAG_THEME)
    return e

# ---- cleaning ------------------------------------------------------------- #
# Cross-references ("see Note 5B", "(Note 12)") and leading note headers carry no theme signal and
# jitter the embedding filing-to-filing; drop them. The `value` is already de-tagged prose (0% HTML
# tags in the corpus) but ~21% carry HTML entities (&#8217; etc.) -> html.unescape.
_XREF = re.compile(r"(?i)\(?\bsee\s+notes?\s+\d+[A-Za-z]?\b\)?|\(\s*notes?\s+\d+[A-Za-z]?\s*\)")
_NOTE_HDR = re.compile(r"(?im)^\s*note[s]?\s+\d+[A-Za-z]?[.:) ]")
_WS = re.compile(r"\s+")
_MIN_CHARS = 120                 # skip near-empty notes (a stub cross-reference etc.)
_CHUNK_CHARS = 6000              # embed long TextBlocks in windows, then mean-pool


def clean_note(text: str | None) -> str:
    """De-escape HTML entities, strip note-number headers + cross-references, collapse whitespace."""
    if not text:
        return ""
    t = html.unescape(str(text))
    t = _NOTE_HDR.sub(" ", t)
    t = _XREF.sub(" ", t)
    return _WS.sub(" ", t).strip()


def _chunks(text: str, size: int = _CHUNK_CHARS) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [text]


def theme_of(tag: str) -> str | None:
    return _TAG_THEME.get(tag)


# --------------------------------------------------------------------------- #
# Stage 1: incremental, cached OpenAI embedding (one pooled vector per note)    #
# --------------------------------------------------------------------------- #
def embed_notes(context: Context, notes: pd.DataFrame | None = None,
                model: str = NOTES_EMBED_MODEL, tickers: list[str] | None = None,
                force: bool = False, client=None, min_chars: int = _MIN_CHARS) -> pd.DataFrame | None:
    """Ensure every themed footnote is embedded + cached in `notes_embedding` (one pooled vector per
    (ticker, adsh, tag)). Incremental (skips already-embedded), per-ticker upsert so an interrupted
    (billed) run keeps its progress. `client` lets tests inject a stub embedder. Returns the table."""
    store, log = context.store, context.log
    if notes is None:
        notes = store.load(NOTES_TEXT_TABLE)
    if notes is None or notes.empty:
        log.warning("No %s -> notes embedding skipped (run fetch_financial_notes).", NOTES_TEXT_TABLE)
        return None
    if client is None and not openai_api_key():
        log.warning("OPENAI/OPEN_AI_API_KEY not set -> notes embedding skipped.")
        return store.load(NOTES_EMBEDDING_TABLE)

    n = notes[notes["tag"].isin(_TAG_THEME)].copy()
    if tickers is not None:
        n = n[n["ticker"].isin(set(tickers))]
    n["clean"] = n["value"].map(clean_note)
    n = n[n["clean"].str.len() >= min_chars]
    if n.empty:
        log.info("No themed notes with >= %d clean chars to embed.", min_chars)
        return store.load(NOTES_EMBEDDING_TABLE)

    existing = store.load(NOTES_EMBEDDING_TABLE)
    done = (set() if existing is None or existing.empty
            else set(map(tuple, existing[["ticker", "adsh", "tag"]].drop_duplicates().to_numpy())))

    todo_tickers = sorted(n["ticker"].dropna().unique())
    log.info("Embedding notes for %d tickers (OpenAI %s)...", len(todo_tickers), model)
    n_new = 0
    run_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for tkr in tqdm(todo_tickers, "notes embeddings"):
        sub = n[n["ticker"] == tkr]
        metas: list[dict] = []
        chunks: list[str] = []
        owner: list[int] = []
        for (adsh, tag), g in sub.groupby(["adsh", "tag"], sort=False):
            if not force and (tkr, adsh, tag) in done:
                continue
            g = g.sort_values(["ddate", "qtrs"])
            text = " ".join(g["clean"].tolist()).strip()
            cks = _chunks(text)
            last = g.iloc[-1]
            metas.append({"ticker": tkr, "cik": last.get("cik"), "adsh": adsh, "tag": tag,
                          "theme": _TAG_THEME[tag], "as_of": last.get("filed"),
                          "ddate": last.get("ddate"), "fy": last.get("fy"), "fp": last.get("fp"),
                          "txtlen": int(len(text)), "n_chunks": len(cks)})
            base = len(metas) - 1
            for c in cks:
                chunks.append(c); owner.append(base)
        if not metas:
            continue
        V = embed_texts(chunks, model=model, client=client)
        own = np.asarray(owner)
        rows = []
        for i, meta in enumerate(metas):
            pooled = V[own == i].mean(axis=0)
            rows.append({**meta, "embedding": [float(x) for x in pooled],
                         "model": model, "run_at": run_at})
        store.save(NOTES_EMBEDDING_TABLE, pd.DataFrame(rows))       # per-ticker upsert
        n_new += len(rows)
    log.info("Notes embeddings: +%d note vectors -> '%s'.", n_new, NOTES_EMBEDDING_TABLE)
    return store.load(NOTES_EMBEDDING_TABLE)


# --------------------------------------------------------------------------- #
# Stage 2A: filing-to-filing narrative DRIFT                                    #
# --------------------------------------------------------------------------- #
def notes_drift(embeddings: pd.DataFrame | None) -> pd.DataFrame | None:
    """Per (ticker, tag): filing-to-filing narrative drift = 1 - cosine(this filing's note vector,
    the PRIOR filing's), ordered by `filed` date. NaN on a ticker/tag's first filing. A footnote
    that is copied verbatim scores ~0; a genuine rewrite (new litigation, changed policy) scores
    high. Returns long: ticker, tag, theme, adsh, filed, ddate, fy, fp, notes_drift."""
    e = _themed(embeddings)
    if e is None:
        return None
    e["as_of"] = pd.to_datetime(e["as_of"], errors="coerce")
    e = e.sort_values(["ticker", "tag", "as_of", "ddate"])
    rows = []
    for (tkr, tag), g in e.groupby(["ticker", "tag"], sort=False):
        prev = None
        for r in g.itertuples(index=False):
            v = np.asarray(r.embedding, dtype="float64")
            drift = np.nan if prev is None else round(1.0 - cosine(v, prev), 6)
            rows.append({"ticker": tkr, "tag": tag, "theme": r.theme, "adsh": r.adsh,
                         "as_of": r.as_of, "ddate": r.ddate, "fy": r.fy, "fp": r.fp,
                         "notes_drift": drift})
            prev = v
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stage 2B: RISK / COMPLIANCE anchor similarity                                 #
# --------------------------------------------------------------------------- #
def embed_anchors(model: str = NOTES_EMBED_MODEL, client=None) -> dict[str, np.ndarray]:
    """Embed the named risk/compliance anchor phrases (NOTES_RISK_ANCHORS) once -> {name: vector}."""
    names = list(NOTES_RISK_ANCHORS)
    V = embed_texts([NOTES_RISK_ANCHORS[n] for n in names], model=model, client=client)
    return {n: np.asarray(V[i], dtype="float64") for i, n in enumerate(names)}


def notes_risk_anchors(embeddings: pd.DataFrame | None, anchor_vecs: dict[str, np.ndarray] | None = None,
                       model: str = NOTES_EMBED_MODEL, client=None) -> pd.DataFrame | None:
    """Per note: cosine similarity of the note embedding to each risk/compliance ANCHOR concept
    -> columns `risk_<anchor>`. 'How close is this disclosure to a known risk pattern' — its
    QoQ rise (feature E makes it peer-relative) is the escalation signal. Returns
    ticker, adsh, tag, theme, as_of, ddate, fy, fp + one risk_<anchor> column per anchor."""
    e = _themed(embeddings)
    if e is None:
        return None
    if anchor_vecs is None:
        anchor_vecs = embed_anchors(model, client)
    rows = []
    for r in e.itertuples(index=False):
        v = np.asarray(r.embedding, dtype="float64")
        d = {"ticker": r.ticker, "adsh": r.adsh, "tag": r.tag, "theme": r.theme,
             "as_of": r.as_of, "ddate": r.ddate, "fy": r.fy, "fp": r.fp}
        for name, av in anchor_vecs.items():
            d[f"risk_{name}"] = round(cosine(v, av), 6)
        rows.append(d)
    return pd.DataFrame(rows)


def notes_text_metrics(notes: pd.DataFrame | None, tickers: list[str] | None = None,
                       min_chars: int = _MIN_CHARS) -> pd.DataFrame | None:
    """Per themed note: Loughran-McDonald LITIGIOUS + UNCERTAINTY word-density and word count on the
    CLEANED text (deterministic, no GPU). Rising litigious density in a contingencies/legal note =
    escalating legal risk; uncertainty density = hedging / doubt. Returns ticker, adsh, tag, theme,
    as_of, ddate, fy, fp, n_words, litigious_ratio, uncertainty_ratio."""
    if notes is None or notes.empty:
        return None
    n = notes[notes["tag"].isin(_TAG_THEME)].copy()
    if tickers is not None:
        n = n[n["ticker"].isin(set(tickers))]
    n["clean"] = n["value"].map(clean_note)
    n = n[n["clean"].str.len() >= min_chars]
    if n.empty:
        return None
    rows = []
    for (tkr, adsh, tag), g in n.groupby(["ticker", "adsh", "tag"], sort=False):
        g = g.sort_values(["ddate", "qtrs"])
        text = " ".join(g["clean"].tolist())
        last = g.iloc[-1]
        rows.append({"ticker": tkr, "adsh": adsh, "tag": tag, "theme": _TAG_THEME[tag],
                     "as_of": last.get("filed"), "ddate": last.get("ddate"),
                     "fy": last.get("fy"), "fp": last.get("fp"), "n_words": word_count(text),
                     "litigious_ratio": round(litigious_ratio(text), 6),
                     "uncertainty_ratio": round(uncertainty_ratio(text), 6)})
    return pd.DataFrame(rows)


def notes_length_delta(text_metrics: pd.DataFrame | None) -> pd.DataFrame | None:
    """Per (ticker, tag): filing-to-filing disclosure-length change = log(n_words_t / n_words_{t-1}),
    plus `notes_is_new` (1 on this ticker's FIRST filing of the tag -> e.g. a going-concern or
    concentration note APPEARING). A sharp expansion of a litigation/contingency note usually means
    new or worsening exposure (firms add words when bad news arrives). Feature C's litigious DENSITY
    can fall when descriptive narrative is added, so this length signal complements it. Returns
    ticker, tag, theme, adsh, as_of, ddate, fy, fp, n_words, notes_len_delta, notes_is_new."""
    if text_metrics is None or text_metrics.empty:
        return None
    m = text_metrics.copy()
    m["as_of"] = pd.to_datetime(m["as_of"], errors="coerce")
    m = m.sort_values(["ticker", "tag", "as_of", "ddate"])
    rows = []
    for (tkr, tag), g in m.groupby(["ticker", "tag"], sort=False):
        prev, first = None, True
        for r in g.itertuples(index=False):
            nw = float(r.n_words) if r.n_words and r.n_words > 0 else np.nan
            ld = (np.nan if prev is None or not np.isfinite(prev) or prev <= 0 or not np.isfinite(nw)
                  else round(float(np.log(nw / prev)), 6))
            rows.append({"ticker": tkr, "tag": tag, "theme": r.theme, "adsh": r.adsh,
                         "as_of": r.as_of, "ddate": r.ddate, "fy": r.fy, "fp": r.fp,
                         "n_words": r.n_words, "notes_len_delta": ld, "notes_is_new": int(first)})
            prev, first = nw, False
    return pd.DataFrame(rows)


def notes_risk_panel(anchor_scores: pd.DataFrame | None) -> pd.DataFrame | None:
    """Collapse per-note anchor scores to ONE row per filing (ticker, adsh, as_of): the MAX cosine
    to each anchor over the filing's notes (the closest-to-archetype disclosure, wherever it sits)."""
    if anchor_scores is None or anchor_scores.empty:
        return None
    risk_cols = [c for c in anchor_scores.columns if c.startswith("risk_")]
    g = (anchor_scores.groupby(["ticker", "adsh"], sort=False)
         .agg({**{c: "max" for c in risk_cols}, "as_of": "first"}).reset_index())
    return g
