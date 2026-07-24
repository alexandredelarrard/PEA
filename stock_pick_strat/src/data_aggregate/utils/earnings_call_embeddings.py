"""
earnings_call_embeddings.py  (src/data_aggregate/utils/earnings_call_embeddings.py)
-----------------------------------------------------------------------------------
OpenAI-embedding layer for earnings calls, on top of the parsed `earnings_call_sections`.

Two stages, mirroring the FinBERT sentiment pipeline:

  1. embed_earnings_calls(context)  — the EXPENSIVE, cached, incremental OpenAI pass. For every
     not-yet-embedded (ticker, quarter) it splits each scored section into CLEANED SPEAKER TURNS
     (see split_turns: real Motley-Fool "Name:" headers + legacy "Name -- Role" headers; Q&A
     segmented by the operator hand-off lines; operator / IR-flow / pure-courtesy turns dropped;
     greeting/thanks/congrats preambles stripped so only the meaty question / answer is embedded)
     and embeds each turn's text, writing ONE ROW PER TURN to `earning_calls_embedding`:
         ticker, quarter, seq (turn order in the call), section (qa / prepared_remarks),
         tag (question / answer / prepared), exchange_idx (which Q&A pair; -1 for prepared),
         answer_idx (0 for the question, 1..k for the 1st..last answer turn; -1 for prepared),
         person (the speaker -- the analyst on question rows, the manager on answer rows),
         text (the cleaned turn), as_of (the call date), embedding (the turn's OpenAI vector),
         model, run_at.
     Storing per turn keeps every question/answer embedding we pay for — auditable and reusable —
     at NO extra API cost vs the old pooled design (same text is embedded either way). Incremental
     & per-call upsert, so an interrupted (billed) run never loses work and re-runs make ZERO calls.

  2. build_embedding_kpis(embeddings) — the CHEAP per-build derivation of point-in-time KPIs
     DERIVED from the turn rows (nothing is precomputed at store time except the vectors):
         * ec_qa_coherence_mean / _std — per exchange, the AVERAGE cosine of the question vs EACH of
           its answer turns (how directly every answering exec addresses the question); the quarter
           KPI is the mean & std of those per-exchange averages (average-of-averages / std-of-averages)
         * ec_n_qa            — number of analyst Q&A exchanges
         * ec_n_answers       — number of answer turns
         * ec_qa_answer_ratio — answer turns / question turns (exec voices per question)
         * ec_qa_answer_ratio_qq — quarter-to-quarter change of ec_qa_answer_ratio
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

# ---- speaker headers ------------------------------------------------------- #
# Real (2024+) Motley Fool transcripts head each turn with "Name:" (colon, NO role -- the analyst
# is named in the operator's hand-off line instead); the older HuggingFace backbone uses
# "Name -- Role -- Firm" (dash). Support BOTH so the splitter works on every source.
_NAME = r"[A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+){0,4}"
_SPEAKER_COLON = re.compile(rf"^({_NAME}):\s*(.*)$")
_SPEAKER_DASH = re.compile(rf"^\s*({_NAME})\s+--\s+.+$")
_NAME_ONLY = re.compile(rf"^{_NAME}$")           # a bare name line (multi-line "Name / -- / Role" header)
_MIN_TURN = 25                                   # chars: ignore "thanks"/"good morning" fragments


def _speaker(line: str) -> tuple[str, str] | None:
    """(person, inline_text) if `line` is a speaker header, else None. The colon form keeps any
    prose typed on the same line; a 1-word 'header' followed by prose ('Revenue: ...') is rejected
    as an ordinary sentence, not a speaker."""
    m = _SPEAKER_DASH.match(line)
    if m and len(line) < 120:
        return m.group(1).strip(), ""
    m = _SPEAKER_COLON.match(line)
    if m:
        name, inline = m.group(1).strip(), m.group(2).strip()
        if inline and len(name.split()) < 2 and name.lower() != "operator":
            return None
        if len(name) <= 40:
            return name, inline
    return None


# ---- operator / logistics turns (dropped; they delimit the Q&A exchanges) --- #
_OPERATOR_NAME = re.compile(r"^operator$", re.I)
_HANDOFF = re.compile(
    r"\b(?:next|first|final|last|following)?\s*questions?\s+(?:comes?|is|will\s+come|will\s+be)\s+from\b"
    r"|please\s+(?:go\s+ahead|proceed|stand\s+by)"
    r"|press\s+(?:the\s+)?star|in\s+order\s+to\s+ask\s+a\s+question|poll\s+for\s+questions"
    r"|(?:open|opening)\s+(?:up\s+)?(?:the\s+)?(?:floor|line|lines|call|phone\s+lines)\b[^.]{0,25}?questions?"
    r"|(?:we(?:'ll| will| are)|now|let's|i(?:'ll| will))\b[^.]{0,45}?"
    r"(?:begin|open|take|start|move\s+to|go\s+to|turn\s+[^.]{0,20}?to)[^.]{0,30}?questions?"
    r"|question[-\s]and[-\s]answer\s+session", re.I)
# IR-host flow logistics / call sign-off between or after questions -- NOT content, so dropped:
# "Operator, next question please.", "we have time for one last question", "that wraps up the
# Q&A ... thank you for joining us."
_FLOW = re.compile(
    r"operator[,.\s][^.]{0,30}?(?:next|last|final|one\s+more)\s+question"
    r"|(?:next|last|final|one\s+more)\s+question[,.\s]*please"
    r"|we\s+have\s+time\s+for\b[^.]{0,25}?questions?"
    r"|that\s+(?:wraps?\s+up|concludes?|will\s+(?:wrap|conclude))\b[^.]{0,30}?"
    r"(?:q\s*(?:&|and)\s*a|call|session|portion)"
    r"|this\s+concludes\b|thank(?:s|\s+you)[^.]{0,20}?for\s+joining", re.I)


def _is_operator(person: str | None, text: str) -> bool:
    """Operator hand-off / call-logistics turn: not content, but marks a new exchange."""
    return bool((person and _OPERATOR_NAME.match(person)) or _HANDOFF.search(text) or _FLOW.search(text))


# ---- pleasantry / non-informative cleaning --------------------------------- #
# Tuned on 14 real 2024-2026 transcripts (AAPL / MSFT / NVDA): analysts open with "Thanks for taking
# my question / congrats on the quarter" and close with "that's helpful / appreciate it / back in
# the queue"; management opens each answer with "Thanks, <name>. / Hi, <name>. / Yeah, <name>.".
_SENT = re.compile(r"[^.?!]+[.?!]*")
_PLEASANTRY_CUE = re.compile(
    r"^(?:hi|hey|hello|good\s+(?:morning|afternoon|evening)|morning|afternoon|thanks?|thank\s+you|"
    r"yeah|yep|yes|sure|okay|ok|great|perfect|excellent|terrific|wonderful|got\s+it|congrats?|"
    r"congratulations|awesome|understood|fair\s+enough)\b"
    r"|taking\s+(?:my|the|our|your)\s+questions?|congrat\w*|appreciate\s+it|back\s+in\s+(?:the\s+)?queue"
    r"|look\s+forward|nice\s+(?:quarter|results?)|great\s+(?:quarter|results?|answer|color|stuff)"
    r"|(?:that'?s|thats)\s+(?:helpful|great|all|it|fair)|thanks\s+so\s+much", re.I)
_NONINFO_Q = re.compile(
    r"^(?:do\s+you\s+have\s+(?:any\s+)?(?:other\s+|more\s+)?questions?|are\s+you\s+(?:okay|ok|good)"
    r"|any\s+(?:other|more|further)\s+questions?|no\s+(?:more|further)\s+questions?"
    r"|(?:i'?m|we'?re)\s+all\s+set|thank\s+you)\b", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def _is_pleasantry(sent: str) -> bool:
    """A short courtesy/greeting sentence carrying no substantive question (never drops a '?')."""
    return "?" not in sent and len(sent.split()) <= 14 and bool(_PLEASANTRY_CUE.search(sent))


def _clean(text: str) -> str:
    """Keep the MEATY part of a turn: drop leading and trailing courtesy sentences (greetings,
    thanks, congrats, 'back in the queue', 'thanks <name>' answer lead-ins), collapse whitespace."""
    sents = _sentences(re.sub(r"\s+", " ", text).strip())
    while sents and _is_pleasantry(sents[0]):
        sents.pop(0)
    while sents and _is_pleasantry(sents[-1]):
        sents.pop()
    return " ".join(sents).strip()


def _is_informative_question(text: str) -> bool:
    """A cleaned analyst turn that is a real question (not 'do you have questions', pure thanks)."""
    t = text.strip()
    return len(t) >= 20 and len(t.split()) >= 4 and not _NONINFO_Q.match(t)


def split_turns(text: str, section: str, min_len: int = _MIN_TURN) -> list[dict]:
    """Split a section blob into CLEANED SPEAKER TURNS
    -> [{section, tag, person, text, exchange_idx, answer_idx}].

      * qa: driven by the OPERATOR hand-off lines (not a role regex). Each hand-off opens a new
        exchange; the first speaker after it is the ANALYST (a QUESTION), the following management
        speakers are ANSWERS of that exchange, and the analyst speaking again is a follow-up
        question. Operator / IR-flow / pure-courtesy turns are dropped; every kept turn is cleaned
        to its substantive core. exchange_idx = which Q&A pair; answer_idx = 0 for the question and
        1,2,... for the 1st, 2nd,... answer turn (so 'first vs last answer' is explicit).
      * prepared_remarks: every management turn is 'prepared' (exchange_idx = -1, answer_idx = -1);
        the host's welcome/logistics is dropped.

    Robust to a leading header-less block (seeded as the operator preamble in qa). Turns whose
    cleaned text is shorter than `min_len` are dropped."""
    if not text:
        return []
    is_qa = section == _QA_TAG
    # 1) raw turns by speaker header. Three real header shapes are supported: the newest MF
    # "Name:" (colon), the 2024-2025 MF multi-line "Name / -- / Role", and the legacy inline
    # "Name -- Role -- Firm"; plus bare "Operator" lines. qa opens with the operator preamble.
    person: str | None = "Operator" if is_qa else None
    buf: list[str] = []
    raw: list[tuple[str | None, str]] = []

    def _flush() -> None:
        if buf:
            t = " ".join(buf).strip()
            if t:
                raw.append((person, t))

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    i, n = 0, len(lines)
    while i < n:
        s = lines[i]
        if i + 2 < n and lines[i + 1] == "--" and _NAME_ONLY.match(s):   # "Name / -- / Role" header
            _flush(); person, buf, i = s, [], i + 3
            continue
        if _OPERATOR_NAME.match(s):                                      # bare "Operator" line
            _flush(); person, buf, i = "Operator", [], i + 1
            continue
        sp = _speaker(s)                                                 # colon or inline "Name -- Role"
        if sp:
            _flush(); person, inline = sp; buf = [inline] if inline else []; i += 1
            continue
        buf.append(s); i += 1
    _flush()

    # 2a) prepared remarks -> one 'prepared' turn per management speaker
    if not is_qa:
        out: list[dict] = []
        for per, txt in raw:
            if _is_operator(per, txt):
                continue
            body = _clean(txt)
            if len(body) >= min_len:
                out.append({"section": section, "tag": EARNINGS_CALL_TAG_PREPARED, "person": per,
                            "text": body, "exchange_idx": -1, "answer_idx": -1})
        return out

    # 2b) qa -> exchanges of (question, mapped answers), segmented by operator hand-offs
    out = []
    ex, ans_i, analyst, new_exchange = -1, 0, None, True
    for per, txt in raw:
        if _is_operator(per, txt):
            new_exchange, analyst = True, None
            continue
        if new_exchange:                                     # first speaker after a hand-off = analyst
            ex, ans_i, analyst, new_exchange = ex + 1, 0, per, False
            body = _clean(txt)
            if _is_informative_question(body):
                out.append({"section": section, "tag": EARNINGS_CALL_TAG_QUESTION, "person": per,
                            "text": body, "exchange_idx": ex, "answer_idx": 0})
            continue
        if per is not None and per == analyst:               # analyst again = follow-up question
            body = _clean(txt)
            if _is_informative_question(body):
                out.append({"section": section, "tag": EARNINGS_CALL_TAG_QUESTION, "person": per,
                            "text": body, "exchange_idx": ex, "answer_idx": 0})
            continue
        body = _clean(txt)                                   # management = answer
        if len(body) >= min_len:
            ans_i += 1
            out.append({"section": section, "tag": EARNINGS_CALL_TAG_ANSWER, "person": per,
                        "text": body, "exchange_idx": ex, "answer_idx": ans_i})
    return out


def split_qa_exchanges(qa_text: str, min_len: int = _MIN_TURN) -> list[dict]:
    """The cleaned LIST OF QUESTIONS with their MAPPED ANSWERS, one dict per exchange:
    {exchange_idx, question, analyst, answers:[...], managers:[...]}. Only exchanges that have a
    real (cleaned) question are returned."""
    by_ex: dict[int, dict] = {}
    for t in split_turns(qa_text, _QA_TAG, min_len):
        e = by_ex.setdefault(t["exchange_idx"], {"exchange_idx": t["exchange_idx"], "question": None,
                                                 "analyst": None, "answers": [], "managers": []})
        if t["tag"] == EARNINGS_CALL_TAG_QUESTION:
            e["question"] = f'{e["question"]} {t["text"]}'.strip() if e["question"] else t["text"]
            e["analyst"] = t["person"]
        else:
            e["answers"].append(t["text"]); e["managers"].append(t["person"])
    return [by_ex[k] for k in sorted(by_ex) if by_ex[k]["question"]]


def split_qa_pairs(qa_text: str, min_len: int = _MIN_TURN) -> list[tuple[str, str]]:
    """Backward-compat helper: pair each analyst question with the concatenated management answer
    turns of the same exchange, derived from `split_turns` so the two stay consistent. [] with no
    structure."""
    pairs: list[tuple[str, str]] = []
    for e in split_qa_exchanges(qa_text, min_len):
        q, a = (e["question"] or "").strip(), " ".join(e["answers"]).strip()
        if len(q) >= min_len and len(a) >= min_len:
            pairs.append((q, a))
    return pairs


def _drop_stale_turns(store, log, counts: dict[tuple[str, str], int]) -> None:
    """After a FORCE re-embed, drop rows whose `seq` no longer exists — a re-parse (e.g. after a
    splitter change) can yield FEWER turns than a prior run left cached; upsert-by-PK would leave
    those tail rows orphaned. Reconciles via load+replace (the store has no per-row delete) so the
    table matches the current parse exactly. Only re-embedded calls are touched."""
    full = store.load(EARNINGS_CALL_EMBEDDING_TABLE)
    if full is None or full.empty:
        return
    n_new = pd.Series([counts.get(k, -1) for k in zip(full["ticker"], full["quarter"])], index=full.index)
    seq = pd.to_numeric(full["seq"], errors="coerce").fillna(0)
    keep = (n_new < 0) | (seq < n_new)                   # -1 = call not re-embedded -> keep all its rows
    if (~keep).any():
        store.replace(EARNINGS_CALL_EMBEDDING_TABLE, full[keep].reset_index(drop=True))
        log.info("Force re-embed reconcile: dropped %d stale turn rows.", int((~keep).sum()))


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
    n_new, counts = 0, {}
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
                 "exchange_idx": int(t["exchange_idx"]), "answer_idx": int(t["answer_idx"]),
                 "person": t["person"], "text": t["text"], "as_of": aod,
                 "embedding": [float(x) for x in V[i]], "model": model, "run_at": run_at}
                for i, t in enumerate(turns)]
        store.save(EARNINGS_CALL_EMBEDDING_TABLE, pd.DataFrame(rows))    # per-call upsert
        counts[(tkr, q)] = len(rows)
        n_new += len(rows)
    if force and counts:                     # re-embed may yield FEWER turns -> drop orphaned tail rows
        _drop_stale_turns(store, log, counts)
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
    """Per (ticker, quarter):
      * ec_qa_coherence_mean / _std — for each exchange, the AVERAGE cosine of the question vector
        (mean of its question turns) vs EACH of its answer turns individually (how directly EVERY
        answering exec addresses the question); the quarter KPI is the mean and the std of those
        per-exchange averages (average-of-averages / std-of-averages).
      * ec_n_qa            — # exchanges that have a question
      * ec_n_answers       — # answer turns in the call
      * ec_qa_answer_ratio — answer turns / question turns (how many exec voices pile onto a question)
    Carries as_of so build_embedding_kpis can take the quarter-to-quarter delta of the ratio."""
    cols = ["ticker", "quarter", "as_of", "ec_qa_coherence_mean", "ec_qa_coherence_std",
            "ec_n_qa", "ec_n_answers", "ec_qa_answer_ratio"]
    qa = turns[turns["section"] == _QA_TAG]
    if qa.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (tkr, q), g in qa.groupby(["ticker", "quarter"], sort=False):
        per_ex = []                                          # one average cosine per exchange
        for _, ge in g.groupby("exchange_idx", sort=False):
            qv = [np.asarray(v, "float64")
                  for v in ge.loc[ge["tag"] == EARNINGS_CALL_TAG_QUESTION, "embedding"]]
            av = [np.asarray(v, "float64")
                  for v in ge.loc[ge["tag"] == EARNINGS_CALL_TAG_ANSWER, "embedding"]]
            if not qv or not av:
                continue
            qvec = np.mean(qv, axis=0)
            per_ex.append(float(np.mean([cosine(qvec, a) for a in av])))   # Q vs EACH answer, averaged
        n_q = int((g["tag"] == EARNINGS_CALL_TAG_QUESTION).sum())
        n_a = int((g["tag"] == EARNINGS_CALL_TAG_ANSWER).sum())
        rows.append({"ticker": tkr, "quarter": q, "as_of": g["as_of"].iloc[0],
                     "ec_qa_coherence_mean": round(float(np.mean(per_ex)), 6) if per_ex else np.nan,
                     "ec_qa_coherence_std": round(float(np.std(per_ex)), 6) if per_ex else np.nan,
                     "ec_n_qa": int(g.loc[g["tag"] == EARNINGS_CALL_TAG_QUESTION, "exchange_idx"].nunique()),
                     "ec_n_answers": n_a,
                     "ec_qa_answer_ratio": round(n_a / n_q, 6) if n_q else np.nan})
    return pd.DataFrame(rows)


def _qq_delta(kpi: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    """Per (ticker, quarter): `col` this call MINUS `col` the prior call (call order by as_of).
    NaN on a ticker's first call. Requires an `as_of` column on `kpi`."""
    if kpi.empty or col not in kpi.columns or "as_of" not in kpi.columns:
        return pd.DataFrame(columns=["ticker", "quarter", name])
    d = kpi[["ticker", "quarter", "as_of", col]].copy()
    d["as_of"] = pd.to_datetime(d["as_of"])
    d = d.sort_values(["ticker", "as_of"])
    d[name] = d.groupby("ticker", sort=False)[col].diff()
    return d[["ticker", "quarter", name]]


def build_embedding_kpis(embeddings: pd.DataFrame | None) -> pd.DataFrame | None:
    """Per (ticker, quarter) earnings-call embedding KPIs derived from the turn rows:
    ec_qa_coherence_mean/_std, ec_n_qa, ec_n_answers, ec_qa_answer_ratio, ec_qa_answer_ratio_qq
    (quarter-to-quarter change of the answer/question ratio), ec_qa_qq_sim, ec_prep_qq_sim.
    None if the cache is empty."""
    if embeddings is None or embeddings.empty:
        return None
    kpi = _qa_coherence(embeddings)
    kpi = kpi.merge(_qq_delta(kpi, "ec_qa_answer_ratio", "ec_qa_answer_ratio_qq"),
                    on=["ticker", "quarter"], how="left")
    for section, name in ((_QA_TAG, "ec_qa_qq_sim"), (_PREP_TAG, "ec_prep_qq_sim")):
        kpi = kpi.merge(_qq_similarity(embeddings, section, name), on=["ticker", "quarter"], how="outer")
    kpi = kpi.drop(columns=["as_of"], errors="ignore")          # as_of was only for the QoQ ordering
    for c in ("ec_n_qa", "ec_n_answers", "ec_qa_answer_ratio", "ec_qa_answer_ratio_qq",
              "ec_qa_coherence_mean", "ec_qa_coherence_std"):
        if c in kpi.columns:
            kpi[c] = pd.to_numeric(kpi[c], errors="coerce")
    return kpi
