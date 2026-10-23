"""Embedding output validation and per-call coherence/narrative-distance mathematics."""

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.text.earnings_call_preparation import quarter_number
from src.gpt_extract import cosine
from src.utils.text.earnings_call_split import CALL_SECTIONS


def completed_embedding_calls(metadata: pd.DataFrame, cache_model: str, source: pd.DataFrame) -> set[tuple[str, str]]:
    """Observable compatible provenance and contiguous saved seq keys, not terminal-tail proof."""
    done: set[tuple[str, str]] = set()
    if metadata.empty:
        return done
    known = metadata.merge(source, on=["ticker", "quarter"], suffixes=("", "_source"))
    known["as_of"] = pd.to_datetime(known["as_of"], format="ISO8601", errors="coerce")
    for key, group in known.groupby(["ticker", "quarter"], sort=False):
        seq = pd.to_numeric(group["seq"], errors="coerce")
        compatible = group["model"].eq(cache_model).all() and group["as_of"].eq(group["as_of_source"]).all()
        contiguous = seq.notna().all() and seq.min() == 0 and seq.max() == len(group) - 1 and seq.nunique() == len(group)
        if compatible and contiguous:
            done.add((str(key[0]), str(key[1])))
    return done


def build_embedding_rows(call: pd.Series, vectors: np.ndarray, cache_model: str) -> pd.DataFrame:
    """Validate a whole call's external vectors before constructing durable turn rows."""
    values = np.asarray(vectors, dtype="float64")
    turns = call["turns"]
    if (
        values.ndim != 2
        or len(values) != len(turns)
        or values.shape[1] == 0
        or not np.isfinite(values).all()
        or (np.linalg.norm(values, axis=1) <= 0).any()
    ):
        raise ValueError("Embedding output must contain one finite, nonzero vector of one dimension per turn")
    run_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return pd.DataFrame(
        [
            {
                **turn,
                "ticker": call.ticker,
                "quarter": call.quarter,
                "seq": index,
                "as_of": call.as_of,
                "embedding": vector.tolist(),
                "model": cache_model,
                "run_at": run_at,
            }
            for index, (turn, vector) in enumerate(zip(turns, values, strict=True))
        ]
    )


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
        calls = list(grp.itertuples(index=False))
        for r, previous in zip(calls, [None, *calls[:-1]], strict=True):
            out.append({"ticker": r.ticker, "quarter": r.quarter, name: _consecutive_distance(r, previous)})
    return pd.DataFrame(out)


def _consecutive_distance(current: Any, previous: Any) -> float:
    """1 - cosine to the previous call of the issuer when it is the immediately preceding fiscal
    quarter with the same model and vector size; NaN otherwise."""
    if previous is None:
        return np.nan
    consecutive = quarter_number(current.quarter) - quarter_number(previous.quarter) == 1
    comparable = current.model == previous.model and len(current.vec) == len(previous.vec)
    if not (consecutive and comparable):
        return np.nan
    return round(1.0 - cosine(cast(np.ndarray, current.vec), cast(np.ndarray, previous.vec)), 6)


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
            valid.append((key[0], key[1]))
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
    qa = turns[turns["section"] == CALL_SECTIONS[1]]
    if qa.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (tkr, q), g in qa.groupby(["ticker", "quarter"], sort=False):
        per_ex = []  # one average cosine per exchange
        for _, ge in g.groupby("exchange_idx", sort=False):
            qv = [np.asarray(v, "float64") for v in ge.loc[ge["tag"] == "question", "embedding"]]
            av = [np.asarray(v, "float64") for v in ge.loc[ge["tag"] == "answer", "embedding"]]
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


def build_embedding_kpis(embeddings: pd.DataFrame, comparison_keys: set[tuple[str, str]]) -> pd.DataFrame:
    """Per (ticker, quarter) earnings-call embedding KPIs derived from the turn rows:
    Q&A coherence plus consecutive-quarter Q&A/prepared/question-text distances."""
    prepared_section, qa_section = CALL_SECTIONS
    if embeddings.empty:
        return pd.DataFrame(columns=["ticker", "quarter"])
    keys = embeddings[["ticker", "quarter"]].drop_duplicates()
    comparable = _comparable_embedding_calls(embeddings)
    kpi = keys.merge(_qa_coherence(comparable), on=["ticker", "quarter"], how="left")
    pooled = comparable[pd.MultiIndex.from_frame(comparable[["ticker", "quarter"]]).isin(comparison_keys)]
    for section, name in ((qa_section, "ec_qa_qq_distance"), (prepared_section, "ec_prep_qq_distance")):
        kpi = kpi.merge(_qq_distance(pooled, section, name), on=["ticker", "quarter"], how="left")
    questions = pooled[pooled["tag"].eq("question")]
    kpi = kpi.merge(_qq_distance(questions, qa_section, "ec_questions_only_qq_distance"), on=["ticker", "quarter"], how="left")
    kpi = kpi.drop(columns=["as_of"], errors="ignore")
    return kpi
