"""Embedding output validation and per-call coherence/narrative-distance mathematics."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator
from typing import Any

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
    # Match saved metadata to actual source calls before testing completion.
    known = metadata.merge(source, on=["ticker", "quarter"], suffixes=("", "_source"))
    known["as_of"] = pd.to_datetime(known["as_of"], format="ISO8601", errors="coerce")
    # A complete call has compatible provenance and one contiguous sequence.
    for key, group in known.groupby(["ticker", "quarter"], sort=False):
        seq = pd.to_numeric(group["seq"], errors="coerce")
        compatible = group["model"].eq(cache_model).all() and group["as_of"].eq(group["as_of_source"]).all()
        contiguous = seq.notna().all() and seq.min() == 0 and seq.max() == len(group) - 1 and seq.nunique() == len(group)
        if compatible and contiguous:
            done.add((str(key[0]), str(key[1])))
    return done


def build_embedding_rows(call: pd.Series, vectors: np.ndarray, cache_model: str) -> pd.DataFrame:
    """Validate a whole call's external vectors before constructing durable turn rows."""
    # Reject malformed external model output before saving any turn.
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
    # Stamp the complete call with its model identity and inference time.
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

    # Pool only the requested section of already validated, identified turns.
    selected = turns[turns["section"].eq(section)]
    if selected.empty:
        return pd.DataFrame(columns=["ticker", "quarter", "as_of", "vec"])
    out = []
    for (issuer, ticker, quarter), call in selected.groupby(["issuer_id", "ticker", "quarter"], sort=False):
        out.append(
            {
                "issuer_id": issuer,
                "ticker": ticker,
                "quarter": quarter,
                "as_of": call["as_of"].iloc[0],
                "model": call["model"].iloc[0],
                "vec": np.mean(call["embedding"].tolist(), axis=0),
            }
        )
    return pd.DataFrame(out)


def _qq_distance(pooled: pd.DataFrame, name: str) -> pd.DataFrame:
    """Cosine distance to the immediately preceding fiscal quarter only."""
    if pooled.empty:
        return pd.DataFrame(columns=["ticker", "quarter", name])

    # Compare chronological issuer history, including supported symbol changes.
    pooled["as_of"] = pd.to_datetime(pooled["as_of"])
    pooled = pooled.sort_values(["issuer_id", "as_of"])
    out = []
    for _, grp in pooled.groupby("issuer_id", sort=False):
        calls = list(grp.itertuples(index=False))
        for r, previous in zip(calls, [None, *calls[:-1]], strict=True):
            out.append({"ticker": r.ticker, "quarter": r.quarter, name: _consecutive_distance(r, previous)})
    return pd.DataFrame(out)


def _consecutive_distance(current: Any, previous: Any) -> float:
    """1 - cosine to the previous call of the issuer when it is the immediately preceding fiscal
    quarter with the same model and vector size; NaN otherwise."""
    if previous is None:
        return np.nan

    # Fiscal adjacency and matching model/dimension are both required.
    consecutive = quarter_number(current.quarter) - quarter_number(previous.quarter) == 1
    comparable = current.model == previous.model and len(current.vec) == len(previous.vec)
    if not (consecutive and comparable):
        return np.nan
    return round(1.0 - cosine(current.vec, previous.vec), 6)


def _comparable_embedding_calls(turns: pd.DataFrame) -> pd.DataFrame:
    """Keep calls with one model, one positive finite vector dimension, and no zero vectors."""

    # Normalize stored arrays once; all downstream mathematics reuses them.
    turns = turns.copy()
    turns["embedding"] = turns["embedding"].map(lambda value: np.asarray(value, dtype="float64"))
    valid: list[tuple[object, object]] = []

    # Validate each complete call before pooling any of its turns.
    for key, group in turns.groupby(["ticker", "quarter"], sort=False):
        vectors = group["embedding"].tolist()
        model_values = group["model"]
        models_ok = model_values.notna().all() and model_values.nunique() == 1
        dimensions = {vector.size for vector in vectors}
        vectors_ok = all(vector.size > 0 and np.isfinite(vector).all() and np.linalg.norm(vector) > 0 for vector in vectors)
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

    # Average question-to-answer similarity within each mapped exchange.
    rows = []
    for (tkr, q), g in qa.groupby(["ticker", "quarter"], sort=False):
        per_ex = []  # one average cosine per exchange
        for _, ge in g.groupby("exchange_idx", sort=False):
            qv = ge.loc[ge["tag"].eq("question"), "embedding"].tolist()
            av = ge.loc[ge["tag"].eq("answer"), "embedding"].tolist()
            if not qv or not av:
                continue
            qvec = np.mean(qv, axis=0)
            per_ex.append(float(np.mean([cosine(qvec, a) for a in av])))  # Q vs EACH answer, averaged

        # A call needs at least two usable exchanges for its coherence KPI.
        rows.append(
            {
                "ticker": tkr,
                "quarter": q,
                "as_of": g["as_of"].iloc[0],
                "ec_qa_coherence_mean": round(float(np.mean(per_ex)), 6) if len(per_ex) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def feature_embedding_chunks(chunks: Iterable[pd.DataFrame], scope: pd.DataFrame, pending: set[tuple[str, str]]) -> Iterator[pd.DataFrame]:
    """Keep matching source dates and identified calls, excluding incomplete saved outputs."""

    # Apply the same call/date contract to every chunk without retaining turn history.
    calls = scope[["ticker", "quarter", "as_of", "issuer_id"]]
    for vectors in chunks:
        vectors = vectors[~pd.MultiIndex.from_frame(vectors[["ticker", "quarter"]]).isin(pending)]
        vectors["as_of"] = pd.to_datetime(vectors["as_of"], format="ISO8601", errors="coerce")
        vectors = vectors.merge(calls, on=["ticker", "quarter", "as_of"])
        yield vectors


def _complete_embedding_calls(chunks: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Yield complete calls from a ticker/quarter-ordered stream, carrying its final call."""
    tail = pd.DataFrame()

    # Carry the last call until the next chunk establishes its end.
    for chunk in chunks:
        if chunk.empty:
            continue
        block = pd.concat([tail, chunk], ignore_index=True) if not tail.empty else chunk
        last = block.iloc[-1]
        unfinished = block["ticker"].eq(last["ticker"]) & block["quarter"].eq(last["quarter"])
        tail = block.loc[unfinished].copy()
        complete = block.loc[~unfinished]
        if not complete.empty:
            yield complete
        del block, chunk, complete
    if not tail.empty:
        yield tail


def build_embedding_kpis(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Reduce ordered turn chunks to coherence and pooled call vectors before issuer comparisons."""
    prepared_section, qa_section = CALL_SECTIONS
    sections = ((qa_section, "ec_qa_qq_distance"), (prepared_section, "ec_prep_qq_distance"), (qa_section, "ec_questions_only_qq_distance"))
    metrics: list[pd.DataFrame] = []
    pooled_parts: dict[str, list[pd.DataFrame]] = {name: [] for _, name in sections}

    # Reduce complete calls immediately; retain only scalar KPIs and pooled vectors.
    for embeddings in _complete_embedding_calls(chunks):
        keys = embeddings[["ticker", "quarter"]].drop_duplicates()
        comparable = _comparable_embedding_calls(embeddings)
        metrics.append(keys.merge(_qa_coherence(comparable), on=["ticker", "quarter"], how="left"))
        for section, name in sections:
            turns = comparable[comparable["tag"].eq("question")] if name == "ec_questions_only_qq_distance" else comparable
            pooled = _pooled_section_vectors(turns, section)
            if not pooled.empty:
                pooled_parts[name].append(pooled)
            del turns, pooled
        del embeddings, comparable
    if not metrics:
        return pd.DataFrame(columns=["ticker", "quarter"])
    kpi = pd.concat(metrics, ignore_index=True)

    # Compare pooled issuer histories after all supported aliases have been read.
    for _, name in sections:
        parts = pooled_parts.pop(name)
        pooled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        kpi = kpi.merge(_qq_distance(pooled, name), on=["ticker", "quarter"], how="left")
    kpi = kpi.drop(columns=["as_of"], errors="ignore")
    return kpi
