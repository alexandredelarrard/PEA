"""Earnings-call preparation metrics used by aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.incremental import PartWindow

SENTIMENT_TAGS = ("prepared_remarks", "questions", "answers")


@dataclass
class TextPlan:
    """Source/cache work sets and the independent feature rewrite window."""

    window: PartWindow
    source: pd.DataFrame
    sentiment: pd.DataFrame
    embedding_keys: pd.DataFrame
    sentiment_work: set[tuple[str, str]]
    embedding_work: set[tuple[str, str]]
    eligible: set[tuple[str, str]]


def quarter_number(value: object) -> float:
    """Fiscal YYYYQn ordinal, or NaN for an invalid label."""
    # Accept only the canonical fiscal-quarter label.
    text = str(value)
    if len(text) == 6 and text[4] == "Q" and text[:4].isdigit() and text[5] in "1234":
        return int(text[:4]) * 4 + int(text[5]) - 1
    return np.nan


def sentiment_rows(call: pd.Series, existing: pd.DataFrame, probabilities: list[dict[str, float] | None], cache_model: str) -> pd.DataFrame:
    """Assemble one completed call from new scores, reusable scores and genuine absence."""
    # Match new probabilities only to nonempty roles without a valid saved partner.
    previous = {str(row.tag): row for row in existing.itertuples(index=False)}
    missing = [tag for tag in SENTIMENT_TAGS if call[tag] and tag not in previous]
    if len(probabilities) != len(missing):
        raise ValueError(f"Sentiment returned {len(probabilities)} scores for {len(missing)} nonempty sections")
    # Validate new external distributions before combining them with valid cache rows.
    scored = {}
    for tag, probability in zip(missing, probabilities, strict=True):
        values = np.asarray([probability[key] for key in ("pos", "neg", "neu")], dtype="float64") if probability is not None else np.asarray([])
        if (
            values.size != 3
            or not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
            or not np.isclose(values.sum(), 1.0, atol=2e-6, rtol=0)
        ):
            raise ValueError("Sentiment output must contain finite pos/neg/neu probabilities summing to one")
        scored[tag] = np.round(values, 6)

    # Save three canonical roles, including explicit absence for empty sections.
    rows = []
    for tag in SENTIMENT_TAGS:
        if not call[tag]:
            pos = neg = neu = np.nan
        elif tag in previous:
            cached = previous[tag]
            pos, neg, neu = cached.sent_pos, cached.sent_neg, cached.sent_neu
        else:
            pos, neg, neu = scored[tag]
        rows.append(
            {
                "ticker": call.ticker,
                "quarter": call.quarter,
                "tag": tag,
                "as_of": call.as_of,
                "sent_pos": pos,
                "sent_neg": neg,
                "sent_neu": neu,
                "n_words": int(call[f"{tag}_words"]),
                "model": cache_model,
            }
        )
    return pd.DataFrame(rows)
