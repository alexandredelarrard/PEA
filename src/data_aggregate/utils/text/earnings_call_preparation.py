"""Earnings-call preparation metrics used by aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.incremental import PartWindow
from src.utils.text.earnings_call_split import CALL_SECTIONS
from src.utils.text.text_metrics import word_tokens

LM_UNCERTAINTY: frozenset[str] = frozenset(
    {
        # weak modal / hedging
        "may",
        "might",
        "could",
        "maybe",
        "perhaps",
        "possible",
        "possibly",
        "probable",
        "probably",
        "seems",
        "appears",
        "appear",
        "suggests",
        "suggest",
        "tend",
        "tends",
        # uncertainty proper
        "approximate",
        "approximately",
        "assume",
        "assumed",
        "assumes",
        "assuming",
        "assumption",
        "assumptions",
        "believe",
        "believed",
        "believes",
        "cautious",
        "cautiously",
        "clarification",
        "conditional",
        "confusion",
        "contingency",
        "contingencies",
        "contingent",
        "depend",
        "depended",
        "dependence",
        "dependent",
        "depending",
        "depends",
        "deviation",
        "deviations",
        "doubt",
        "doubtful",
        "doubts",
        "exposure",
        "exposures",
        "fluctuate",
        "fluctuated",
        "fluctuates",
        "fluctuating",
        "fluctuation",
        "fluctuations",
        "imprecise",
        "imprecision",
        "indefinite",
        "indefinitely",
        "indeterminate",
        "instabilities",
        "instability",
        "likelihood",
        "nonassessable",
        "occasionally",
        "pending",
        "precaution",
        "precautionary",
        "precautions",
        "predict",
        "predictability",
        "predicted",
        "predicting",
        "prediction",
        "predictions",
        "predictive",
        "predictor",
        "predictors",
        "predicts",
        "preliminarily",
        "preliminary",
        "presumably",
        "presume",
        "presumed",
        "reassess",
        "reassessed",
        "reassessing",
        "reassessment",
        "reconsider",
        "reconsidered",
        "reconsidering",
        "recalculate",
        "recalculated",
        "revise",
        "revised",
        "revises",
        "revising",
        "risk",
        "risked",
        "riskier",
        "riskiest",
        "riskiness",
        "risks",
        "risky",
        "roughly",
        "sometime",
        "sometimes",
        "somewhat",
        "speculate",
        "speculated",
        "speculates",
        "speculating",
        "speculation",
        "speculations",
        "speculative",
        "sudden",
        "suddenly",
        "susceptibility",
        "tentative",
        "tentatively",
        "turbulence",
        "uncertain",
        "uncertainly",
        "uncertainties",
        "uncertainty",
        "unclear",
        "unconfirmed",
        "undecided",
        "undefined",
        "undetermined",
        "unexpected",
        "unexpectedly",
        "unforeseeable",
        "unforeseen",
        "unknown",
        "unknowns",
        "unlikely",
        "unpredictability",
        "unpredictable",
        "unpredictably",
        "unproven",
        "unquantifiable",
        "unsettled",
        "unspecific",
        "unspecified",
        "untested",
        "unusual",
        "unusually",
        "vagaries",
        "vague",
        "vaguely",
        "vagueness",
        "variability",
        "variable",
        "variables",
        "variation",
        "variations",
        "varied",
        "varies",
        "vary",
        "varying",
        "volatile",
        "volatility",
    }
)


def uncertainty_ratio(text: str) -> float:
    """Share of tokens that are LM uncertainty / weak-modal words. 0.0 for empty text."""
    toks = word_tokens(text)
    if not toks:
        return 0.0
    hits = sum(1 for w in toks if w in LM_UNCERTAINTY)
    return hits / len(toks)


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
    text = str(value)
    if len(text) == 6 and text[4] == "Q" and text[:4].isdigit() and text[5] in "1234":
        return int(text[:4]) * 4 + int(text[5]) - 1
    return np.nan


def sentiment_rows(call: pd.Series, existing: pd.DataFrame, probabilities: list[dict[str, float] | None], cache_model: str) -> pd.DataFrame:
    """Assemble one completed call from new scores, reusable scores and genuine absence."""
    tags = CALL_SECTIONS
    previous = {str(row.tag): row for row in existing.itertuples(index=False)}
    missing = [tag for tag in tags if call[tag] and tag not in previous]
    if len(probabilities) != len(missing):
        raise ValueError(f"Sentiment returned {len(probabilities)} scores for {len(missing)} nonempty sections")
    scored = dict(zip(missing, probabilities, strict=True))
    rows = []
    for tag in tags:
        text = str(call[tag])
        probability = scored.get(tag)
        if text and tag in previous:
            cached = previous[tag]
            probability = {"pos": cached.sent_pos, "neg": cached.sent_neg, "neu": cached.sent_neu}
        if text:
            values = np.asarray([probability[key] for key in ("pos", "neg", "neu")], dtype="float64") if probability is not None else np.asarray([])
            if (
                values.size != 3
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
                or not np.isclose(values.sum(), 1.0, atol=2e-6, rtol=0)
            ):
                raise ValueError("Sentiment output must contain finite pos/neg/neu probabilities summing to one")
            pos, neg, neu = np.round(values, 6)
        else:
            pos = neg = neu = np.nan
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
                "uncertainty_ratio": round(float(call[f"{tag}_uncertainty"]), 6) if text else np.nan,
                "model": cache_model,
            }
        )
    return pd.DataFrame(rows)
