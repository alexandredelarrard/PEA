"""
text_metrics.py  (src/utils/text_metrics.py)
--------------------------------------------
Cheap, pure (no-GPU) text characterizations for earnings-call sections, complementing
the FinBERT tone score:

  * word_count            length of the section (words)
  * uncertainty_ratio     share of Loughran-McDonald UNCERTAINTY / weak-modal words
                          (hedging language — managers hedge more when the outlook is
                          weak; a classic disclosure-tone signal)
  * content_frequency     non-stopword token frequencies (bag of words)
  * cosine_similarity     cosine between two bag-of-words vectors -> feeds the
                          VOCABULARY-NOVELTY KPI (1 - similarity vs the prior call:
                          high = a new narrative / strategy shift)

The uncertainty list is a curated subset of the Loughran-McDonald financial
sentiment dictionary's Uncertainty + Weak-Modal entries (the high-frequency mass);
everything here is deterministic and unit-tested.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Curated subset of Loughran-McDonald "Uncertainty" + "Weak Modal" words (lowercased).
LM_UNCERTAINTY: frozenset[str] = frozenset({
    # weak modal / hedging
    "may", "might", "could", "maybe", "perhaps", "possible", "possibly", "probable",
    "probably", "seems", "appears", "appear", "suggests", "suggest", "tend", "tends",
    # uncertainty proper
    "approximate", "approximately", "assume", "assumed", "assumes", "assuming",
    "assumption", "assumptions", "believe", "believed", "believes", "cautious",
    "cautiously", "clarification", "conditional", "confusion", "contingency",
    "contingencies", "contingent", "depend", "depended", "dependence", "dependent",
    "depending", "depends", "deviation", "deviations", "doubt", "doubtful", "doubts",
    "exposure", "exposures", "fluctuate", "fluctuated", "fluctuates", "fluctuating",
    "fluctuation", "fluctuations", "imprecise", "imprecision", "indefinite",
    "indefinitely", "indeterminate", "instabilities", "instability", "likelihood",
    "nonassessable", "occasionally", "pending", "precaution", "precautionary",
    "precautions", "predict", "predictability", "predicted", "predicting",
    "prediction", "predictions", "predictive", "predictor", "predictors", "predicts",
    "preliminarily", "preliminary", "presumably", "presume", "presumed", "reassess",
    "reassessed", "reassessing", "reassessment", "reconsider", "reconsidered",
    "reconsidering", "recalculate", "recalculated", "revise", "revised", "revises",
    "revising", "risk", "risked", "riskier", "riskiest", "riskiness", "risks", "risky",
    "roughly", "sometime", "sometimes", "somewhat", "speculate", "speculated",
    "speculates", "speculating", "speculation", "speculations", "speculative",
    "sudden", "suddenly", "susceptibility", "tentative", "tentatively", "turbulence",
    "uncertain", "uncertainly", "uncertainties", "uncertainty", "unclear",
    "unconfirmed", "undecided", "undefined", "undetermined", "unexpected",
    "unexpectedly", "unforeseeable", "unforeseen", "unknown", "unknowns", "unlikely",
    "unpredictability", "unpredictable", "unpredictably", "unproven", "unquantifiable",
    "unsettled", "unspecific", "unspecified", "untested", "unusual", "unusually",
    "vagaries", "vague", "vaguely", "vagueness", "variability", "variable", "variables",
    "variation", "variations", "varied", "varies", "vary", "varying", "volatile",
    "volatility",
})

# Compact English stopword set for the vocabulary-novelty bag of words (drop the
# high-frequency function words that would dominate a cosine and swamp real topic shifts).
_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "out", "over", "own", "quarter", "s",
    "same", "she", "so", "some", "such", "t", "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "year", "you",
    "your", "yours", "thank", "thanks", "yeah", "okay", "ok", "going", "think", "know",
    "well", "right", "sure", "look", "kind", "lot", "really", "actually", "question",
})

_WORD_RE = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")


def word_tokens(text: str) -> list[str]:
    """Lowercase alphabetic word tokens (apostrophes/hyphens kept internal). Pure."""
    if not text:
        return []
    return _WORD_RE.findall(str(text).lower())


def word_count(text: str) -> int:
    return len(word_tokens(text))


def uncertainty_ratio(text: str) -> float:
    """Share of tokens that are LM uncertainty / weak-modal words. 0.0 for empty text."""
    toks = word_tokens(text)
    if not toks:
        return 0.0
    hits = sum(1 for w in toks if w in LM_UNCERTAINTY)
    return hits / len(toks)


def content_frequency(text: str) -> Counter:
    """Bag-of-words Counter over CONTENT tokens (stopwords + 1-char tokens dropped).
    Feeds the vocabulary-novelty cosine. Pure."""
    return Counter(w for w in word_tokens(text)
                   if len(w) > 1 and w not in _STOPWORDS)


def cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity of two bag-of-words Counters in [0, 1]. 0.0 if either is empty
    (an empty side => maximal novelty downstream)."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
