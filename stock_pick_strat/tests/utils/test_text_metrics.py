"""
Pure text-metric helpers for earnings-call features (src/utils/text_metrics.py):
word count, Loughran-McDonald uncertainty ratio, and the bag-of-words cosine that
feeds the vocabulary-novelty KPI. No model / network.
"""
from __future__ import annotations

from src.utils.text_metrics import (
    content_frequency,
    cosine_similarity,
    uncertainty_ratio,
    word_count,
    word_tokens,
)


def test_word_count_and_tokens():
    assert word_count("") == 0
    assert word_count("The company grew revenue 10%.") == 4   # 10 is dropped (non-alpha)
    assert word_tokens("Don't stop") == ["don't", "stop"]


def test_uncertainty_ratio_counts_lm_words():
    # 2 uncertainty words ("may", "uncertain") out of 10 tokens -> 0.2
    txt = "Results may be uncertain but the team executed the plan"
    assert word_count(txt) == 10
    assert abs(uncertainty_ratio(txt) - 0.2) < 1e-9
    assert uncertainty_ratio("") == 0.0
    assert uncertainty_ratio("strong solid confident growth") == 0.0   # no hedging words


def test_cosine_similarity_and_novelty_direction():
    a = content_frequency("We are launching a new cloud platform for enterprise customers")
    a2 = content_frequency("We are launching a new cloud platform for enterprise customers")
    b = content_frequency("Litigation and restructuring charges weighed on margins this period")
    assert abs(cosine_similarity(a, a2) - 1.0) < 1e-9      # identical -> cosine 1 (novelty 0)
    assert cosine_similarity(a, b) < 0.2                    # different topic -> low cosine
    assert cosine_similarity(a, content_frequency("")) == 0.0   # empty -> 0 (max novelty)


def test_content_frequency_drops_stopwords():
    cf = content_frequency("the the and of cloud cloud revenue")
    assert "the" not in cf and "and" not in cf and "of" not in cf   # stopwords dropped
    assert cf["cloud"] == 2 and cf["revenue"] == 1

    print("\n=== SANITY CHECK: text metrics ===")
    print("  word count ignores non-alpha; LM uncertainty ratio counts hedging words "
          "(2/10=0.2); bag-of-words cosine = 1 for identical text, ~0 for different topics "
          "/ empty; stopwords dropped from the vocab vector. Validated.")


if __name__ == "__main__":
    test_word_count_and_tokens()
    test_uncertainty_ratio_counts_lm_words()
    test_cosine_similarity_and_novelty_direction()
    test_content_frequency_drops_stopwords()
    print("\n=== SANITY CHECK: text metrics ===")
    print("  word count ignores non-alpha; LM uncertainty ratio counts hedging words; "
          "bag-of-words cosine = 1 for identical text, ~0 for different topics / empty. Validated.")
