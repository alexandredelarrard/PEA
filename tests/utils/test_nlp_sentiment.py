"""
Pure, model-free helpers of the sentiment engine (src/utils/nlp_sentiment.py):
the long-document token windowing and the length-weighted probability aggregation.
No torch / transformers / network needed (the GPU path is exercised by the
end-to-end 10-ticker validation).
"""
from __future__ import annotations

from src.utils.nlp_sentiment import _length_weighted_average, _window_ids


def test_window_ids_splits_long_docs():
    assert _window_ids([], 3) == []                          # empty -> no windows
    assert _window_ids([1, 2, 3], 5) == [[1, 2, 3]]          # shorter than window -> one
    assert _window_ids([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]  # exact chunking + remainder


def test_length_weighted_average_weights_by_tokens():
    # two windows: a very positive short one + a neutral long one -> weighted toward long
    rows = [[0.9, 0.05, 0.05],    # pos, neg, neu  (short window)
            [0.1, 0.10, 0.80]]    # long window
    out = _length_weighted_average(rows, [10, 90])
    # expected = 0.1*[.9,.05,.05] + 0.9*[.1,.10,.80]
    assert abs(out[0] - (0.1 * 0.9 + 0.9 * 0.1)) < 1e-9
    assert abs(out[2] - (0.1 * 0.05 + 0.9 * 0.80)) < 1e-9
    assert abs(sum(out) - 1.0) < 1e-9                        # still a distribution

    # zero/negative weights -> falls back to a plain mean (never divides by zero)
    eq = _length_weighted_average([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0, 0])
    assert eq == [0.5, 0.5, 0.0]
    assert _length_weighted_average([], []) == []

    print("\n=== SANITY CHECK: nlp_sentiment pure helpers ===")
    print("  long docs split into <=stride token windows (with remainder); per-window "
          "probs aggregate length-weighted to one distribution (sums to 1), with a safe "
          "plain-mean fallback when weights are non-positive. Validated.")


if __name__ == "__main__":
    test_window_ids_splits_long_docs()
    test_length_weighted_average_weights_by_tokens()
    print("\n=== SANITY CHECK: nlp_sentiment pure helpers ===")
    print("  long docs split into ≤stride token windows (with remainder); per-window "
          "probs aggregate length-weighted to one distribution (sums to 1), with a safe "
          "plain-mean fallback when weights are non-positive. Validated.")
