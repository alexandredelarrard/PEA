"""Model-free checks of sentiment windowing, probability aggregation and CPU/CUDA routing."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.utils import nlp_sentiment
from src.utils.nlp_sentiment import _length_weighted_average, _window_ids


def test_window_ids_splits_long_docs() -> None:
    assert _window_ids([], 3) == []  # empty -> no windows
    assert _window_ids([1, 2, 3], 5) == [[1, 2, 3]]  # shorter than window -> one
    assert _window_ids([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]  # exact chunking + remainder
    print("SANITY: long documents split into bounded token windows, including the remainder.")


def test_length_weighted_average_weights_by_tokens() -> None:
    # two windows: a very positive short one + a neutral long one -> weighted toward long
    rows = [
        [0.9, 0.05, 0.05],  # pos, neg, neu  (short window)
        [0.1, 0.10, 0.80],
    ]  # long window
    out = _length_weighted_average(rows, [10, 90])
    # expected = 0.1*[.9,.05,.05] + 0.9*[.1,.10,.80]
    assert abs(out[0] - (0.1 * 0.9 + 0.9 * 0.1)) < 1e-9
    assert abs(out[2] - (0.1 * 0.05 + 0.9 * 0.80)) < 1e-9
    assert abs(sum(out) - 1.0) < 1e-9  # still a distribution

    # zero/negative weights -> falls back to a plain mean (never divides by zero)
    eq = _length_weighted_average([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0, 0])
    assert eq == [0.5, 0.5, 0.0]
    assert _length_weighted_average([], []) == []

    print("\n=== SANITY CHECK: nlp_sentiment pure helpers ===")
    print(
        "  long docs split into <=stride token windows (with remainder); per-window "
        "probs aggregate length-weighted to one distribution (sums to 1), with a safe "
        "plain-mean fallback when weights are non-positive. Validated."
    )


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [(None, False, "cpu"), (None, True, "cuda"), ("cpu", True, "cpu"), ("cuda", True, "cuda"), (" CUDA ", False, None)],
)
def test_device_selection_and_tensor_placement(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, requested: str | None, available: bool, expected: str | None
) -> None:
    torch = MagicMock(__version__="test+cu130", version=SimpleNamespace(cuda="13.0"))
    torch.cuda.is_available.return_value = available
    torch.cuda.get_device_name.return_value = "test GPU"
    transformers = MagicMock()
    model = transformers.AutoModelForSequenceClassification.from_pretrained.return_value
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    torch.softmax.return_value.cpu.return_value.tolist.return_value = [[0.7, 0.2, 0.1]]
    mirror = MagicMock(return_value="offline-model")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(nlp_sentiment, "ensure_local_model", mirror)
    if requested is None:
        monkeypatch.delenv("FINBERT_DEVICE", raising=False)
    else:
        monkeypatch.setenv("FINBERT_DEVICE", requested)
    caplog.set_level(logging.INFO, logger=nlp_sentiment.__name__)

    if expected is None:
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            nlp_sentiment.SentimentEngine()
        mirror.assert_not_called()
    else:
        engine = nlp_sentiment.SentimentEngine()
        assert engine.device == expected
        model.to.assert_called_once_with(expected)
        assert engine._score_windows(["Revenue increased."]) == [{"pos": 0.7, "neg": 0.2, "neu": 0.1}]
        engine._tok.return_value.to.assert_called_once_with(expected)
        assert f"device={expected}" in caplog.text
        assert "CUDA build=13.0" in caplog.text
    print(f"SANITY: requested={requested!r}, CUDA available={available} -> {expected or 'failure before model download'}; model and inputs agree.")


@pytest.mark.parametrize("requested", ["cuda", "cpu", ""])
@pytest.mark.parametrize("stack_available", [False, True])
def test_factory_propagates_required_cuda_failures(monkeypatch: pytest.MonkeyPatch, requested: str, stack_available: bool) -> None:
    monkeypatch.setenv("FINBERT_DEVICE", requested)
    monkeypatch.setattr(nlp_sentiment, "_ENGINE", None)
    monkeypatch.setattr(nlp_sentiment, "_ENGINE_TRIED", False)
    monkeypatch.setattr(nlp_sentiment, "ml_stack_available", lambda: stack_available)
    monkeypatch.setattr(nlp_sentiment, "SentimentEngine", MagicMock(side_effect=RuntimeError("GPU/model load failed")))
    for _ in range(2):
        if requested == "cuda":
            with pytest.raises(RuntimeError):
                nlp_sentiment.get_sentiment_engine()
        else:
            assert nlp_sentiment.get_sentiment_engine() is None
    print(f"SANITY: device={requested or 'auto'}, stack={stack_available}; CUDA failures propagate on every call; optional CPU loads can skip.")


if __name__ == "__main__":
    test_window_ids_splits_long_docs()
    test_length_weighted_average_weights_by_tokens()
    print("\n=== SANITY CHECK: nlp_sentiment pure helpers ===")
    print(
        "  long docs split into ≤stride token windows (with remainder); per-window "
        "probs aggregate length-weighted to one distribution (sums to 1), with a safe "
        "plain-mean fallback when weights are non-positive. Validated."
    )
