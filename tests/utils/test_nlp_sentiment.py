"""Model-free checks of sentiment windowing, probability aggregation and CPU/CUDA routing."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.data_aggregate.utils.text import nlp_sentiment
from src.data_aggregate.utils.text.nlp_sentiment import _length_weighted_average, _window_ids


def test_window_ids_splits_long_docs() -> None:
    assert _window_ids([]) == []
    assert _window_ids([1, 2, 3]) == [[1, 2, 3]]
    tokens = list(range(1021))
    assert _window_ids(tokens) == [tokens[:510], tokens[510:1020], tokens[1020:]]
    print("SANITY: real FinBERT's 510-token windows preserve document order and the final remainder.")


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
            nlp_sentiment.SentimentEngine("fixture/tone")
        mirror.assert_not_called()
    else:
        engine = nlp_sentiment.SentimentEngine("fixture/tone")
        mirror.assert_called_once_with("fixture/tone")
        assert engine.device == expected
        model.to.assert_called_once_with(expected)
        assert engine._score_windows(["Revenue increased."]) == [{"pos": 0.7, "neg": 0.2, "neu": 0.1}]
        engine._tok.return_value.to.assert_called_once_with(expected)
        assert f"device={expected}" in caplog.text
        assert "CUDA build=13.0" in caplog.text
    print(f"SANITY: requested={requested!r}, CUDA available={available} -> {expected or 'failure before model download'}; model and inputs agree.")


@pytest.mark.parametrize("requested", ["cuda", "cpu", ""])
@pytest.mark.parametrize("stack_available", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_factory_propagates_required_cuda_failures(
    monkeypatch: pytest.MonkeyPatch, requested: str, stack_available: bool, error_type: type[Exception]
) -> None:
    monkeypatch.setenv("FINBERT_DEVICE", requested)
    monkeypatch.setattr(nlp_sentiment, "ml_stack_available", lambda: stack_available)
    constructor = MagicMock(side_effect=error_type("GPU/model load failed"))
    monkeypatch.setattr(nlp_sentiment, "SentimentEngine", constructor)
    for _ in range(2):
        if requested == "cuda":
            with pytest.raises(error_type if stack_available else RuntimeError):
                nlp_sentiment.get_sentiment_engine("fixture/tone")
        else:
            assert nlp_sentiment.get_sentiment_engine("fixture/tone") is None
    if stack_available:
        constructor.assert_called_with("fixture/tone")
        assert constructor.call_count == 2
    else:
        constructor.assert_not_called()
    print(f"SANITY: device={requested or 'auto'}, stack={stack_available}; CUDA failures propagate on every call; optional CPU loads can skip.")


def test_factory_loads_requested_model_without_global_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINBERT_DEVICE", "cpu")
    monkeypatch.setattr(nlp_sentiment, "ml_stack_available", lambda: True)
    constructor = MagicMock()
    monkeypatch.setattr(nlp_sentiment, "SentimentEngine", constructor)

    first = nlp_sentiment.get_sentiment_engine("fixture/tone")

    assert first is constructor.return_value
    assert nlp_sentiment.get_sentiment_engine("fixture/tone") is first
    assert constructor.call_count == 2
    print("SANITY: factories construct their requested model; production reuses one local engine inside the stage.")


def test_score_texts_preserves_document_weights_and_shared_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = MagicMock(__version__="test", version=SimpleNamespace(cuda=None))
    torch.cuda.is_available.return_value = False
    transformers = MagicMock()
    model = transformers.AutoModelForSequenceClassification.from_pretrained.return_value
    model.config.id2label = {0: "neutral", 1: "positive", 2: "negative"}
    tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
    encoded = MagicMock()
    encoded.to.return_value = {"input_ids": object()}

    def tokenize(text: str | list[str], **_kwargs: object) -> dict[str, list[int]] | MagicMock:
        if isinstance(text, str):
            return {"input_ids": [1] * (1021 if text == "long" else 2)}
        return encoded

    tokenizer.side_effect = tokenize
    tokenizer.decode.side_effect = lambda ids, **_kwargs: f"{len(ids)} tokens"
    probabilities = [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.7, 0.1, 0.2]] + [[0.3, 0.2, 0.5]] * 14
    batches = [MagicMock(), MagicMock()]
    batches[0].cpu.return_value.tolist.return_value = probabilities[:16]
    batches[1].cpu.return_value.tolist.return_value = probabilities[16:]
    torch.softmax.side_effect = batches
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(nlp_sentiment, "ensure_local_model", lambda _model: "offline-model")
    monkeypatch.setenv("FINBERT_DEVICE", "cpu")
    engine = nlp_sentiment.SentimentEngine("fixture/tone")

    result = engine.score_texts(["", None, "long", *[f"short {i}" for i in range(14)], "   "])

    assert len(result) == 18
    assert result[:2] == [None, None]
    assert result[-1] is None
    assert result[2] == pytest.approx(
        {"pos": (510 * 0.8 + 510 * 0.6 + 0.1) / 1021, "neg": (510 * 0.1 + 510 * 0.2 + 0.2) / 1021, "neu": (510 * 0.1 + 510 * 0.2 + 0.7) / 1021}
    )
    assert result[3:-1] == [{"pos": 0.2, "neg": 0.5, "neu": 0.3}] * 14
    batch_calls = [call for call in tokenizer.call_args_list if isinstance(call.args[0], list)]
    assert [len(call.args[0]) for call in batch_calls] == [16, 1]
    assert all(call.kwargs["max_length"] == 512 for call in batch_calls)
    assert [call.args[0] for call in tokenizer.decode.call_args_list][:3] == [[1] * 510, [1] * 510, [1]]
    print("SANITY: 17 windows share 16+1 batches; label order, token weights, blanks and document order remain correct.")


def test_local_model_cache_avoids_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PEA_MODEL_DIR", str(tmp_path))
    model_dir = tmp_path / "fixture/tone".replace("/", "__")
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"cached fixture weights")
    (model_dir / "vocab.txt").write_text("fixture", encoding="utf-8")
    mirror_client = MagicMock()
    monkeypatch.setitem(sys.modules, "curl_cffi", mirror_client)

    assert nlp_sentiment.ensure_local_model("fixture/tone") == str(model_dir)

    mirror_client.requests.Session.assert_not_called()
    print("SANITY: existing local model config and weights are reused without a network session.")


def test_model_mirror_preserves_proxy_and_weight_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("PEA_MODEL_DIR", str(tmp_path))
    for variable in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(variable, raising=False)
    client = MagicMock()
    config_attempts = 0

    def fetch(url: str) -> SimpleNamespace:
        nonlocal config_attempts
        if url.endswith("config.json") and not url.endswith("tokenizer_config.json"):
            config_attempts += 1
            if config_attempts == 1:
                raise OSError("proxy CA failure")
            return SimpleNamespace(status_code=200, content=b"{}")
        if url.endswith("pytorch_model.bin"):
            return SimpleNamespace(status_code=200, content=b"fixture weights")
        return SimpleNamespace(status_code=404, content=b"")

    client.Session.return_value.get.side_effect = fetch
    monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(requests=client))
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", client)
    caplog.set_level(logging.WARNING, logger=nlp_sentiment.__name__)

    result = Path(nlp_sentiment.ensure_local_model("fixture/tone"))

    assert (result / "config.json").read_bytes() == b"{}"
    assert (result / "pytorch_model.bin").read_bytes() == b"fixture weights"
    assert not (result / "model.safetensors").exists()
    assert [call.kwargs["verify"] for call in client.Session.call_args_list][:2] == [True, False]
    assert "UNVERIFIED" in caplog.text
    print("SANITY: public model mirror retries proxy CA failure, logs unverified fetch and falls back from safetensors to bin.")


@pytest.mark.parametrize("error_type", [NameError, AttributeError, TypeError, KeyError, ImportError])
def test_factory_propagates_programming_errors(monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]) -> None:
    monkeypatch.setenv("FINBERT_DEVICE", "cpu")
    monkeypatch.setattr(nlp_sentiment, "ml_stack_available", lambda: True)
    constructor = MagicMock(side_effect=error_type("programming failure"))
    monkeypatch.setattr(nlp_sentiment, "SentimentEngine", constructor)

    with pytest.raises(error_type, match="programming failure"):
        nlp_sentiment.get_sentiment_engine("fixture/tone")

    print(f"SANITY: {error_type.__name__} escapes model initialization rather than silently skipping sentiment.")


@pytest.mark.parametrize("missing_name", ["curl_cffi", "broken_dependency"])
def test_model_mirror_only_skips_absent_optional_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_name: str) -> None:
    monkeypatch.setenv("PEA_MODEL_DIR", str(tmp_path))
    importer = MagicMock(side_effect=ModuleNotFoundError("missing fixture dependency", name=missing_name))
    monkeypatch.setattr(nlp_sentiment, "import_module", importer)

    if missing_name == "curl_cffi":
        assert nlp_sentiment.ensure_local_model("fixture/tone") == "fixture/tone"
    else:
        with pytest.raises(ModuleNotFoundError, match="missing fixture dependency"):
            nlp_sentiment.ensure_local_model("fixture/tone")

    print(f"SANITY: missing dependency {missing_name} {'falls back to HF' if missing_name == 'curl_cffi' else 'fails visibly'}.")


@pytest.mark.parametrize("error_type", [OSError, NameError])
def test_model_mirror_only_absorbs_network_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]) -> None:
    monkeypatch.setenv("PEA_MODEL_DIR", str(tmp_path))
    client = MagicMock()
    client.Session.return_value.get.side_effect = error_type("mirror failure")
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", client)

    if error_type is OSError:
        assert nlp_sentiment.ensure_local_model("fixture/tone") == "fixture/tone"
    else:
        with pytest.raises(NameError, match="mirror failure"):
            nlp_sentiment.ensure_local_model("fixture/tone")

    print(f"SANITY: mirror {error_type.__name__} {'uses the HF fallback' if error_type is OSError else 'fails visibly'}.")
