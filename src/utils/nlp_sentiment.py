"""Local finance-tone scoring of long documents (earnings-call sections) with a HuggingFace classifier (FinBERT-tone).

    engine = get_sentiment_engine(context.log)      # None if torch/transformers absent
    probs  = engine.score_texts([doc1, doc2, ...])   # -> [{'pos','neg','neu'}|None, ...]

torch/transformers are optional: without them the builder returns None and callers skip. Runs on CUDA when
available. Documents are split into <=510-token windows whose class probabilities are length-weighted back to
one distribution; columns are mapped from the model's own `config.id2label`, never hardcoded indices.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from src.constants.constants import FINBERT_TONE_MODEL

# BERT's token window; longer sections are chunked and length-weighted.
FINBERT_MAX_TOKENS = 512


def ml_stack_available() -> bool:
    """True if both torch and transformers can be imported (does NOT load a model)."""
    import importlib.util as u

    return bool(u.find_spec("torch")) and bool(u.find_spec("transformers"))


# --- Corporate-proxy model mirror (curl_cffi) ---
# OpenSSL 3.x rejects some corporate MITM-proxy CAs, so model files are mirrored via curl_cffi and loaded offline.
_MODEL_META_FILES = ("config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "merges.txt", "vocab.json")
_MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _local_model_dir(model_name: str) -> Path:
    base = os.getenv("PEA_MODEL_DIR") or str(Path.home() / ".cache" / "pea_models")
    return Path(base) / model_name.replace("/", "__")


def ensure_local_model(model_name: str, logger: logging.Logger) -> str:
    """Mirror a HuggingFace model locally via curl_cffi and return the local dir for an offline `from_pretrained`.

    Tries the CA bundle first, then an unverified fetch (logged) as a last resort for this public, read-only artifact.
    Returns `model_name` unchanged when curl_cffi is unavailable or the download fails, so the HF client still runs."""
    dest = _local_model_dir(model_name)
    if (dest / "config.json").exists() and any((dest / w).exists() for w in _MODEL_WEIGHT_FILES):
        return str(dest)  # already mirrored
    try:
        from curl_cffi import requests as cffi
    except Exception:  # curl_cffi absent -> let HF try
        return model_name

    ca = next((os.environ[v] for v in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE") if os.environ.get(v)), None)
    base_url = f"https://huggingface.co/{model_name}/resolve/main/"

    def _fetch(fname: str, required: bool) -> bool:
        out = dest / fname
        if out.exists():
            return True
        for verify in [ca, False] if ca else [True, False]:
            try:
                r = cffi.Session(impersonate="chrome124", verify=verify, timeout=180).get(base_url + fname)
            except Exception:
                continue
            if r.status_code == 404:
                return False  # legitimately absent (e.g. no safetensors)
            if r.status_code == 200 and r.content:
                if verify is False:
                    logger.warning("Fetched %s UNVERIFIED via curl_cffi (public model behind corporate proxy).", fname)
                dest.mkdir(parents=True, exist_ok=True)
                out.write_bytes(r.content)
                return True
        if required:
            raise RuntimeError(f"could not download {fname}")
        return False

    try:
        _fetch("config.json", required=True)
        if not any(_fetch(w, required=False) for w in _MODEL_WEIGHT_FILES):
            raise RuntimeError("no weight file (safetensors / pytorch_model.bin) found")
        for f in _MODEL_META_FILES:
            if f != "config.json":
                _fetch(f, required=False)  # tokenizer files, best-effort
    except Exception as e:  # noqa: BLE001
        logger.warning("curl_cffi model mirror failed for %s (%s) -> trying HF client.", model_name, e)
        return model_name
    logger.info("Mirrored %s locally to %s (curl_cffi).", model_name, dest)
    return str(dest)


# --- Pure helpers (no torch) ---
def _window_ids(ids: list[int], stride: int) -> list[list[int]]:
    """Split a token-id list into consecutive non-overlapping windows of <=`stride` tokens; [] for empty input."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    return [ids[i : i + stride] for i in range(0, len(ids), stride)] if ids else []


def _length_weighted_average(prob_rows: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """Length-weighted mean of per-window probability vectors; plain mean if all weights are non-positive, [] if empty."""
    rows = [list(map(float, r)) for r in prob_rows]
    if not rows:
        return []
    n = len(rows[0])
    w = [max(float(x), 0.0) for x in weights]
    tot = sum(w)
    if tot <= 0:  # degenerate -> uniform mean
        w = [1.0] * len(rows)
        tot = float(len(rows))
    out = [0.0] * n
    for row, wi in zip(rows, w, strict=False):
        for j in range(n):
            out[j] += wi * row[j]
    return [v / tot for v in out]


# --- Engine ---
class SentimentEngine:
    """Thin wrapper over a HuggingFace tone classifier; build it via `get_sentiment_engine`."""

    def __init__(
        self, model_name: str = FINBERT_TONE_MODEL, max_tokens: int = FINBERT_MAX_TOKENS, batch_size: int = 16, logger: logging.Logger | None = None
    ) -> None:
        import torch  # local import (heavy, optional)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._log = logger or logging.getLogger(__name__)
        self.max_tokens = int(max_tokens)
        self.batch_size = int(batch_size)

        device_env = os.getenv("FINBERT_DEVICE", "").strip().lower()
        if device_env in ("cpu", "cuda"):
            self.device = device_env if (device_env == "cpu" or torch.cuda.is_available()) else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # mirror locally via curl_cffi first (corporate-proxy workaround), else HF client
        resolved = ensure_local_model(model_name, self._log)
        self._tok = AutoTokenizer.from_pretrained(resolved)
        self._model = AutoModelForSequenceClassification.from_pretrained(resolved)
        self._model.to(self.device).eval()

        # map model class indices -> our (pos, neg, neu) columns from id2label names
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self._col_for = {}
        for idx, lab in id2label.items():
            if "pos" in lab:
                self._col_for[idx] = "pos"
            elif "neg" in lab:
                self._col_for[idx] = "neg"
            else:  # neutral / anything else
                self._col_for[idx] = "neu"
        self._log.info("SentimentEngine ready: %s on %s (labels=%s)", model_name, self.device, id2label)

    # -- internal: score a batch of ≤max_len token windows -> list of {pos,neg,neu} --
    def _score_windows(self, windows_text: list[str]) -> list[dict[str, float]]:
        torch = self._torch
        out: list[dict[str, float]] = []
        for i in range(0, len(windows_text), self.batch_size):
            batch = windows_text[i : i + self.batch_size]
            enc = self._tok(batch, padding=True, truncation=True, max_length=self.max_tokens, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for row in probs:
                d = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
                for idx, p in enumerate(row):
                    d[self._col_for.get(idx, "neu")] += float(p)
                out.append(d)
        return out

    def score_texts(self, texts: Sequence[str | None]) -> list[dict[str, float] | None]:
        """Score each document into a length-weighted {pos, neg, neu} distribution; blank/None docs -> None.
        Windows of every document are scored in shared batches."""
        stride = max(1, self.max_tokens - 2)  # room for [CLS]/[SEP]
        # 1) tokenize + window each doc, remembering which windows belong to which doc
        all_windows_text: list[str] = []
        owner: list[int] = []
        weights: list[int] = []
        for di, txt in enumerate(texts):
            if not txt or not str(txt).strip():
                continue
            ids = self._tok(str(txt), add_special_tokens=False, truncation=False)["input_ids"]
            for w in _window_ids(ids, stride):
                all_windows_text.append(self._tok.decode(w, skip_special_tokens=True))
                owner.append(di)
                weights.append(len(w))
        # 2) one batched forward pass over ALL windows
        scored = self._score_windows(all_windows_text) if all_windows_text else []
        # 3) length-weighted aggregate per doc
        per_doc_rows: dict[int, list[list[float]]] = {}
        per_doc_w: dict[int, list[float]] = {}
        for row, di, wt in zip(scored, owner, weights, strict=False):
            per_doc_rows.setdefault(di, []).append([row["pos"], row["neg"], row["neu"]])
            per_doc_w.setdefault(di, []).append(float(wt))
        out: list[dict[str, float] | None] = []
        for di in range(len(texts)):
            if di not in per_doc_rows:
                out.append(None)
                continue
            avg = _length_weighted_average(per_doc_rows[di], per_doc_w[di])
            out.append({"pos": avg[0], "neg": avg[1], "neu": avg[2]})
        return out


_ENGINE: SentimentEngine | None = None
_ENGINE_TRIED = False


def get_sentiment_engine(logger: logging.Logger | None = None, model_name: str = FINBERT_TONE_MODEL) -> SentimentEngine | None:
    """The process-wide SentimentEngine, or None if torch/transformers are missing or the model fails to load.
    A failed load is not retried in the same process."""
    global _ENGINE, _ENGINE_TRIED
    log = logger or logging.getLogger(__name__)
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_TRIED:  # already failed once — don't retry
        return None
    _ENGINE_TRIED = True
    if not ml_stack_available():
        log.warning("torch/transformers not installed -> earnings-call sentiment skipped (pip install torch transformers).")
        return None
    try:
        _ENGINE = SentimentEngine(model_name=model_name, logger=log)
    except Exception as e:  # noqa: BLE001 - model download / GPU OOM
        log.warning("Could not load sentiment model '%s' -> sentiment skipped: %s", model_name, e)
        _ENGINE = None
    return _ENGINE
