"""
nlp_sentiment.py  (src/utils/nlp_sentiment.py)
----------------------------------------------
Local, FREE, GPU-capable finance sentiment scoring for long documents (earnings-call
sections). Wraps a HuggingFace tone classifier (default FinBERT-tone) behind a small,
dependency-light API so the rest of the pipeline never imports torch/transformers
directly:

    engine = get_sentiment_engine(context.log)      # None if torch/transformers absent
    probs  = engine.score_texts([doc1, doc2, ...])   # -> [{'pos','neg','neu'}|None, ...]

Design notes
  * LAZY / OPTIONAL. torch + transformers are heavy optional deps. If either is
    missing the engine builder returns None and callers skip cleanly (same pattern as
    curl_cffi in the Google-Trends fetcher) — the pipeline must never hard-break just
    because the ML stack isn't installed on a given machine.
  * GPU-FIRST, FITS 6GB. The model (~440MB) loads on CUDA when available (else CPU);
    inference runs in small batches under torch.no_grad() so a 6GB card is plenty.
  * LONG DOCS. Earnings-call sections run to thousands of tokens; BERT caps at 512.
    We split each doc into ≤510-token windows, score every window, and LENGTH-WEIGHT
    average the per-window class probabilities back to one distribution per doc.
  * LABEL-SAFE. Class order differs across models, so we map columns from the model's
    own `config.id2label` (match on 'pos'/'neg'/'neu') rather than hardcoding indices.

The pure windowing/aggregation helpers (`_window_ids`, `_length_weighted_average`) are
unit-tested without any model download.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence
from src.constants.constants import FINBERT_TONE_MODEL

# FinBERT-tone: finance-domain tone classifier (positive / neutral / negative),
# trained on analyst reports & earnings text. ~440MB, runs locally on GPU (fits 6GB)
# or CPU; free (HuggingFace). Sections longer than the 512-token BERT window are
# chunked and length-weighted (see src/utils/nlp_sentiment.py).
FINBERT_MAX_TOKENS = 512

def ml_stack_available() -> bool:
    """True if both torch and transformers can be imported (does NOT load a model)."""
    import importlib.util as u
    return bool(u.find_spec("torch")) and bool(u.find_spec("transformers"))


# --------------------------------------------------------------------------- #
# Corporate-proxy model mirror (curl_cffi)                                      #
# --------------------------------------------------------------------------- #
# The default HuggingFace downloader uses OpenSSL, which on OpenSSL 3.x REJECTS some
# corporate MITM-proxy CAs ("Basic Constraints of CA cert not marked critical"), so the
# model can't be fetched behind the proxy. curl_cffi impersonates a real Chrome TLS
# handshake (BoringSSL) — the repo's proven proxy workaround (see Google Trends) — and
# mirrors the model files locally so transformers can then load them OFFLINE.
_MODEL_META_FILES = ("config.json", "vocab.txt", "tokenizer_config.json",
                     "special_tokens_map.json", "tokenizer.json", "merges.txt", "vocab.json")
_MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _local_model_dir(model_name: str) -> Path:
    base = os.getenv("PEA_MODEL_DIR") or str(Path.home() / ".cache" / "pea_models")
    return Path(base) / model_name.replace("/", "__")


def ensure_local_model(model_name: str, logger: logging.Logger) -> str:
    """Mirror a HuggingFace model locally via curl_cffi and return the local dir (so
    `from_pretrained` loads it offline). Corporate-proxy workaround: tries the CA
    bundle first, then — as a last resort for this PUBLIC, read-only model artifact —
    an UNVERIFIED fetch (logged). Returns `model_name` unchanged if curl_cffi is
    unavailable or the download fails, so the normal HF client is still attempted."""
    dest = _local_model_dir(model_name)
    if (dest / "config.json").exists() and any((dest / w).exists() for w in _MODEL_WEIGHT_FILES):
        return str(dest)                              # already mirrored
    try:
        from curl_cffi import requests as cffi
    except Exception:                                 # curl_cffi absent -> let HF try
        return model_name

    ca = next((os.environ[v] for v in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE",
                                       "CURL_CA_BUNDLE") if os.environ.get(v)), None)
    base_url = f"https://huggingface.co/{model_name}/resolve/main/"

    def _fetch(fname: str, required: bool) -> bool:
        out = dest / fname
        if out.exists():
            return True
        for verify in ([ca, False] if ca else [True, False]):
            try:
                r = cffi.Session(impersonate="chrome124", verify=verify,
                                 timeout=180).get(base_url + fname)
            except Exception:
                continue
            if r.status_code == 404:
                return False                          # legitimately absent (e.g. no safetensors)
            if r.status_code == 200 and r.content:
                if verify is False:
                    logger.warning("Fetched %s UNVERIFIED via curl_cffi (public model "
                                   "behind corporate proxy).", fname)
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
                _fetch(f, required=False)             # tokenizer files, best-effort
    except Exception as e:                            # noqa: BLE001
        logger.warning("curl_cffi model mirror failed for %s (%s) -> trying HF client.",
                       model_name, e)
        return model_name
    logger.info("Mirrored %s locally to %s (curl_cffi).", model_name, dest)
    return str(dest)


# --------------------------------------------------------------------------- #
# Pure helpers (no torch) — unit-tested                                         #
# --------------------------------------------------------------------------- #
def _window_ids(ids: list[int], stride: int) -> list[list[int]]:
    """Split a token-id list into consecutive non-overlapping windows of ≤`stride`
    tokens. Empty input -> one empty window is NOT produced (returns [])."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    return [ids[i:i + stride] for i in range(0, len(ids), stride)] if ids else []


def _length_weighted_average(prob_rows: Sequence[Sequence[float]],
                             weights: Sequence[float]) -> list[float]:
    """Length-weighted mean of per-window probability vectors -> one vector. Falls back
    to a plain mean if all weights are non-positive; empty input -> []."""
    rows = [list(map(float, r)) for r in prob_rows]
    if not rows:
        return []
    n = len(rows[0])
    w = [max(float(x), 0.0) for x in weights]
    tot = sum(w)
    if tot <= 0:                                   # degenerate -> uniform mean
        w = [1.0] * len(rows)
        tot = float(len(rows))
    out = [0.0] * n
    for row, wi in zip(rows, w):
        for j in range(n):
            out[j] += wi * row[j]
    return [v / tot for v in out]


# --------------------------------------------------------------------------- #
# Engine                                                                        #
# --------------------------------------------------------------------------- #
class SentimentEngine:
    """Thin wrapper over a HuggingFace sequence-classification tone model. Built via
    `get_sentiment_engine` (which returns None when the ML stack is unavailable)."""

    def __init__(self, model_name: str = FINBERT_TONE_MODEL,
                 max_tokens: int = FINBERT_MAX_TOKENS,
                 batch_size: int = 16,
                 logger: logging.Logger | None = None) -> None:
        import torch                                  # local import (heavy, optional)
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

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
            else:                                     # neutral / anything else
                self._col_for[idx] = "neu"
        self._log.info("SentimentEngine ready: %s on %s (labels=%s)",
                       model_name, self.device, id2label)

    # -- internal: score a batch of ≤max_len token windows -> list of {pos,neg,neu} --
    def _score_windows(self, windows_text: list[str]) -> list[dict[str, float]]:
        torch = self._torch
        out: list[dict[str, float]] = []
        for i in range(0, len(windows_text), self.batch_size):
            batch = windows_text[i:i + self.batch_size]
            enc = self._tok(batch, padding=True, truncation=True,
                            max_length=self.max_tokens, return_tensors="pt").to(self.device)
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
        """Score each document into a length-weighted {pos, neg, neu} distribution.
        Blank/None docs -> None. Long docs are windowed to ≤(max_tokens-2) tokens and
        every window scored, then length-weighted back to one distribution per doc.
        All windows across all docs are batched together for GPU efficiency."""
        stride = max(1, self.max_tokens - 2)          # room for [CLS]/[SEP]
        # 1) tokenize + window each doc, remembering which windows belong to which doc
        all_windows_text: list[str] = []
        owner: list[int] = []
        weights: list[int] = []
        for di, txt in enumerate(texts):
            if not txt or not str(txt).strip():
                continue
            ids = self._tok(str(txt), add_special_tokens=False,
                            truncation=False)["input_ids"]
            for w in _window_ids(ids, stride):
                all_windows_text.append(self._tok.decode(w, skip_special_tokens=True))
                owner.append(di)
                weights.append(len(w))
        # 2) one batched forward pass over ALL windows
        scored = self._score_windows(all_windows_text) if all_windows_text else []
        # 3) length-weighted aggregate per doc
        per_doc_rows: dict[int, list[list[float]]] = {}
        per_doc_w: dict[int, list[float]] = {}
        for row, di, wt in zip(scored, owner, weights):
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


def get_sentiment_engine(logger: logging.Logger | None = None,
                         model_name: str = FINBERT_TONE_MODEL) -> SentimentEngine | None:
    """Return a cached SentimentEngine, or None if torch/transformers are unavailable
    or the model fails to load. Loads the model on first call (downloads ~440MB to the
    HuggingFace cache once); subsequent calls reuse the in-process instance."""
    global _ENGINE, _ENGINE_TRIED
    log = logger or logging.getLogger(__name__)
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_TRIED:                                 # already failed once — don't retry
        return None
    _ENGINE_TRIED = True
    if not ml_stack_available():
        log.warning("torch/transformers not installed -> earnings-call sentiment skipped "
                    "(pip install torch transformers).")
        return None
    try:
        _ENGINE = SentimentEngine(model_name=model_name, logger=log)
    except Exception as e:                            # noqa: BLE001 - model download / GPU OOM
        log.warning("Could not load sentiment model '%s' -> sentiment skipped: %s",
                    model_name, e)
        _ENGINE = None
    return _ENGINE
