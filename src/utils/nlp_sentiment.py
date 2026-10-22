"""Optional FinBERT scoring with shared batches and token-weighted 510-token windows.

Model labels determine probability columns. CUDA is preferred when available;
FINBERT_DEVICE=cuda requires a working GPU instead of skipping failures.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Sequence
from functools import cache
from importlib import import_module
from pathlib import Path

logger = logging.getLogger(__name__)

FINBERT_TONE_MODEL = "yiyanghkust/finbert-tone"
EARNINGS_CALL_SENTIMENT_CACHE_MODEL = f"{FINBERT_TONE_MODEL}:speaker-clean-v2"
EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL = f"{EARNINGS_CALL_SENTIMENT_CACHE_MODEL}:invalid-pending"
EARNINGS_CALL_SENTIMENT_INVALID_HANDLED_MODEL = f"{EARNINGS_CALL_SENTIMENT_CACHE_MODEL}:invalid-handled"


def ml_stack_available() -> bool:
    """True if both torch and transformers can be imported (does NOT load a model)."""
    return bool(importlib.util.find_spec("torch")) and bool(importlib.util.find_spec("transformers"))


def ensure_local_model(model_name: str) -> str:
    """Mirror public model files through curl_cffi, falling back to the HF client on I/O failure.

    Try the CA bundle before a logged unverified fetch for corporate proxies.
    """
    base = os.getenv("PEA_MODEL_DIR") or str(Path.home() / ".cache" / "pea_models")
    dest = Path(base) / model_name.replace("/", "__")
    weight_files = ("model.safetensors", "pytorch_model.bin")
    meta_files = ("vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "merges.txt", "vocab.json")
    if (dest / "config.json").exists() and any((dest / w).exists() for w in weight_files):
        return str(dest)  # already mirrored
    try:
        cffi = import_module("curl_cffi.requests")
    except ModuleNotFoundError as error:
        if error.name != "curl_cffi":
            raise
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
            except OSError:
                continue
            if r.status_code == 404:
                return False  # legitimately absent (e.g. no safetensors)
            if r.status_code != 200 or not r.content:
                continue
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
        if not any(_fetch(w, required=False) for w in weight_files):
            raise RuntimeError("no weight file (safetensors / pytorch_model.bin) found")
        for filename in meta_files:
            _fetch(filename, required=False)  # tokenizer files, best-effort
    except (OSError, RuntimeError) as e:
        logger.warning("curl_cffi model mirror failed for %s (%s) -> trying HF client.", model_name, e)
        return model_name
    logger.info("Mirrored %s locally to %s (curl_cffi).", model_name, dest)
    return str(dest)


def _window_ids(ids: list[int], stride: int) -> list[list[int]]:
    """Split a token-id list into consecutive non-overlapping windows of <=`stride` tokens; [] for empty input."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    return [ids[i : i + stride] for i in range(0, len(ids), stride)]


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


class SentimentEngine:
    """Thin wrapper over a HuggingFace tone classifier; build it via `get_sentiment_engine`."""

    def __init__(self) -> None:
        torch = import_module("torch")
        transformers = import_module("transformers")
        model_name = FINBERT_TONE_MODEL
        self._torch = torch
        self.max_tokens = 512
        self.batch_size = 16

        device_env = os.getenv("FINBERT_DEVICE", "").strip().lower()
        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available and device_env != "cpu" else "cpu"
        logger.info(
            "FinBERT device=%s (FINBERT_DEVICE=%s, torch=%s, CUDA build=%s, CUDA available=%s)",
            self.device,
            device_env or "auto",
            torch.__version__,
            torch.version.cuda,
            cuda_available,
        )
        if device_env == "cuda" and not cuda_available:
            raise RuntimeError(
                "FINBERT_DEVICE=cuda but CUDA is unavailable. Check the CUDA-enabled PyTorch build, "
                "NVIDIA driver and Docker GPU reservation (docker-compose.gpu.yml)."
            )
        if self.device == "cuda":
            logger.info("FinBERT GPU: %s", torch.cuda.get_device_name())

        # mirror locally via curl_cffi first (corporate-proxy workaround), else HF client
        resolved = ensure_local_model(model_name)
        self._tok = transformers.AutoTokenizer.from_pretrained(resolved)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(resolved)
        self._model.to(self.device).eval()

        # map model class indices -> our (pos, neg, neu) columns from id2label names
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self._col_for: dict[int, str] = {}
        for idx, lab in id2label.items():
            if "pos" in lab:
                self._col_for[idx] = "pos"
            elif "neg" in lab:
                self._col_for[idx] = "neg"
            else:  # neutral / anything else
                self._col_for[idx] = "neu"
        logger.info("SentimentEngine ready: %s on %s (labels=%s)", model_name, self.device, id2label)

    def _score_windows(self, windows_text: list[str]) -> list[dict[str, float]]:
        """Score bounded windows in shared batches, mapping model labels to tone columns."""
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

        scored = self._score_windows(all_windows_text) if all_windows_text else []
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


@cache
def get_sentiment_engine() -> SentimentEngine | None:
    """Cache one engine or optional load failure; required-CUDA failures always propagate."""
    requires_cuda = os.getenv("FINBERT_DEVICE", "").strip().lower() == "cuda"
    if not ml_stack_available():
        if requires_cuda:
            raise RuntimeError("FINBERT_DEVICE=cuda requires torch and transformers to be installed.")
        logger.warning("torch/transformers not installed -> earnings-call sentiment skipped (pip install torch transformers).")
        return None
    try:
        return SentimentEngine()
    except (OSError, RuntimeError) as e:
        if requires_cuda:
            raise
        logger.warning("Could not load sentiment model '%s' -> sentiment skipped: %s", FINBERT_TONE_MODEL, e)
        return None
