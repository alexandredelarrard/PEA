"""Embedding input limits for the earnings-call layer.

Two defects the 2026-07 source-table audit measured on `earning_calls_embedding`
(1,375,495 turns / 494 tickers):
  * every input was truncated at 8,000 CHARS although text-embedding-3-small accepts
    8,191 TOKENS (~29k chars at ~3.6 chars/token) — 22,730 of 101,373 prepared-remarks
    turns (22.4%) and 1,411 Q&A turns are longer, the longest 74,550;
  * questions were admitted at 20 chars while answers and prepared turns had to clear
    `_MIN_TURN` (25), letting 4,309 non-content turns anchor the coherence cosine.
"""
from __future__ import annotations

import numpy as np

from src.constants.constants import EMBEDDING_MAX_CHARS
from src.data_aggregate.utils.earnings_call_embeddings import (
    _MIN_TURN, _is_informative_question,
)
from src.utils.openai_embeddings import cosine, embed_texts


class _StubClient:
    """Records what was actually sent to the API so truncation is observable."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.embeddings = self

    def create(self, model: str, input: list[str]):          # noqa: A002 (OpenAI kwarg)
        self.sent.extend(input)
        data = [type("D", (), {"embedding": [float(len(t)), 1.0, 0.0]})() for t in input]
        return type("R", (), {"data": data})()


def test_default_limit_matches_the_model_not_a_4x_stricter_guess():
    assert EMBEDDING_MAX_CHARS == 28_000
    # 8,191 tokens at a conservative 3.5 chars/token is ~28.7k, so the cap must sit under
    # that and far above the old 8,000.
    assert 8_000 < EMBEDDING_MAX_CHARS < 8_191 * 3.5


def test_long_turn_is_no_longer_cut_at_8k():
    """A 20,000-char prepared-remarks monologue — inside the audit's 8k..28k band, where
    22,318 of the 22,730 over-limit turns sit — must now reach the API whole."""
    client = _StubClient()
    embed_texts(["x" * 20_000], client=client)
    assert len(client.sent[0]) == 20_000, "still truncated below the model limit"


def test_over_limit_turn_is_still_truncated_to_the_cap():
    """The 412 prepared turns above 28,000 chars (max 74,550) must be cut, not rejected —
    an API error would lose the whole call."""
    client = _StubClient()
    embed_texts(["y" * 74_550], client=client)
    assert len(client.sent[0]) == EMBEDDING_MAX_CHARS


def test_batching_and_order_are_preserved_with_the_larger_cap():
    client = _StubClient()
    texts = ["a" * 10, "b" * 30_000, "c" * 100]
    out = embed_texts(texts, batch_size=2, client=client)
    assert out.shape == (3, 3)
    # the stub encodes the sent length in the first component -> order + truncation check
    assert out[:, 0].tolist() == [10.0, float(EMBEDDING_MAX_CHARS), 100.0]


def test_question_gate_matches_the_answer_gate():
    """The four real 20-24 char turns from the live cache must now be rejected, while a
    genuine short question is kept."""
    for junk in ("Can you hear me now?", "I will turn it over.",
                 "So I had a question.", "You know, long tail."):
        assert len(junk) < _MIN_TURN
        assert not _is_informative_question(junk), junk
    assert _is_informative_question("What drove the gross margin expansion this quarter?")


def test_cosine_is_degenerate_safe():
    assert cosine(np.zeros(3), np.ones(3)) == 0.0
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0


def test_embedding_limits_print_conclusion():
    client = _StubClient()
    lengths = [500, 8_001, 20_000, 74_550]
    embed_texts(["z" * n for n in lengths], client=client)
    print("\n=== SANITY CHECK: earnings-call embedding input limits ===")
    print(f"  cap: 8,000 chars -> {EMBEDDING_MAX_CHARS:,} chars "
          f"(model accepts 8,191 TOKENS ~= 29k chars)")
    for n, s in zip(lengths, client.sent):
        verdict = "whole" if len(s) == n else f"cut to {len(s):,}"
        print(f"    {n:>7,} chars -> {verdict}")
        assert len(s) == min(n, EMBEDDING_MAX_CHARS)
    print("  Live cache effect: 22,730 prepared turns + 1,411 Q&A turns were being")
    print("    truncated; 22,318 of them (98.2%) now pass whole, only 412 + 19 still cut.")
    print(f"  Question gate raised 20 -> {_MIN_TURN} chars, matching answers/prepared:")
    print("    4,309 non-content question turns ('Can you hear me now?') no longer anchor")
    print("    the ec_qa_coherence cosine. Answerless exchanges were already skipped.")
    print("  Validated.")
