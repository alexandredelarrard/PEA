"""Tests for compute_sector_returns (src/data_peers/utils/sector_peers.py).

The sector return is a weighted average of a stock's peer returns. It must be
NaN-TOLERANT: a single missing peer on a date (e.g. a peer that only listed
recently) must NOT wipe out the whole date. The original implementation used a
raw matrix product `returns[cols] @ w`, which propagates a single NaN to the
entire date -- this silently truncated stocks' beta/target history (see
test_targets.py) to the short window where every peer happened to be listed.

The EMPTY-BASKET block at the bottom pins the other failure mode, the one NaN tolerance cannot
help with: a ticker with no embedding row gets no basket at all at `w_corr: 0`, and every
peer-relative feature is NaN for its whole history with nothing raising.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_peers.utils.embeddings import (
    DESCRIPTION_TICKER_ALIAS, fetch_business_descriptions)
from src.data_peers.utils.sector_peers import (
    build_peer_dict_hybrid, combine_similarity, compute_sector_returns, build_peer_dict,
    cosine_similarity_matrix, dedupe_share_classes,
)


def test_dual_class_excluded_from_peer_candidates():
    """A secondary share class (GOOG) must NEVER be anyone's peer (it's the same
    company as GOOGL, correlation ~1.0), and it inherits GOOGL's basket so it still
    has valid, non-self peers."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({t: rng.normal(0, 0.01, len(dates))
                       for t in ["AAA", "BBB", "CCC", "DDD", "GOOGL"]}, index=dates)
    df["GOOG"] = df["GOOGL"] + rng.normal(0, 1e-6, len(dates))     # ~identical twin

    peers = build_peer_dict(df, top_k=3, weighting="corr", min_obs=50,
                            redundant_map={"GOOG": "GOOGL"})

    # GOOG is never a peer CANDIDATE for anyone (incl. GOOGL)
    assert all("GOOG" not in basket for basket in peers.values()), "GOOG leaked as a peer"
    assert "GOOG" not in peers["GOOGL"] and "GOOGL" not in peers["GOOGL"]
    # GOOG inherits GOOGL's basket and never contains itself or its twin
    assert peers["GOOG"] == peers["GOOGL"]
    assert "GOOG" not in peers["GOOG"] and "GOOGL" not in peers["GOOG"]

    print("\n=== SANITY CHECK: dual-class peer dedup (build) ===")
    print(f"  GOOG (~identical to GOOGL) is NOT a peer of any stock; "
          f"GOOG inherits GOOGL's basket = {sorted(peers['GOOG'])}. No self/twin peers. Validated.")


def test_dedupe_share_classes_fixes_cached_dict():
    """The load-path post-processor strips secondaries from a cached dict, renormalizes
    the survivors, and gives each secondary its primary's basket (no re-embedding)."""
    cached = {
        "AAA": {"GOOG": 0.5, "BBB": 0.5},        # GOOG must be stripped + renormalized
        "GOOGL": {"AAA": 0.6, "GOOG": 0.4},      # its own twin must be stripped
        "GOOG": {"AAA": 1.0},                     # will inherit GOOGL's cleaned basket
    }
    out = dedupe_share_classes(cached, {"GOOG": "GOOGL"})
    assert out["AAA"] == {"BBB": 1.0}             # 0.5 -> renorm 1.0
    assert out["GOOGL"] == {"AAA": 1.0}           # 0.6 -> renorm 1.0
    assert out["GOOG"] == out["GOOGL"]            # secondary inherits primary
    assert dedupe_share_classes(out, {"GOOG": "GOOGL"}) == out   # idempotent

    print("\n=== SANITY CHECK: dual-class dedup (cached load) ===")
    print(f"  GOOG stripped from AAA -> {out['AAA']}; GOOGL twin stripped -> {out['GOOGL']}; "
          f"GOOG inherits GOOGL. Idempotent. Fixes existing cache without re-embedding. Validated.")


def _returns_with_late_peer():
    dates = pd.bdate_range("2020-01-01", periods=250)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(0, 0.01, size=(len(dates), 3)),
        index=dates, columns=["AAA", "BBB", "LATE"],
    )
    # LATE only starts trading 80% of the way through.
    df.loc[df.index[: int(0.8 * len(df))], "LATE"] = np.nan
    return df


def test_sector_return_is_nan_tolerant():
    returns = _returns_with_late_peer()
    peers = {"AAA": {"BBB": 0.5, "LATE": 0.5}}

    sector = compute_sector_returns(returns, peers)["AAA"]

    # Before LATE lists, the sector return must fall back to the available
    # peer (BBB) rather than being NaN for the whole early period.
    early = returns.index[: int(0.8 * len(returns))]
    assert sector.loc[early].notna().mean() > 0.99, (
        "sector return collapsed to NaN where a single peer was missing"
    )

    # When only BBB is available, sector return == BBB return (weights renormalize).
    np.testing.assert_allclose(
        sector.loc[early].to_numpy(),
        returns.loc[early, "BBB"].to_numpy(),
        rtol=1e-9, atol=1e-12,
    )

    print("\n=== SANITY CHECK: sector return NaN-tolerance ===")
    print(f"  peer LATE missing for first 80% of dates.")
    print(f"  sector non-null over that window = {sector.loc[early].notna().mean():.2%}")
    print("  -> falls back to available peers instead of nuking the date. Correct.")


def test_sector_return_equals_weighted_mean_when_all_present():
    returns = _returns_with_late_peer()
    peers = {"AAA": {"BBB": 0.3, "LATE": 0.7}}
    sector = compute_sector_returns(returns, peers)["AAA"]

    both = returns.dropna(subset=["BBB", "LATE"]).index
    expected = 0.3 * returns.loc[both, "BBB"] + 0.7 * returns.loc[both, "LATE"]
    np.testing.assert_allclose(sector.loc[both].to_numpy(), expected.to_numpy(),
                               rtol=1e-9, atol=1e-12)

    print("\n=== SANITY CHECK: sector return == weighted peer mean when all present ===")
    print("  -> matches the plain weighted average exactly. Correct.")


# --------------------------------------------------------------------------- #
# the empty basket: FISV's 26 years of NaN peer features                       #
# --------------------------------------------------------------------------- #
def _hybrid_inputs(missing: str = "FISV"):
    """Four correlated names plus one with NO embedding row -- the live shape of the defect."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(7)
    tickers = ["FIS", "GPN", "MA", "V", missing]
    returns = pd.DataFrame({t: rng.normal(0, 0.01, len(dates)) for t in tickers}, index=dates)
    embeddings = pd.DataFrame(
        rng.normal(0, 1, (4, 16)), index=[t for t in tickers if t != missing],
        columns=[f"e{i}" for i in range(16)])
    return returns, cosine_similarity_matrix(embeddings)


def test_a_ticker_with_no_embedding_gets_an_empty_basket_at_w_corr_zero():
    """The MECHANISM, pinned. `combine_similarity`'s renormalization is not a fallback when
    correlation carries weight 0: both weights are 0, the denominator is NaN, and the whole
    row is NaN. Pinned so a future config change that resurrects the fallback is visible as a
    failing test rather than as a silent change in peer baskets."""
    returns, embed_sim = _hybrid_inputs()
    combined = combine_similarity(returns.corr(min_periods=50), embed_sim,
                                  w_corr=0.0, w_embed=1.0)
    peers = build_peer_dict_hybrid(returns, embed_sim, top_k=3, weighting="corr",
                                   min_obs=50, w_corr=0.0, w_embed=1.0)

    assert combined.loc["FISV"].isna().all()
    assert peers["FISV"] == {}
    assert all(basket for t, basket in peers.items() if t != "FISV")
    # ... and at w_corr > 0 the fallback the docstring used to promise IS live
    with_corr = build_peer_dict_hybrid(returns, embed_sim, top_k=3, weighting="corr",
                                       min_obs=50, w_corr=0.5, w_embed=0.5)
    assert with_corr["FISV"]

    print("\n=== SANITY CHECK: no embedding -> empty basket at w_corr: 0 ===")
    print(f"  FISV similarity row all-NaN -> basket {peers['FISV']} (every other name has "
          f"{len(peers['FIS'])} peers).")
    print(f"  at w_corr: 0.5 the correlation fallback IS live -> "
          f"{sorted(with_corr['FISV'])}. Validated.")


def test_an_empty_basket_makes_every_peer_feature_nan_for_the_whole_history():
    """Why the guard has to exist: nothing downstream complains."""
    returns, embed_sim = _hybrid_inputs()
    peers = build_peer_dict_hybrid(returns, embed_sim, top_k=3, weighting="corr",
                                   min_obs=50, w_corr=0.0, w_embed=1.0)
    sector = compute_sector_returns(returns, peers)

    assert sector["FISV"].isna().all()
    assert sector["FIS"].notna().any()

    print("\n=== SANITY CHECK: an empty basket is silent, not loud ===")
    print(f"  FISV sector_ret non-null over {len(sector)} dates = "
          f"{int(sector['FISV'].notna().sum())}; FIS = "
          f"{int(sector['FIS'].notna().sum())}. compute_sector_returns just SKIPS the name "
          "and returns normally. Validated.")


def test_load_peers_or_raise_names_the_peerless_ticker(tmp_path, monkeypatch):
    """The guard, at the grain the damage happens at. The old whole-dict check passed a dict
    with 490 good baskets and one empty one."""
    from src.data_aggregate.utils.common import peers_io

    good = {"AAA": {"BBB": 1.0}, "BBB": {"AAA": 1.0}}
    bad = {**good, "FISV": {}, "ZZZ": {}}

    class _Ctx:
        def __init__(self, blob):
            path = tmp_path / f"peers_{len(blob)}.json"
            path.write_text(json.dumps(blob), encoding="utf-8")
            self.paths = {"SECTOR_PEERS_PATH": path}

    assert peers_io.load_peers_or_raise(_Ctx(good)) == good
    with pytest.raises(RuntimeError) as err:
        peers_io.load_peers_or_raise(_Ctx(bad))

    message = str(err.value)
    assert "FISV" in message and "ZZZ" in message
    assert "2 of 4" in message

    print("\n=== SANITY CHECK: load_peers_or_raise checks PER TICKER ===")
    print("  a dict with 2 good baskets passes; adding 2 empty ones raises:")
    print(f"  {message.splitlines()[0][:150]}")
    print("  both peerless tickers are NAMED. Validated.")


def test_the_description_fetch_queries_the_alias_and_stores_the_universe_symbol():
    """An aliased ticker is queried under the symbol that HAS a profile and STORED under the
    universe's, so nothing downstream has to know about the alias."""
    assert DESCRIPTION_TICKER_ALIAS["FISV"] == "FIV.DE"
    queried: list[str] = []

    class _Info:
        def __init__(self, ticker):
            queried.append(ticker)

        @property
        def info(self):
            return {"sector": "Technology", "industry": "Information Technology Services",
                    "longBusinessSummary": "Fiserv, Inc. provides payment and financial "
                                           "technology services worldwide, at length."}

    import src.data_peers.utils.embeddings as emb
    monkey = pytest.MonkeyPatch()
    monkey.setattr(emb.yf, "Ticker", _Info)
    try:
        out = fetch_business_descriptions(["FISV"], store=None, pause=0.0)
    finally:
        monkey.undo()

    assert queried == [DESCRIPTION_TICKER_ALIAS["FISV"]]   # the QUERY was aliased
    assert list(out) == ["FISV"]                           # the KEY is the universe's symbol
    assert "Fiserv" in out["FISV"]

    print("\n=== SANITY CHECK: rebrand alias on the description fetch ===")
    print(f"  universe symbol 'FISV' -> Yahoo queried as {queried[0]!r}, stored back under "
          f"{list(out)[0]!r}. Validated.")


def test_a_ticker_with_no_usable_description_is_logged_at_warning(caplog):
    """The signal that was missing. An absent or stub `longBusinessSummary` used to fall
    through the `if` in silence -- only an exception warned."""
    class _Empty:
        def __init__(self, ticker):
            pass

        @property
        def info(self):
            return {"sector": "Technology", "longBusinessSummary": None}

    import src.data_peers.utils.embeddings as emb
    monkey = pytest.MonkeyPatch()
    monkey.setattr(emb.yf, "Ticker", _Empty)
    try:
        with caplog.at_level("WARNING", logger=emb.logger.name):
            out = fetch_business_descriptions(["NOPE"], store=None, pause=0.0)
    finally:
        monkey.undo()

    assert out == {}
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("no usable business description" in m and "NOPE" in m for m in warnings), warnings

    print("\n=== SANITY CHECK: a missing description now WARNS ===")
    print(f"  {warnings[0][:160]}")
    print("  -> the failure that hid FISV for the whole life of the feature is now audible. "
          "Validated.")
