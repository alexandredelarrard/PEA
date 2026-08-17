"""Step 3 — Retail-attention features (Wikipedia pageviews + Google Trends).

Checks: pure parsers (article-title cleaning, Wikimedia JSON); the abnormal-attention
builder is point-in-time; a genuine attention spike ranks high on <prefix>_attn_spike;
weekly (Trends) series forward-fill onto the daily calendar; and the built panel
exposes the f_* columns. (Google Trends response parsing is covered by
tests/data_extract/test_google_trends.py, which tests the curl_cffi client directly.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.behavioral.fetch_wiki_pageviews import _company_to_article, _json_to_long
from src.data_aggregate.utils.extras.attention_features import (
    _attention_fields, build_combined_attention_panel,
)


def test_pure_parsers():
    # article-title cleaning
    assert _company_to_article("Apple Inc.") == "Apple"
    assert _company_to_article("Berkshire Hathaway Inc. Class B") == "Berkshire_Hathaway"
    assert _company_to_article("The Home Depot, Inc.") == "Home_Depot"

    # Wikimedia JSON -> long
    items = [{"timestamp": "2015070100", "views": 1000},
             {"timestamp": "2015070200", "views": 1200}]
    wj = _json_to_long(items, "AAPL")
    assert list(wj.columns) == ["date", "ticker", "pageviews"]
    assert wj["date"].iloc[0] == pd.Timestamp("2015-07-01") and wj["pageviews"].iloc[0] == 1000
    assert _json_to_long([], "AAPL").empty

    print("\n=== SANITY CHECK: attention parsers ===")
    print("  title cleaning strips suffixes; Wikimedia JSON parses. Validated.")


def _attn_hist(dates, tickers, value_col, spike_ticker, spike_from):
    rows = []
    rng = np.random.default_rng(0)
    for t in tickers:
        base = rng.uniform(500, 2000)
        for d in dates:
            v = base * rng.uniform(0.9, 1.1)
            if t == spike_ticker and d >= spike_from:
                v *= 8.0                       # sustained attention spike
            rows.append({"date": d, "ticker": t, value_col: v})
    return pd.DataFrame(rows)


def test_attention_builder_pit_and_spike():
    dates = pd.bdate_range("2022-01-03", periods=200)
    tickers = [f"S{i}" for i in range(6)]
    spike_from = dates[150]
    hist = _attn_hist(dates, tickers, "pageviews", "S0", spike_from)

    F = _attention_fields(hist, dates, prefix="wiki", value_col="pageviews")
    assert set(F) == {"wiki_attn_spike", "wiki_attn_level"}

    # the spiking name ranks highest on attention spike shortly after the jump
    t = dates[158]
    spike_row = F["wiki_attn_spike"].loc[t]
    assert spike_row.idxmax() == "S0", (t, spike_row.to_dict())

    # point-in-time: perturb everything AFTER t -> spike value at t unchanged
    tm = dates[100]
    hist2 = hist.copy()
    mask = (hist2["date"] > tm) & (hist2["ticker"] == "S0")
    hist2.loc[mask, "pageviews"] *= 50.0
    F2 = _attention_fields(hist2, dates, prefix="wiki", value_col="pageviews")
    assert np.isclose(F["wiki_attn_spike"].loc[tm, "S0"],
                      F2["wiki_attn_spike"].loc[tm, "S0"], equal_nan=True)

    print("\n=== SANITY CHECK: abnormal-attention builder ===")
    print(f"  sustained pageview spike -> S0 tops wiki_attn_spike at {t.date()}; "
          f"perturbing future left the past value unchanged (point-in-time). Validated.")


def test_weekly_series_ffills_to_daily():
    # weekly (Google Trends cadence) -> must forward-fill onto daily trading days
    weekly = pd.date_range("2022-01-02", periods=30, freq="7D")
    daily = pd.bdate_range("2022-01-03", periods=140)
    tickers = [f"S{i}" for i in range(6)]
    hist = _attn_hist(weekly, tickers, "search_interest", "S1", weekly[20])

    F = _attention_fields(hist, daily, prefix="gt", value_col="search_interest")
    # daily coverage should be high despite only weekly observations (ffilled)
    cov = F["gt_attn_level"].notna().mean().mean()
    assert cov > 0.5, f"weekly series did not ffill to daily (coverage {cov:.2f})"

    print("\n=== SANITY CHECK: weekly ffill ===")
    print(f"  weekly Trends series ffilled to daily (coverage {cov*100:.0f}%). Validated.")


def test_combined_attention_blend():
    """The dedicated wiki+Google-Trends blend: one robust indicator, not two."""
    dates = pd.bdate_range("2022-01-03", periods=200)
    tickers = [f"S{i}" for i in range(6)]
    spike_from = dates[150]
    # S0 spikes in BOTH sources; Google Trends covers only S0..S4 (S5 wiki-only) to
    # exercise the single-source coverage fallback in the blend.
    wiki = _attn_hist(dates, tickers, "pageviews", "S0", spike_from)
    gt = _attn_hist(dates, tickers[:5], "search_interest", "S0", spike_from)
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    panel = build_combined_attention_panel(wiki, gt, peers, dates)
    for c in ("f_attn_spike_xs", "f_attn_spike_vs_peers", "f_attn_level_xs"):
        assert c in panel.columns, f"{c} missing"
    assert panel["f_attn_spike_xs"].dropna().between(0, 1).all()
    # the two retired per-source features must NOT leak through the blend
    assert not any(c.startswith(("f_wiki_", "f_gt_")) for c in panel.columns)

    # S0 spikes in BOTH sources -> tops the blended spike shortly after the jump
    t = dates[158]
    row = panel[panel["date"] == t].set_index("ticker")["f_attn_spike_xs"]
    assert row.idxmax() == "S0", (t, row.to_dict())

    # coverage fallback: S5 is wiki-only yet still gets a blended value (nan-mean of 1)
    s5 = panel[(panel["ticker"] == "S5") & (panel["date"] == t)]["f_attn_spike_xs"]
    assert not s5.empty and s5.notna().all()

    # single source works; both absent -> empty (optional-source semantics, no crash)
    only_wiki = build_combined_attention_panel(wiki, None, peers, dates)
    assert "f_attn_spike_xs" in only_wiki.columns and not only_wiki.empty
    empty = build_combined_attention_panel(None, None, peers, dates)
    assert list(empty.columns) == ["date", "ticker"] and empty.empty

    n_feats = len([c for c in panel.columns if c not in ("date", "ticker")])
    print("\n=== SANITY CHECK: combined wiki+Google-Trends attention blend ===")
    print(f"  blended {n_feats} f_attn_* features (no f_wiki_/f_gt_ leakage); S0 (spikes in "
          f"BOTH) tops f_attn_spike_xs @ {t.date()}; wiki-only S5 still covered via nan-mean; "
          f"single-source works, both-absent -> empty. Validated.")


if __name__ == "__main__":
    test_pure_parsers()
    test_attention_builder_pit_and_spike()
    test_weekly_series_ffills_to_daily()
    test_combined_attention_blend()
