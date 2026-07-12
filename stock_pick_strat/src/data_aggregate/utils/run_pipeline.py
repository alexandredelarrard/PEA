"""
run_pipeline.py
---------------
End-to-end glue with REAL data. Requires: yfinance, xgboost, pandas, numpy,
scipy, scikit-learn.

Flow:
    yfinance download  ->  normalize (data_utils)
    peer dict          ->  sector returns (sector_peers)
    rolling betas      ->  (betas)
    epsilon + rank     ->  labels per horizon (targets)
    price alphas       ->  feature panel (features)
    XGBoost ranker     ->  purged-CV IC + live signal (model)

The final object you trade is `signal`: one score per stock for today, higher =
long side. Feed it to your portfolio optimizer (the next stage we discussed:
maximize alpha - risk - cost, under beta / sector / style neutrality).
"""

import numpy as np
import pandas as pd

import data_utils as du
import sector_peers as sp
import betas as bt
import targets as tg
import features as ft
import model as ml

# --------------------------------------------------------------------------- #
# 0. Download (uncomment to run for real)                                     #
# --------------------------------------------------------------------------- #
# import yfinance as yf
# TICKERS = [...]                      # your 500 SP500 constituents
# raw = yf.download(TICKERS + ["SPY"], start="2015-01-01", auto_adjust=True,
#                   group_by="column")
# raw.to_parquet("prices.parquet")
raw = pd.read_parquet("prices.parquet")

HORIZONS = (5, 10, 20, 60)
PRIMARY_H = 20          # anchor horizon for the live signal (cost-aware choice)
MARKET = "SPY"

# --------------------------------------------------------------------------- #
# 1. Normalize                                                                #
# --------------------------------------------------------------------------- #
close = du.extract_field(raw, "Close")
open_ = du.extract_field(raw, "Open")
returns = du.daily_returns(close)
mkt_ret, stock_ret = du.split_market(returns, MARKET)
stock_close = close.drop(columns=[MARKET])
stock_open = open_.drop(columns=[MARKET])

# --------------------------------------------------------------------------- #
# 2. Peers -> sector returns                                                  #
#    For an honest backtest, replace with build_peer_dict_rolling recomputed  #
#    on a schedule (e.g. monthly) so peers are point-in-time.                 #
# --------------------------------------------------------------------------- #
peers = sp.build_peer_dict(stock_ret, top_k=10, weighting="corr")
sector_ret = sp.compute_sector_returns(stock_ret, peers)

# --------------------------------------------------------------------------- #
# 3. Rolling shrunk betas (market + orthogonalized sector)                    #
# --------------------------------------------------------------------------- #
beta_dict = bt.estimate_all_betas(stock_ret, mkt_ret, sector_ret,
                                  window=63, min_obs=40, shrink_weight=0.7)

# --------------------------------------------------------------------------- #
# 4. Targets: epsilon -> cross-sectional rank, per horizon                    #
# --------------------------------------------------------------------------- #
labels = tg.build_targets(stock_close, close[MARKET], stock_ret,
                          peers, beta_dict, horizons=HORIZONS, label="rank")

# --------------------------------------------------------------------------- #
# 5. Features (cross-sectionally ranked price alphas)                         #
# --------------------------------------------------------------------------- #
feat_panel = ft.build_feature_panel(stock_close, stock_open, sector_ret,
                                    method="rank")

# --------------------------------------------------------------------------- #
# 6. Model: validate with purged CV, then fit on all history                  #
# --------------------------------------------------------------------------- #
panel = ml.make_panel(feat_panel, labels[PRIMARY_H], "y")
feats = ml.feature_columns(panel, "y")

cv = ml.cross_validate(panel, feats, "y", n_splits=5, embargo=PRIMARY_H)
print("Purged walk-forward IC by fold:")
for i, r in enumerate(cv):
    print(f"  fold {i}: mean_IC={r['mean_ic']:+.4f}  IC_IR={r['ic_ir']:+.2f}")
print(f"avg mean_IC = {np.nanmean([r['mean_ic'] for r in cv]):+.4f}")

booster = ml.train_ranker(panel, feats, "y")

# --------------------------------------------------------------------------- #
# 7. Live signal for the most recent date -> hand to the optimizer            #
# --------------------------------------------------------------------------- #
last_date = feat_panel["date"].max()
today = feat_panel[feat_panel["date"] == last_date].dropna(subset=feats)
today = today.sort_values("ticker").reset_index(drop=True)
today["score"] = ml.predict(booster, today, feats).to_numpy()
signal = today.set_index("ticker")["score"].sort_values(ascending=False)

print(f"\nTop 10 longs for {last_date.date()}:")
print(signal.head(10).round(4).to_string())
print(f"\nTop 10 shorts for {last_date.date()}:")
print(signal.tail(10).round(4).to_string())

# Optional: blend multiple horizons instead of a single one.
# Train one ranker per horizon, standardize each day's scores, IR-weight-average.
