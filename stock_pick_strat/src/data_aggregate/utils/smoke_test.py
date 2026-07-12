"""
Smoke test: generate a synthetic yfinance-style panel with a KNOWN factor
structure (market + sector + idiosyncratic), run the full pipeline, and check
that (a) shapes/timing are right and (b) the epsilon actually strips market and
sector (residual should correlate weakly with market/sector, strongly with the
injected idiosyncratic component).
"""
import numpy as np
import pandas as pd

import data_utils as du
import sector_peers as sp
import betas as bt
import targets as tg
import features as ft
import model as ml

rng = np.random.default_rng(0)

# ---- synthetic universe: 3 sectors x 20 stocks + SPY, 900 trading days ----
n_days = 900
dates = pd.bdate_range("2019-01-01", periods=n_days)
n_sectors, per_sector = 3, 20
tickers = [f"S{sec}_{i:02d}" for sec in range(n_sectors) for i in range(per_sector)]

mkt = rng.normal(0.0004, 0.010, n_days)                       # market factor
sec_factors = rng.normal(0, 0.008, (n_sectors, n_days))       # sector factors

daily_ret = {}
true_idio = {}
for sec in range(n_sectors):
    for i in range(per_sector):
        tk = f"S{sec}_{i:02d}"
        bm = rng.uniform(0.8, 1.2)     # market beta
        bs = rng.uniform(0.8, 1.2)     # sector beta
        # Persistent (AR) idiosyncratic component -> residual MOMENTUM, which
        # the peer_mom / momentum features can genuinely pick up. This makes
        # the smoke test show the model learning a real (planted) signal.
        idio = np.zeros(n_days)
        shock = rng.normal(0, 0.010, n_days)
        for t in range(1, n_days):
            idio[t] = 0.97 * idio[t - 1] + shock[t]
        idio = idio - idio.mean()
        r = bm * mkt + bs * sec_factors[sec] + idio
        daily_ret[tk] = r
        true_idio[tk] = idio
daily_ret["SPY"] = mkt

ret_df = pd.DataFrame(daily_ret, index=dates)
close = (1 + ret_df).cumprod() * 100.0
open_ = close.shift(1) * (1 + rng.normal(0, 0.002, close.shape))
open_.iloc[0] = close.iloc[0]

# emulate a yfinance MultiIndex download
raw = pd.concat({"Close": close, "Open": open_}, axis=1)

# ---- 1. normalize ----
close_w = du.extract_field(raw, "Close")
open_w = du.extract_field(raw, "Open")
returns = du.daily_returns(close_w)
mkt_ret, stock_ret = du.split_market(returns, "SPY")
stock_close = close_w.drop(columns=["SPY"])
stock_open = open_w.drop(columns=["SPY"])
print("shapes:", close_w.shape, "| stocks:", stock_ret.shape[1])

# ---- 2. peer dict + sector returns ----
peers = sp.build_peer_dict(stock_ret, top_k=8, weighting="corr")
sector_ret = sp.compute_sector_returns(stock_ret, peers)
# sanity: a stock's top peers should mostly be from its own sector
same_sector = []
for tk, pd_ in peers.items():
    own = tk.split("_")[0]
    if pd_:
        same_sector.append(np.mean([p.startswith(own) for p in pd_]))
print(f"peer purity (share of peers in same true sector): {np.mean(same_sector):.2f}")

# ---- 3. betas ----
betas = bt.estimate_all_betas(stock_ret, mkt_ret, sector_ret,
                              window=63, min_obs=40, shrink_weight=0.7)
bm_est = np.nanmean([betas[t]["beta_m"].mean() for t in betas])
print(f"mean estimated market beta (true ~1.0): {bm_est:.2f}")

# ---- 4. targets (epsilon + rank) for one horizon ----
H = 10
eps = tg.compute_epsilon(stock_close, close_w["SPY"], stock_ret, peers, betas, H)
label = tg.cross_sectional_rank(eps)
print("epsilon frame:", eps.shape, "| non-null:", int(eps.notna().sum().sum()))

# check residualization worked: corr(eps, forward market) should be near 0,
# corr(eps, forward idio) should be positive.
fwd_mkt = tg.forward_return(close_w[["SPY"]], H)["SPY"]
idio_df = pd.DataFrame(true_idio, index=dates)
fwd_idio = tg.forward_return((1 + idio_df).cumprod(), H)
c_mkt, c_idio = [], []
for tk in eps.columns[:30]:
    a = eps[tk]
    c_mkt.append(a.corr(fwd_mkt))
    c_idio.append(a.corr(fwd_idio[tk]))
print(f"corr(eps, fwd market): {np.nanmean(c_mkt):+.3f}  (want ~0)")
print(f"corr(eps, fwd idio):   {np.nanmean(c_idio):+.3f}  (want >0)")

# ---- 5. features ----
panel_feats = ft.build_feature_panel(stock_close, stock_open, sector_ret, method="rank")
print("feature panel:", panel_feats.shape, "| cols:", list(panel_feats.columns))

# ---- 6. modeling plumbing (sklearn stand-in for xgboost, which isn't installed) ----
panel = ml.make_panel(panel_feats, label, "y")
feats = ml.feature_columns(panel, "y")
print("model panel:", panel.shape, "| features:", len(feats))

# purged walk-forward splits
n_folds = 0
from sklearn.ensemble import HistGradientBoostingRegressor
fold_ic = []
for tr_days, te_days in ml.purged_wf_splits(panel["date"], n_splits=4, embargo=H):
    tr = panel[panel["date"].isin(tr_days)]
    te = panel[panel["date"].isin(te_days)]
    if tr.empty or te.empty:
        continue
    reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                        max_iter=200, l2_regularization=1.0)
    reg.fit(tr[feats], tr["y"])
    preds = pd.Series(reg.predict(te[feats]), index=te.index)
    fold_ic.append(ml.daily_ic(te, preds, "y"))
    n_folds += 1
print(f"CV folds run: {n_folds}")
for i, r in enumerate(fold_ic):
    print(f"  fold {i}: mean_IC={r['mean_ic']:+.3f}  IC_IR={r['ic_ir']:+.2f}  days={r['n_days']}")
print("SMOKE TEST OK")
