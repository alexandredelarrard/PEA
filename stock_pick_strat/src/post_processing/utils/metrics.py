import numpy as np

def compute_metrics(daily, rf_annual=0.0):
    r = daily["net_ret"].to_numpy()
    if len(r) == 0:
        return {}
    rf_daily = rf_annual / 252.0

    def _stats(rets, equity):
        ann = (1 + rets).prod() ** (252 / len(rets)) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = ((rets - rf_daily).mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
        peak = np.maximum.accumulate(equity)
        return ann, vol, sharpe, float(((equity - peak) / peak).min())

    s_ann, s_vol, s_sharpe, s_dd = _stats(r, daily["portfolio_value"].to_numpy())
    spy_r = daily["spy_value"].pct_change().fillna(0).to_numpy()
    b_ann, b_vol, b_sharpe, b_dd = _stats(spy_r, daily["spy_value"].to_numpy())
    return {"days": len(r),
            "total_return": float(daily["portfolio_value"].iloc[-1] / daily["portfolio_value"].iloc[0] - 1),
            "ann_return": float(s_ann), "ann_vol": float(s_vol), "sharpe": float(s_sharpe),
            "max_drawdown": s_dd, "avg_daily_turnover": float(daily["turnover"].mean()),
            "avg_daily_cost": float(daily["cost"].mean()),
            "spy_total_return": float(daily["spy_value"].iloc[-1] / daily["spy_value"].iloc[0] - 1),
            "spy_ann_return": float(b_ann), "spy_sharpe": float(b_sharpe), "spy_max_drawdown": b_dd}
