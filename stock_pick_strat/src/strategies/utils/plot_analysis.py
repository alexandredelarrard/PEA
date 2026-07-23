import matplotlib.pyplot as plt
import numpy as np

def plot_equity(daily, metrics, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot(daily.index, daily["portfolio_value"], label="Strategy", lw=1.6)
    ax1.plot(daily.index, daily["spy_value"], label="SPY (buy & hold)", lw=1.2, alpha=0.8)
    ax1.set_title(f"Backtest — Sharpe {metrics.get('sharpe', float('nan')):.2f} "
                  f"vs SPY {metrics.get('spy_sharpe', float('nan')):.2f}  |  "
                  f"total {metrics.get('total_return', 0)*100:.1f}% "
                  f"vs {metrics.get('spy_total_return', 0)*100:.1f}%")
    ax1.set_ylabel("Portfolio value"); ax1.legend(); ax1.grid(alpha=0.3)
    eq = daily["portfolio_value"].to_numpy()
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    ax2.fill_between(daily.index, dd * 100, 0, color="crimson", alpha=0.4)
    ax2.set_ylabel("Drawdown %"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)