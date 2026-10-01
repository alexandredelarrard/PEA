"""
backtest.py  (src/modelling/transformers/backtest.py)
-----------------------------------------------------
`Backtest`: a label-only quantile backtest of ANY cross-sectional prediction.

Each date, names are bucketed into `n_quantiles` by the prediction and the realised label is
averaged per bucket. The top-minus-bottom spread is the return (in label units) of a
dollar-neutral quantile book; its cumulative curve, drawdown, hit rate and IR (annualised by the
number of independent horizon-length windows, like the IC) summarise it, together with the
monotonicity of the bucket profile and the membership turnover of the two extreme buckets. It
knows nothing about the model or the strategy, and needs no prices: on the cube the label is the
factor-neutralised forward-return rank, so the spread measures residual ranking skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.modelling.utils.metrics import max_drawdown  # noqa: E402

DEFAULT_N_QUANTILES = 10


@dataclass
class BacktestResult:
    buckets: pd.DataFrame  # date x bucket (1 = lowest prediction) -> mean label
    spread: pd.Series  # date -> top bucket mean - bottom bucket mean
    turnover: pd.DataFrame  # date -> share of the top / bottom bucket that is new vs the previous date
    summary: dict[str, float] = field(default_factory=dict)


def _membership_turnover(members: pd.Series) -> pd.Series:
    """1 - |S_t ∩ S_{t-1}| / |S_t| per date for a date -> set-of-tickers series (NaN on the first)."""
    out, prev = {}, None
    for d, names in members.items():
        out[d] = np.nan if prev is None or not names else 1.0 - len(names & prev) / len(names)
        prev = names
    return pd.Series(out, dtype=float)


class Backtest:
    """Label-only quantile backtest; `model.backtest.n_quantiles` sets the bucket count."""

    def __init__(self, config: DictConfig) -> None:
        bt = config.model.get("backtest") or {}
        self.n_quantiles = int(bt.get("n_quantiles", DEFAULT_N_QUANTILES))
        if self.n_quantiles < 2:
            raise ValueError(f"model.backtest.n_quantiles must be >= 2, got {self.n_quantiles}")

    def run(self, frame: pd.DataFrame, *, label_col: str, pred_col: str = "pred", horizon: int = 1) -> BacktestResult:
        """Bucket `frame` (date, ticker, `pred_col`, `label_col`) per date and summarise. Dates
        with fewer names than buckets are skipped; ties are broken by row order (deterministic)."""
        n_q = self.n_quantiles
        df = frame[["date", "ticker", pred_col, label_col]].dropna(subset=[pred_col, label_col])
        df = df[df.groupby("date")[pred_col].transform("size") >= n_q].copy()
        if df.empty:
            empty = pd.DataFrame(columns=list(range(1, n_q + 1)), dtype=float)
            return BacktestResult(
                empty,
                pd.Series(dtype=float, name="spread"),
                pd.DataFrame(columns=["top", "bottom"], dtype=float),
                {"n_days": 0.0, "n_quantiles": float(n_q)},
            )
        pct = df.groupby("date")[pred_col].rank(method="first", pct=True)
        df["bucket"] = np.clip(np.ceil(pct.to_numpy() * n_q), 1, n_q).astype(int)
        buckets = df.groupby(["date", "bucket"])[label_col].mean().unstack("bucket").reindex(columns=range(1, n_q + 1)).sort_index()
        spread = (buckets[n_q] - buckets[1]).rename("spread")
        top = df[df["bucket"] == n_q].groupby("date")["ticker"].agg(frozenset)
        bottom = df[df["bucket"] == 1].groupby("date")["ticker"].agg(frozenset)
        turnover = pd.DataFrame({"top": _membership_turnover(top), "bottom": _membership_turnover(bottom)}).reindex(buckets.index)
        return BacktestResult(buckets, spread, turnover, self._summary(buckets, spread, turnover, horizon))

    def _summary(self, buckets: pd.DataFrame, spread: pd.Series, turnover: pd.DataFrame, horizon: int) -> dict[str, float]:
        n = int(spread.notna().sum())
        mean, std = float(spread.mean()), float(spread.std())
        depth, length = max_drawdown(spread.cumsum())
        profile = buckets.mean()
        mono = (
            cast(float, spearmanr(np.arange(1, len(profile) + 1), profile.to_numpy())[0])
            if profile.notna().all() and profile.nunique() > 1
            else float("nan")
        )
        return {
            "n_days": float(n),
            "n_quantiles": float(self.n_quantiles),
            "top_mean": float(buckets.iloc[:, -1].mean()),
            "bottom_mean": float(buckets.iloc[:, 0].mean()),
            "spread_mean": mean,
            "spread_std": std,
            "spread_ir": mean / std * float(np.sqrt(252.0 / max(1, int(horizon)))) if (n > 1 and std > 0) else float("nan"),
            "hit_rate": float((spread > 0).mean()) if n else float("nan"),
            "cum_spread": float(spread.sum()),
            "max_drawdown": depth,
            "drawdown_days": float(length),
            "monotonicity": mono,
            "turnover_top": float(turnover["top"].mean()),
            "turnover_bottom": float(turnover["bottom"].mean()),
        }

    @staticmethod
    def save(result: BacktestResult, out_dir: Path, horizon: int) -> None:
        """`backtest_buckets.csv`, `backtest_spread.csv` and `backtest.png` under `out_dir`."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.buckets.to_csv(out_dir / "backtest_buckets.csv")
        cum = result.spread.cumsum()
        pd.DataFrame({"spread": result.spread, "cum_spread": cum, "drawdown": cum - cum.cummax().clip(lower=0.0)}).to_csv(
            out_dir / "backtest_spread.csv"
        )
        if result.spread.empty:
            return
        s = result.summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        profile = result.buckets.mean()
        ax1.bar(profile.index.astype(str), profile.to_numpy(), color="steelblue")
        ax1.set_xlabel("prediction bucket (1 = lowest)")
        ax1.set_ylabel("mean label")
        ax1.set_title(f"Bucket profile — horizon {horizon} (monotonicity {s['monotonicity']:+.2f})", fontsize=10)
        ax2.plot(cum.index, cum.to_numpy(), color="darkorange", lw=1.4)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_ylabel("cumulative top - bottom spread")
        ax2.set_title(f"Spread IR {s['spread_ir']:+.2f}, hit {s['hit_rate']:.0%}, max DD {s['max_drawdown']:+.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "backtest.png", dpi=140)
        plt.close(fig)
