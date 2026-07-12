"""
Backtest a top-N factor portfolio against SPY.

HONEST CAVEAT (read this before trusting any backtest number):
Because free fundamentals data only gives us a CURRENT snapshot (see
data/fetch_fundamentals.py), we cannot know what the factor scores actually
WERE 5 or 10 years ago without historical fundamentals. Two modes here:

  - `run_static_backtest()`: picks today's top-N stocks by today's factor
    scores, then looks at their historical price performance. This has
    obvious lookahead bias (you're using current knowledge of "good"
    companies and checking if they did well) — it tells you nothing
    reliable about whether the STRATEGY would have worked, only how the
    stocks happened to perform. Useful only as a sanity check / demo.

  - `run_walk_forward_backtest()`: the correct approach — at each rebalance
    date, score stocks using ONLY fundamentals known as of that date, then
    hold forward. This requires historical fundamentals (e.g. from SimFin
    bulk export, or from your own accumulated `fundamentals_history.parquet`
    over time — see fetch_fundamentals.py). Implemented here so it's ready
    to use once you have real historical fundamentals; it will raise a
    clear error if your fundamentals history doesn't have enough distinct
    dates yet.
"""
import pandas as pd
import numpy as np

from strategy.factors import compute_factor_scores, top_n


def _monthly_rebalance_dates(price_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    df = pd.DataFrame({"date": price_dates})
    df["ym"] = df["date"].dt.to_period("M")
    return df.groupby("ym")["date"].first().tolist()


def _portfolio_equity_curve(prices: pd.DataFrame, holdings_by_date: dict,
                             starting_value: float = 100.0) -> pd.Series:
    """Given {rebalance_date: [tickers]} equal-weighted, compute equity curve."""
    prices = prices.sort_values(["ticker", "date"])
    pivot = prices.pivot(index="date", columns="ticker", values="close")
    returns = pivot.pct_change()

    dates = sorted(returns.index)
    equity = pd.Series(index=dates, dtype=float)
    value = starting_value
    current_holdings = None

    rebalance_dates = sorted(holdings_by_date.keys())
    for d in dates:
        applicable = [rd for rd in rebalance_dates if rd <= d]
        current_holdings = holdings_by_date[applicable[-1]] if applicable else None

        if current_holdings:
            valid = [t for t in current_holdings if t in returns.columns]
            day_ret = returns.loc[d, valid].mean(skipna=True) if valid else 0.0
            if pd.isna(day_ret):
                day_ret = 0.0
            value *= (1 + day_ret)
        equity[d] = value

    return equity


def run_static_backtest(prices: pd.DataFrame, fundamentals: pd.DataFrame,
                         n: int = 30, benchmark_ticker: str = "SPY") -> pd.DataFrame:
    """See module docstring — this mode has lookahead bias, demo/sanity-check only."""
    scored = compute_factor_scores(fundamentals)
    picks = top_n(scored, n)["ticker"].tolist()

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    first_date = prices["date"].min()

    port_equity = _portfolio_equity_curve(prices, {first_date: picks})
    bench_equity = _portfolio_equity_curve(prices, {first_date: [benchmark_ticker]})

    result = pd.DataFrame({
        "date": port_equity.index,
        "portfolio": port_equity.values,
        "benchmark": bench_equity.reindex(port_equity.index).values,
    })
    result.attrs["mode"] = "static (lookahead-biased, demo only)"
    result.attrs["holdings"] = picks
    return result


def run_walk_forward_backtest(prices: pd.DataFrame, fundamentals_history: pd.DataFrame,
                               n: int = 30, benchmark_ticker: str = "SPY") -> pd.DataFrame:
    """
    Correct point-in-time backtest. Requires fundamentals_history with
    multiple distinct `as_of` dates (built up over time by repeatedly
    running fetch_fundamentals.py, or loaded from a SimFin bulk export).
    """
    distinct_dates = fundamentals_history["as_of"].nunique()
    if distinct_dates < 2:
        raise ValueError(
            f"fundamentals_history only has {distinct_dates} distinct date(s). "
            "Walk-forward backtesting needs fundamentals snapshots at multiple "
            "points in time. Either (a) keep running fetch_fundamentals.py "
            "periodically to accumulate history, or (b) load a SimFin bulk "
            "export via load_simfin_bulk() in data/fetch_fundamentals.py."
        )

    holdings_by_date = {}
    for as_of, group in fundamentals_history.groupby("as_of"):
        scored = compute_factor_scores(group)
        picks = top_n(scored, n)["ticker"].tolist()
        holdings_by_date[pd.Timestamp(as_of)] = picks

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    port_equity = _portfolio_equity_curve(prices, holdings_by_date)
    bench_equity = _portfolio_equity_curve(prices, {min(holdings_by_date): [benchmark_ticker]})

    result = pd.DataFrame({
        "date": port_equity.index,
        "portfolio": port_equity.values,
        "benchmark": bench_equity.reindex(port_equity.index).values,
    })
    result.attrs["mode"] = "walk-forward (point-in-time, correct)"
    return result


def performance_stats(equity: pd.Series) -> dict:
    """CAGR, vol, Sharpe (rf=0), max drawdown for an equity curve series."""
    rets = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol > 0 else np.nan
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    return {
        "CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "Max Drawdown": max_dd,
    }
