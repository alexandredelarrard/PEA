"""Self-financing, delayed-close simulation of actual held quantities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.portfolio.transformers.construction import construct, risk_panels
from src.portfolio.utils.calendar import cash_return, is_due
from src.portfolio.utils.contracts import IncompleteScenario, MarketInputs, Settings, Simulation


def _positive(values: pd.Series, date: pd.Timestamp, field: str) -> pd.Series:
    bad = values.index[~np.isfinite(values) | (values <= 0)]
    if len(bad):
        raise IncompleteScenario(f"{date.date()}: missing/invalid {field}: {', '.join(map(str, bad))}")
    return values


def _fee(notional: np.ndarray, settings: Settings) -> np.ndarray:
    return (np.abs(notional) * (settings.commission_bps + settings.spread_bps) + np.maximum(-notional, 0) * settings.sell_fee_bps) / 10000


def _funded_capital(capital: float, cost: Callable[[float], float]) -> float:
    """Solve capital_after + costs(capital_after) = capital_before, without credit."""
    try:
        after = brentq(lambda value: value + cost(value) - capital, 0, capital, xtol=1e-12)
    except ValueError as exc:
        raise IncompleteScenario("Transaction costs cannot be funded by available capital") from exc
    if after <= 0 or not np.isclose(after + cost(after), capital, rtol=1e-10, atol=1e-9):
        raise IncompleteScenario("Transaction costs cannot be funded by available capital")
    return after


def _rebalance(
    capital: float,
    coefficients: pd.Series,
    fixed: pd.Series,
    old: pd.Series,
    prices: pd.Series,
    settings: Settings,
) -> tuple[float, pd.Series, pd.Series]:
    """Solve the joint net orders once, with a frozen affine target per instrument."""

    def cost(value: float) -> float:
        return float(_fee(((coefficients * value + fixed - old) * prices).to_numpy(), settings).sum())

    funded = _funded_capital(capital, cost)
    quantities = coefficients * funded + fixed
    fees = pd.Series(_fee(((quantities - old) * prices).to_numpy(), settings), index=prices.index)
    return funded, quantities, fees


def _execution_prices(quotes: pd.Series, date: pd.Timestamp, market_price: float, needs_benchmark: bool) -> pd.Series:
    """Validate execution quotes and materialize the joint stock/benchmark vector."""
    prices = _positive(quotes, date, "execution quote").copy()
    if needs_benchmark and (not np.isfinite(market_price) or market_price <= 0):
        raise IncompleteScenario(f"{date.date()}: missing S&P execution price")
    prices["S&P500_TR"] = market_price if np.isfinite(market_price) and market_price > 0 else 1.0
    return prices


def _order_records(
    date: pd.Timestamp,
    signal_date: pd.Timestamp,
    old: pd.Series,
    quantities: pd.Series,
    prices: pd.Series,
    fees: pd.Series,
    cause: str,
) -> list[dict]:
    """Materialize each nonzero net order from the joint stock/benchmark vectors."""
    delta = quantities - old
    notionals = delta * prices
    records = []
    for ticker in quantities.index:
        if abs(notionals[ticker]) <= 1e-10:
            continue
        before, after = float(old[ticker]), float(quantities[ticker])
        benchmark = ticker == "S&P500_TR"
        change = "entry" if abs(before) <= 1e-12 else ("exit" if abs(after) <= 1e-12 else ("inversion" if before * after < 0 else "adjustment"))
        records.append(
            {
                "date": date,
                "signal_date": signal_date,
                "ticker": ticker,
                "pocket": "sp500" if benchmark else "long_short",
                "cause": "allocation" if benchmark else cause,
                "change": ("entry" if before == 0 else "adjustment") if benchmark else change,
                "old_quantity": before,
                "quantity": after,
                "delta": float(delta[ticker]),
                "price": float(prices[ticker]),
                "notional": float(notionals[ticker]),
                "fee": float(fees[ticker]),
            }
        )
    return records


def _position_records(date: pd.Timestamp, data: MarketInputs, quantities: pd.Series, values: pd.Series, nav: float) -> list[dict]:
    """Snapshot held stock quantities and their already-computed marked values."""
    return [
        {
            "date": date,
            "ticker": ticker,
            "sector": data.sectors[ticker],
            "quantity": float(quantities[ticker]),
            "price": float(data.quotes.loc[date, ticker]),
            "value": float(values[ticker]),
            "weight": float(values[ticker] / nav),
        }
        for ticker in values.index
    ]


def simulate(data: MarketInputs, settings: Settings, n: int, frequency: str) -> Simulation:
    """Keep quantities between schedules; signal at J, fill at J+1 close."""
    if n < 1 or frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("Invalid long/short scenario")
    # Macro prices can be carried over holidays; require a stock/feature session witness.
    sessions = data.quotes.index.union(data.returns.index).union(data.share_basis.index).union(data.scores.index).sort_values().unique()
    data = replace(
        data,
        quotes=data.quotes.reindex(sessions),
        returns=data.returns.reindex(sessions),
        share_basis=data.share_basis.reindex(sessions),
    )
    dates = sessions[(sessions >= settings.start) & (sessions <= settings.end if settings.end is not None else True)]
    if len(dates) < 2:
        raise ValueError("Backtest needs at least two observed sessions")
    tickers = data.quotes.columns
    q = pd.Series(0.0, index=tickers)
    ls_cash, sp_idle, free_cash = (100 * v for v in settings.allocation)
    sp_q, previous_nav = 0.0, 100.0
    daily: list[dict] = []
    orders: list[dict] = []
    positions: list[dict] = []
    variance, momentum = risk_panels(data, settings) if settings.allocation[0] > 0 else (pd.DataFrame(), pd.DataFrame())
    daily.append(
        {
            "date": dates[0],
            "nav": 100.0,
            "long_short": ls_cash,
            "sp500": sp_idle,
            "cash": free_cash,
            "ls_reserved": ls_cash,
            "sp_quantity": 0.0,
            "fee": 0.0,
            "ls_fee": 0.0,
            "sp_fee": 0.0,
            "interest": 0.0,
            "ls_pnl": 0.0,
            "sp_pnl": 0.0,
            "return": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
        }
    )
    status, reason = "complete", None
    for i, date in enumerate(dates[1:], 1):
        previous = dates[i - 1]
        try:
            held = q.index[q != 0]
            before = _positive(data.quotes.loc[previous, held], previous, "held quote")
            quote = _positive(data.quotes.loc[date, held], date, "held quote")
            basis_before = _positive(data.share_basis.loc[previous, held], previous, "share basis")
            basis_now = _positive(data.share_basis.loc[date, held], date, "share basis")
            returns = data.returns.loc[date, held]
            if not np.isfinite(returns).all() or (returns <= -1).any():
                raise IncompleteScenario(f"{date.date()}: missing held total return: {list(returns.index[~np.isfinite(returns)])}")
            old_values = q.loc[held] * before
            ls_pnl = float((old_values * returns).sum())
            q.loc[held] *= basis_before / basis_now
            # Total-return distributions enter reserved cash once; splits change quantities, never orders.
            ls_cash += ls_pnl - float((q.loc[held] * quote).sum() - old_values.sum())
            ls_value = ls_cash + float((q.loc[held] * quote).sum())
            market_now = data.market.get(date, np.nan)
            market_before = data.market.get(previous, np.nan)
            if sp_q and (not np.isfinite(market_now) or market_now <= 0 or not np.isfinite(market_before) or market_before <= 0):
                raise IncompleteScenario(f"{date.date()}: missing S&P total-return benchmark")
            sp_pnl = sp_q * (market_now - market_before) if sp_q else 0.0
            sp_value = sp_q * market_now + sp_idle if sp_q else sp_idle
            interest = free_cash * cash_return(previous, date, settings.cash_rate)
            free_cash += interest
            nav_before = ls_value + sp_value + free_cash
            if not np.isfinite(nav_before) or nav_before <= 0 or (settings.allocation[0] > 0 and ls_value <= 0):
                raise IncompleteScenario(f"{date.date()}: nonpositive portfolio/LS capital")
            outer = i == 1 or is_due(previous, date, settings.allocation_frequency)
            inner = settings.allocation[0] > 0 and (i == 1 or is_due(previous, date, frequency))
            target = construct(data, previous, n, variance, momentum).reindex(tickers, fill_value=0) if inner else None
            if outer or inner:
                active = q.index[(q != 0) | ((target != 0) if target is not None else False)]
                execution_all = _execution_prices(data.quotes.loc[date, active], date, market_now, outer and settings.allocation[1] > 0)
                execution = execution_all.loc[active]
                old_q = q.loc[active].copy()

                multiplier = settings.allocation[0] if outer else 1.0
                coefficients = (
                    target.loc[active] * multiplier / execution if target is not None else old_q * (multiplier / ls_value if ls_value else 0)
                )
                coefficients["S&P500_TR"] = settings.allocation[1] / market_now if outer and settings.allocation[1] > 0 else 0.0
                fixed = coefficients * 0
                fixed["S&P500_TR"] = 0.0 if outer else sp_q
                old = old_q.copy()
                old["S&P500_TR"] = sp_q
                funded, desired, fees = _rebalance(nav_before if outer else ls_value, coefficients, fixed, old, execution_all, settings)
                new_q, new_sp = desired.loc[active], float(desired["S&P500_TR"])
                sp_fee = float(fees["S&P500_TR"])
                ls_fee = float(fees.loc[active].sum())
                cause = "allocation+selection" if outer and inner else ("allocation" if outer else "selection")
                orders.extend(_order_records(date, previous, old, desired, execution_all, fees, cause))
                q.loc[active] = new_q
                ls_value = funded * settings.allocation[0] if outer else funded
                free_cash = funded * settings.allocation[2] if outer else free_cash
                sp_value = funded * settings.allocation[1] if outer else sp_value
                sp_idle = 0.0 if outer else sp_idle
                ls_cash = ls_value - float((new_q * execution).sum())
                sp_q = new_sp
            else:
                ls_fee, sp_fee = 0.0, 0.0
            nav = ls_value + sp_value + free_cash
            fee = ls_fee + sp_fee
            if free_cash < -1e-9 or ls_cash < -1e-9 or not np.isclose(nav, previous_nav + ls_pnl + sp_pnl + interest - fee, atol=1e-8):
                raise IncompleteScenario(f"{date.date()}: unfunded capital or accounting mismatch")
            held = q.index[q != 0]
            values = q.loc[held] * data.quotes.loc[date, held]
            positions.extend(_position_records(date, data, q, values, nav))
            daily.append(
                {
                    "date": date,
                    "nav": nav,
                    "long_short": ls_value,
                    "sp500": sp_value,
                    "cash": free_cash,
                    "ls_reserved": ls_cash,
                    "sp_quantity": sp_q,
                    "fee": fee,
                    "ls_fee": ls_fee,
                    "sp_fee": sp_fee,
                    "interest": interest,
                    "ls_pnl": ls_pnl,
                    "sp_pnl": sp_pnl,
                    "return": nav / previous_nav - 1,
                    "gross_exposure": float(values.abs().sum()) + sp_value,
                    "net_exposure": float(values.sum()) + sp_value,
                }
            )
            previous_nav = nav
        except IncompleteScenario as exc:
            status, reason = "incomplete", f"{date.date()}: {exc}"
            # Do not retain an order from an accounting step that was never published.
            orders = [row for row in orders if row["date"] < date]
            break
    return Simulation(n, frequency, pd.DataFrame(daily).set_index("date"), pd.DataFrame(orders), pd.DataFrame(positions), status, reason)


def without_costs(settings: Settings) -> Settings:
    """The gross shadow book uses exactly the same engine and decision information."""
    return replace(settings, commission_bps=0, spread_bps=0, sell_fee_bps=0)
