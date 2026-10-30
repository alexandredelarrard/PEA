"""Self-financing, delayed-close simulation of actual held quantities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.portfolio.transformers.construction import construct, risk_panels
from src.portfolio.utils.calendar import cash_returns, due_flags
from src.portfolio.utils.contracts import IncompleteScenario, MarketInputs, Settings, Simulation

BENCHMARK = "S&P500_TR"


@dataclass
class _Grid:
    """Every panel aligned on the simulated sessions, with all state-independent vectors precomputed."""

    data: MarketInputs
    dates: pd.DatetimeIndex
    tickers: pd.Index
    sectors: np.ndarray
    quotes: np.ndarray
    basis: np.ndarray
    returns: np.ndarray
    market: np.ndarray
    outer: np.ndarray
    inner: np.ndarray
    interest: np.ndarray
    variance: pd.DataFrame
    momentum: pd.DataFrame


@dataclass
class _Book:
    """Mutable state carried from one session to the next."""

    q: np.ndarray
    ls_cash: float
    sp_idle: float
    free_cash: float
    sp_q: float = 0.0
    nav: float = 100.0


class _Mark(NamedTuple):
    ls_pnl: float
    ls_value: float
    sp_pnl: float
    sp_value: float
    interest: float
    nav_before: float


class _Trade(NamedTuple):
    ls_value: float
    sp_value: float
    ls_fee: float
    sp_fee: float
    orders: pd.DataFrame | None


class _Snapshot(NamedTuple):
    i: int
    held: np.ndarray
    quantity: np.ndarray
    value: np.ndarray
    nav: float


def _is_price(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def _positive(values: np.ndarray, names: pd.Index, date: pd.Timestamp, field: str) -> np.ndarray:
    bad = names[~np.isfinite(values) | (values <= 0)]
    if len(bad):
        raise IncompleteScenario(f"{date.date()}: missing/invalid {field}: {', '.join(map(str, bad))}")
    return values


def _fee(notional: np.ndarray, settings: Settings) -> np.ndarray:
    return (np.abs(notional) * (settings.commission_bps + settings.spread_bps) + np.maximum(-notional, 0) * settings.sell_fee_bps) / 10000


def _funded_capital(capital: float, cost: Callable[[float], float]) -> float:
    """Solve capital_after + costs(capital_after) = capital_before, without credit."""
    try:
        after = cast(float, brentq(lambda value: value + cost(value) - capital, 0, capital, xtol=1e-12))
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
    _positive(quotes.to_numpy(), quotes.index, date, "execution quote")
    if needs_benchmark and not _is_price(market_price):
        raise IncompleteScenario(f"{date.date()}: missing S&P execution price")
    prices = quotes.copy()
    prices[BENCHMARK] = market_price if _is_price(market_price) else 1.0
    return prices


def _order_records(
    date: pd.Timestamp,
    signal_date: pd.Timestamp,
    old: pd.Series,
    quantities: pd.Series,
    prices: pd.Series,
    fees: pd.Series,
    cause: str,
) -> pd.DataFrame:
    """One row per nonzero net order from the joint stock/benchmark vectors."""
    delta = quantities - old
    notional = delta * prices
    keep = (notional.abs() > 1e-10).to_numpy()
    before, after = old.to_numpy()[keep], quantities.to_numpy()[keep]
    benchmark = quantities.index[keep] == BENCHMARK

    # Stock orders classify by sign path; the benchmark is only ever entered or adjusted.
    stock_change = np.select([np.abs(before) <= 1e-12, np.abs(after) <= 1e-12, before * after < 0], ["entry", "exit", "inversion"], "adjustment")
    change = np.where(benchmark, np.where(before == 0, "entry", "adjustment"), stock_change)
    return pd.DataFrame(
        {
            "date": date,
            "signal_date": signal_date,
            "ticker": quantities.index[keep],
            "pocket": np.where(benchmark, "sp500", "long_short"),
            "cause": np.where(benchmark, "allocation", cause),
            "change": change,
            "old_quantity": before,
            "quantity": after,
            "delta": delta.to_numpy()[keep],
            "price": prices.to_numpy()[keep],
            "notional": notional.to_numpy()[keep],
            "fee": fees.to_numpy()[keep],
        }
    )


def _daily_row(date: pd.Timestamp, nav: float, ls_value: float, sp_value: float, cash: float, reserved: float, **flows: float) -> dict:
    """One published session; every flow not given is zero."""
    flow = {
        "sp_quantity": 0.0,
        "ls_fee": 0.0,
        "sp_fee": 0.0,
        "interest": 0.0,
        "ls_pnl": 0.0,
        "sp_pnl": 0.0,
        "ret": 0.0,
        "gross": 0.0,
        "net": 0.0,
        **flows,
    }
    return {
        "date": date,
        "nav": nav,
        "long_short": ls_value,
        "sp500": sp_value,
        "cash": cash,
        "ls_reserved": reserved,
        "sp_quantity": flow["sp_quantity"],
        "fee": flow["ls_fee"] + flow["sp_fee"],
        "ls_fee": flow["ls_fee"],
        "sp_fee": flow["sp_fee"],
        "interest": flow["interest"],
        "ls_pnl": flow["ls_pnl"],
        "sp_pnl": flow["sp_pnl"],
        "return": flow["ret"],
        "gross_exposure": flow["gross"],
        "net_exposure": flow["net"],
    }


def _build_grid(data: MarketInputs, settings: Settings, frequency: str) -> _Grid:
    """Align panels on observed sessions and precompute schedules, interest, benchmark and risk panels."""
    # Macro prices can be carried over holidays; require a stock/feature session witness.
    sessions = data.quotes.index.union(data.returns.index).union(data.share_basis.index).union(data.scores.index).sort_values().unique()
    data = replace(data, quotes=data.quotes.reindex(sessions), returns=data.returns.reindex(sessions), share_basis=data.share_basis.reindex(sessions))
    dates = sessions[(sessions >= settings.start) & (sessions <= settings.end if settings.end is not None else True)]
    if len(dates) < 2:
        raise ValueError("Backtest needs at least two observed sessions")

    # The first session after the opening one always allocates and selects.
    outer, inner = due_flags(dates, settings.allocation_frequency), due_flags(dates, frequency)
    outer[1] = inner[1] = True
    inner &= settings.allocation[0] > 0
    tickers = data.quotes.columns
    variance, momentum = risk_panels(data, settings) if settings.allocation[0] > 0 else (pd.DataFrame(), pd.DataFrame())
    return _Grid(
        data, dates, tickers, data.sectors.reindex(tickers).to_numpy(),
        data.quotes.reindex(dates).to_numpy(float), data.share_basis.reindex(dates).to_numpy(float), data.returns.reindex(dates).to_numpy(float),
        data.market.reindex(dates).to_numpy(float), outer, inner, cash_returns(dates, settings.cash_rate), variance, momentum,
    )  # fmt: skip


def _mark(book: _Book, grid: _Grid, settings: Settings, i: int) -> _Mark:
    """Roll the held book from session i-1 to i: stock P&L, share-basis changes, benchmark P&L and cash interest."""
    date, previous = grid.dates[i], grid.dates[i - 1]
    held = np.flatnonzero(book.q)
    names = grid.tickers[held]
    before = _positive(grid.quotes[i - 1, held], names, previous, "held quote")
    quote = _positive(grid.quotes[i, held], names, date, "held quote")
    basis_before = _positive(grid.basis[i - 1, held], names, previous, "share basis")
    basis_now = _positive(grid.basis[i, held], names, date, "share basis")
    returns = grid.returns[i, held]
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise IncompleteScenario(f"{date.date()}: missing held total return: {list(names[~np.isfinite(returns)])}")

    # Total-return distributions enter reserved cash once; splits change quantities, never orders.
    old_values = book.q[held] * before
    ls_pnl = float((old_values * returns).sum())
    book.q[held] *= basis_before / basis_now
    held_value = float((book.q[held] * quote).sum())
    book.ls_cash += ls_pnl - (held_value - old_values.sum())
    ls_value = book.ls_cash + held_value

    # The S&P pocket moves with the benchmark only once units are held.
    now, then = grid.market[i], grid.market[i - 1]
    if book.sp_q and not (_is_price(now) and _is_price(then)):
        raise IncompleteScenario(f"{date.date()}: missing S&P total-return benchmark")
    sp_pnl = book.sp_q * (now - then) if book.sp_q else 0.0
    sp_value = book.sp_q * now + book.sp_idle if book.sp_q else book.sp_idle

    # Free cash accrues calendar-day interest.
    interest = book.free_cash * grid.interest[i]
    book.free_cash += interest
    nav_before = ls_value + sp_value + book.free_cash
    if not np.isfinite(nav_before) or nav_before <= 0 or (settings.allocation[0] > 0 and ls_value <= 0):
        raise IncompleteScenario(f"{date.date()}: nonpositive portfolio/LS capital")
    return _Mark(ls_pnl, ls_value, sp_pnl, sp_value, interest, nav_before)


def _affine_target(
    book: _Book, settings: Settings, active: np.ndarray, weights: np.ndarray | None, execution: pd.Series, ls_value: float, outer: bool, market: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Frozen affine quantity target: coefficients scale with funded capital, `fixed` stays untouched."""
    stocks = execution.iloc[:-1]
    old_q = pd.Series(book.q[active], index=stocks.index)
    multiplier = settings.allocation[0] if outer else 1.0
    if weights is not None:
        stock = pd.Series(weights[active], index=stocks.index) * multiplier / stocks
    else:
        stock = old_q * (multiplier / ls_value if ls_value else 0)

    # The benchmark leg is funded on allocation sessions and otherwise held as is.
    bench = settings.allocation[1] / market if outer and settings.allocation[1] > 0 else 0.0
    coefficients = pd.concat([stock, pd.Series({BENCHMARK: bench})])
    fixed = pd.Series(0.0, index=coefficients.index)
    fixed[BENCHMARK] = 0.0 if outer else book.sp_q
    return coefficients, fixed, pd.concat([old_q, pd.Series({BENCHMARK: book.sp_q})])


def _trade(book: _Book, grid: _Grid, settings: Settings, n: int, i: int, mark: _Mark) -> _Trade:
    """Rebalance at the close of due sessions: allocation resets every pocket, selection only the long/short one."""
    outer, inner = bool(grid.outer[i]), bool(grid.inner[i])
    if not (outer or inner):
        return _Trade(mark.ls_value, mark.sp_value, 0.0, 0.0, None)
    date, previous, market = grid.dates[i], grid.dates[i - 1], grid.market[i]

    # Signal at the previous close, fill at this close on every name held or targeted.
    weights = construct(grid.data, previous, n, grid.variance, grid.momentum).reindex(grid.tickers, fill_value=0).to_numpy() if inner else None
    active = np.flatnonzero((book.q != 0) | ((weights != 0) if weights is not None else False))
    quotes = pd.Series(grid.quotes[i, active], index=grid.tickers[active])
    execution = _execution_prices(quotes, date, market, outer and settings.allocation[1] > 0)
    coefficients, fixed, old = _affine_target(book, settings, active, weights, execution, mark.ls_value, outer, market)
    funded, desired, fees = _rebalance(mark.nav_before if outer else mark.ls_value, coefficients, fixed, old, execution, settings)
    cause = "allocation+selection" if outer and inner else ("allocation" if outer else "selection")
    orders = _order_records(date, previous, old, desired, execution, fees, cause)

    # Re-split capital across pockets only on allocation sessions.
    stocks = desired.iloc[:-1]
    book.q[active] = stocks.to_numpy()
    book.sp_q = float(desired[BENCHMARK])
    ls_value = funded * settings.allocation[0] if outer else funded
    sp_value = funded * settings.allocation[1] if outer else mark.sp_value
    book.free_cash = funded * settings.allocation[2] if outer else book.free_cash
    book.sp_idle = 0.0 if outer else book.sp_idle
    book.ls_cash = ls_value - float((stocks * execution.iloc[:-1]).sum())
    return _Trade(ls_value, sp_value, float(fees.iloc[:-1].sum()), float(fees[BENCHMARK]), orders)


def _close(book: _Book, grid: _Grid, i: int, mark: _Mark, trade: _Trade) -> tuple[dict, _Snapshot]:
    """Check the day's accounting identity, then publish the daily row and the holdings snapshot."""
    date = grid.dates[i]
    nav = trade.ls_value + trade.sp_value + book.free_cash
    fee = trade.ls_fee + trade.sp_fee
    if book.free_cash < -1e-9 or book.ls_cash < -1e-9 or not np.isclose(nav, book.nav + mark.ls_pnl + mark.sp_pnl + mark.interest - fee, atol=1e-8):
        raise IncompleteScenario(f"{date.date()}: unfunded capital or accounting mismatch")

    held = np.flatnonzero(book.q)
    values = book.q[held] * grid.quotes[i, held]
    row = _daily_row(
        date, nav, trade.ls_value, trade.sp_value, book.free_cash, book.ls_cash,
        sp_quantity=book.sp_q, ls_fee=trade.ls_fee, sp_fee=trade.sp_fee, interest=mark.interest, ls_pnl=mark.ls_pnl, sp_pnl=mark.sp_pnl,
        ret=nav / book.nav - 1, gross=float(np.abs(values).sum()) + trade.sp_value, net=float(values.sum()) + trade.sp_value,
    )  # fmt: skip
    book.nav = nav
    return row, _Snapshot(i, held, book.q[held], values, nav)


def _orders_frame(frames: list[pd.DataFrame | None]) -> pd.DataFrame:
    kept = [frame for frame in frames if frame is not None and len(frame)]
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def _positions_frame(grid: _Grid, snapshots: list[_Snapshot]) -> pd.DataFrame:
    """Stack every day's held names into one frame, built once instead of row by row."""
    kept = [s for s in snapshots if len(s.held)]
    if not kept:
        return pd.DataFrame()
    day = np.concatenate([np.full(len(s.held), s.i) for s in kept])
    held = np.concatenate([s.held for s in kept])
    value = np.concatenate([s.value for s in kept])
    nav = np.concatenate([np.full(len(s.held), s.nav) for s in kept])
    return pd.DataFrame(
        {
            "date": grid.dates[day],
            "ticker": grid.tickers[held],
            "sector": grid.sectors[held],
            "quantity": np.concatenate([s.quantity for s in kept]),
            "price": grid.quotes[day, held],
            "value": value,
            "weight": value / nav,
        }
    )


def simulate(data: MarketInputs, settings: Settings, n: int, frequency: str) -> Simulation:
    """Keep quantities between schedules; signal at J, fill at J+1 close."""
    if n < 1 or frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("Invalid long/short scenario")

    # Align panels and precompute everything that does not depend on the book.
    grid = _build_grid(data, settings, frequency)
    ls_cash, sp_idle, free_cash = (100 * v for v in settings.allocation)
    book = _Book(np.zeros(len(grid.tickers)), ls_cash, sp_idle, free_cash)
    daily = [_daily_row(grid.dates[0], book.nav, ls_cash, sp_idle, free_cash, ls_cash)]
    orders: list[pd.DataFrame | None] = []
    snapshots: list[_Snapshot] = []

    # Day loop: mark the book, trade if due, check accounting, publish.
    status, reason = "complete", None
    for i in range(1, len(grid.dates)):
        try:
            mark = _mark(book, grid, settings, i)
            trade = _trade(book, grid, settings, n, i, mark)
            row, snapshot = _close(book, grid, i, mark, trade)
        except IncompleteScenario as exc:
            status, reason = "incomplete", f"{grid.dates[i].date()}: {exc}"
            break
        daily.append(row)
        orders.append(trade.orders)
        snapshots.append(snapshot)

    # A day that failed publishes nothing: no row, order or position.
    daily_frame = pd.DataFrame(daily).set_index("date")
    return Simulation(n, frequency, daily_frame, _orders_frame(orders), _positions_frame(grid, snapshots), status, reason)


def without_costs(settings: Settings) -> Settings:
    """The gross shadow book uses exactly the same engine and decision information."""
    return replace(settings, commission_bps=0, spread_bps=0, sell_fee_bps=0)
