"""One offline HTML artifact; each filter selects a separately simulated book."""

from __future__ import annotations

import base64
import gzip
import json
import os
import tempfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from src.modelling.utils.assessment import json_value
from src.portfolio.transformers.metrics import metrics
from src.portfolio.utils.contracts import Settings, Simulation


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=12))


def _encoded(value: Any) -> str:
    return base64.b64encode(gzip.compress(json.dumps(json_value(value), allow_nan=False, separators=(",", ":")).encode())).decode("ascii")


def _benchmark_window(benchmark: Simulation, dates: pd.DatetimeIndex) -> Simulation:
    """Retain the shared initial capital and require every compared benchmark observation."""
    if benchmark.daily.empty or benchmark.daily.index[0] != dates[0] or not dates.isin(benchmark.daily.index).all():
        raise ValueError("Missing benchmark observations or initial capital date for scenario window")
    daily = benchmark.daily.loc[dates]
    if not np.isfinite(daily.nav).all() or not daily.nav.gt(0).all():
        raise ValueError("Missing/invalid benchmark NAV for scenario window")
    orders = benchmark.orders.loc[benchmark.orders.date.isin(dates)] if not benchmark.orders.empty else benchmark.orders
    return replace(benchmark, daily=daily, orders=orders, status="complete", reason=None)


def write_report(
    path: Path,
    metadata: dict[str, Any],
    settings: Settings,
    results: list[tuple[Simulation, Simulation]],
    benchmark: Simulation,
    benchmark_gross: Simulation,
) -> Path:
    """Replace the prior report only after rendering one complete self-contained file."""
    if len(results) != 9 or {r.n for r, _ in results} != {10, 50, 100} or {r.frequency for r, _ in results} != {"daily", "weekly", "monthly"}:
        raise ValueError("The report requires all nine separately simulated scenarios")
    manifest = {
        "run_id": metadata["run_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "settings": asdict(settings),
        "models": metadata,
        "execution": "Signal close J; execution next observed session close J+1",
        "limitations": [
            "Current universe and sector classification: survivorship bias",
            "S&P synthetic total-return units, not SPY quotes",
            "Fractional shorts assumed borrowable, no borrow charge",
            "Proportional fee proxy, not historical Alpaca tariffs",
            "Cash interest only on free cash; implicit distributions inferred from total returns",
            "Dollar neutrality at LS rebalance is not beta neutrality; drift is reported",
            "First/last calendar month may be partial; incomplete scenarios are not comparable rankings",
        ],
        "scenarios": {},
        "benchmark": metrics(benchmark, settings),
    }
    payloads = {}
    for result, gross in results:
        daily = result.daily.copy()
        benchmark_window = _benchmark_window(benchmark, daily.index)
        benchmark_gross_window = _benchmark_window(benchmark_gross, daily.index)
        key = f"{result.n}_{result.frequency}"
        manifest["scenarios"][key] = {**metrics(result, settings), "benchmark": metrics(benchmark_window, settings)}
        daily["gross_nav"] = gross.daily.nav.reindex(daily.index)
        daily["benchmark_net"] = benchmark_window.daily.nav
        daily["benchmark_gross"] = benchmark_gross_window.daily.nav
        daily["drawdown"] = daily.nav / daily.nav.cummax() - 1
        daily["benchmark_drawdown"] = daily.benchmark_net / daily.benchmark_net.cummax() - 1
        payloads[key] = _encoded({"daily": _records(daily.reset_index()), "orders": _records(result.orders), "positions": _records(result.positions)})
    template = Path(__file__).with_name("report.html").read_text(encoding="utf-8")
    body = template.replace("__PLOTLY__", get_plotlyjs()).replace(
        "__MANIFEST__", json.dumps(json_value(manifest), allow_nan=False).replace("<", "\\u003c")
    )
    body = body.replace("__PAYLOADS__", json.dumps(payloads))
    if any(marker in body for marker in ("__MANIFEST__", "__PAYLOADS__")):
        raise ValueError("Unresolved portfolio report placeholders")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".backtest-", suffix=".html", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
