"""Offline report payloads, performance math and atomic recovery."""

from __future__ import annotations

import base64
import gzip
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.portfolio.transformers.metrics import metrics
from src.portfolio.transformers.report import write_report
from src.portfolio.transformers.simulation import simulate
from tests.portfolio.test_simulation import market, settings


def test_report_has_nine_real_paths_and_preserves_prior_on_render_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, options = market(), settings(allocation=(0, 0, 1))
    results = [(simulate(data, options, n, f), simulate(data, options, n, f)) for n in (10, 50, 100) for f in ("daily", "weekly", "monthly")]
    benchmark = simulate(data, replace(options, allocation=(0, 1, 0)), 10, "monthly")
    target = tmp_path / "report.html"
    write_report(target, {"run_id": "known-truth"}, options, results, benchmark, benchmark)
    html = target.read_text(encoding="utf-8")
    manifest = json.loads(re.search(r'id="portfolio-manifest">(.*?)</script>', html, re.S).group(1))
    payloads = json.loads(re.search(r'id="portfolio-payloads">(.*?)</script>', html, re.S).group(1))
    assert len(manifest["scenarios"]) == len(payloads) == 9
    decoded = json.loads(gzip.decompress(base64.b64decode(payloads["10_weekly"])))
    assert decoded["daily"][0]["nav"] == 100
    assert "<script src=" not in html and "<link " not in html
    assert metrics(results[0][0], options)["sharpe"] is None
    original = target.read_bytes()
    from src.portfolio.transformers import report

    monkeypatch.setattr(report, "get_plotlyjs", lambda: (_ for _ in ()).throw(RuntimeError("render failure")))
    with pytest.raises(RuntimeError, match="render failure"):
        write_report(target, {"run_id": "known-truth"}, options, results, benchmark, benchmark)
    assert target.read_bytes() == original
    print("SANITY: nine saved simulations decode offline; initial100 survives setup convention and render failure preserves prior bytes.")


def test_metrics_use_net_excess_returns_and_all_months() -> None:
    data, options = market(), settings(allocation=(0, 1, 0))
    data.market.loc["2022-01-05":] = np.linspace(100, 120, len(data.market.loc["2022-01-05":]))
    result = simulate(data, options, 10, "monthly")
    measured = metrics(result, options)
    days = result.daily.index.to_series().diff().dt.days.iloc[1:]
    excess = result.daily["return"].iloc[1:] - (1.033 ** (days / 365) - 1)
    assert measured["sharpe"] == pytest.approx(np.sqrt(252) * excess.mean() / excess.std(ddof=1))
    assert set(measured["monthly_changes"]) == {"2022-01", "2022-02"}
    assert measured["changes_per_month"] == pytest.approx(len(result.orders) / 2)
    assert measured["max_drawdown"] <= 0
    print("SANITY: Sharpe uses net calendar-interval excess returns; zero-trade months stay in monthly change averages.")
