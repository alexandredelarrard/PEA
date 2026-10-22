"""Orchestrate one saved ensemble, nine independent books, and an offline report."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.modelling.utils.artifacts import read_metadata
from src.portfolio.transformers.report import write_report
from src.portfolio.transformers.signals import load_scores
from src.portfolio.transformers.simulation import simulate, without_costs
from src.portfolio.utils.contracts import Settings
from src.portfolio.utils.inputs import load_market
from src.utils.step import Step


class StepPortfolio(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)

    def run(self) -> Path:
        settings = Settings.from_config(self._config)
        directory = Path(self._config.model.models_dir)
        metadata = read_metadata(directory)
        data = load_market(self._context, self._config, settings, metadata)
        end = data.quotes.index.max()
        data.scores = load_scores(self._context, self._config, directory, metadata, end)
        metadata = {
            **metadata,
            "read_frontier": {
                "quotes": str(end),
                "scores": str(data.scores.index.max()),
                "market": str(data.market.index.max()),
                "universe_size": len(data.quotes.columns),
            },
        }
        benchmark_settings = replace(settings, allocation=(0, 1, 0))
        benchmark = simulate(data, benchmark_settings, 10, "monthly")
        benchmark_gross = simulate(data, without_costs(benchmark_settings), 10, "monthly")
        if benchmark.status != "complete":
            raise ValueError(f"Incomplete benchmark: {benchmark.reason}; prior report preserved")
        results = []
        for n in (10, 50, 100):
            for frequency in ("daily", "weekly", "monthly"):
                self._log.info("Simulating N=%s cadence=%s", n, frequency)
                result = simulate(data, settings, n, frequency)
                gross = simulate(data, without_costs(settings), n, frequency)
                gross.orders, gross.positions = pd.DataFrame(), pd.DataFrame()
                results.append((result, gross))
                self._log.info("N=%s cadence=%s: %s %s", n, frequency, result.status, result.reason or "")
        if not any(result.status == "complete" for result, _ in results):
            raise ValueError("All scenarios incomplete; prior report preserved: " + str([r.reason for r, _ in results]))
        report = write_report(Path(self._context.paths["OUTPUT_DIR"]) / "report.html", metadata, settings, results, benchmark, benchmark_gross)
        self._log.info("Portfolio report: %s", report)
        return report
