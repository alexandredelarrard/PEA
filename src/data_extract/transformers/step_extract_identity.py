"""Identity step: `symbol_tenure`, `entity_lineage` and `security_master` built offline from the cached zips and the DB,
then the lineage change carried into the stored SEC rows.

Order: the Notes cover-page and FTD captures first, so the builders read this run's evidence (the FTD capture scopes
on the lineage and master as stored before this run's rebuild), then tenure -> lineage -> master -> propagation. One
Form 3/4/5 scan per run. A stage run alone reads its input table from the store.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.identity.capture_dei import capture_pending_dei
from src.data_extract.utils.identity.capture_ftd import capture_ftd_cusips
from src.data_extract.utils.identity.entity_lineage import build_entity_lineage, load_stored_lineage
from src.data_extract.utils.identity.identity_propagate import PropagationResult, propagate_identity
from src.data_extract.utils.identity.security_master import build_security_master
from src.data_extract.utils.identity.symbol_tenure import Form345Scan, build_symbol_tenure, load_built_tenure, scan_form345_cache
from src.utils.step import Step
from src.utils.string import run_date
from src.utils.universe import load_universe_tickers


@dataclass(frozen=True)
class IdentityOptions:
    """The `identity` command's options: the propagated ticker subset (None: the universe), `full` re-capture and
    re-check, `dry_run` propagation, the run date and the approved `(old, new)` entity re-keys."""

    tickers: list[str] | None = None
    full: bool = False
    dry_run: bool = False
    as_of: pd.Timestamp | None = None
    approved_rekeys: frozenset[tuple[str, str]] = frozenset()


class StepExtractIdentity(Step):
    def __init__(self, context: Context, config: DictConfig, options: IdentityOptions | None = None) -> None:
        super().__init__(context=context, config=config)
        self._options = options or IdentityOptions()
        paths = config.local.paths
        self._insider_cache = cache_dir(context, paths.insider_transactions)
        self._notes_cache = cache_dir(context, paths.financial_notes)
        self._ftd_cache = cache_dir(context, paths.fails_deliver)
        self._years_history = int(config.data_extract.years_history)
        self._redundant_symbols = frozenset(config.data_extract.redundant_ticks)
        self._all_tickers = load_universe_tickers(context)
        self._tickers = self._options.tickers or self._all_tickers
        self._run_date = run_date(self._options.as_of)
        self._scan: Form345Scan | None = None
        self._tenure: pd.DataFrame | None = None
        self._lineage: pd.DataFrame | None = None

    def _form345_scan(self) -> Form345Scan:
        if self._scan is None:
            self._scan = scan_form345_cache(self._insider_cache)
        return self._scan

    def run_capture_dei_symbols(self) -> int:
        """Notes cover-page `dei:TradingSymbol` evidence of the cached periods not stored yet (`full`: all)."""
        return capture_pending_dei(self._context, self._notes_cache, full=self._options.full)

    def run_capture_ftd_cusips(self) -> int:
        """In-scope raw FTD lines of the pending cached periods; a no-op before the first lineage build."""
        return capture_ftd_cusips(self._context, self._ftd_cache, self._years_history, full=self._options.full, as_of=self._run_date)

    def run_symbol_tenure(self) -> pd.DataFrame:
        self._tenure = build_symbol_tenure(self._context, self._form345_scan(), str(self._config_dir))
        return self._tenure

    def run_entity_lineage(self) -> pd.DataFrame:
        tenure = self._tenure if self._tenure is not None else load_built_tenure(self._context)
        self._lineage = build_entity_lineage(
            self._context,
            tenure,
            self._form345_scan().owner_pairs,
            str(self._config_dir),
            approved_rekeys=self._options.approved_rekeys,
            redundant_symbols=self._redundant_symbols,
        )
        return self._lineage

    def run_security_master(self) -> pd.DataFrame:
        lineage = self._lineage if self._lineage is not None else load_stored_lineage(self._context)
        if lineage is None:
            raise ValueError("security_master needs entity_lineage: run `identity --only lineage` first")
        return build_security_master(self._context, lineage, str(self._config_dir))

    def run_propagate(self) -> PropagationResult:
        """Carry the lineage changes into the stored rows, under the resolver reloaded once from the rebuilt tables."""
        return propagate_identity(
            self._context,
            self._tickers,
            dry_run=self._options.dry_run,
            every_ticker=self._options.full,
            identity=load_identity(self._context, refresh=True),
            as_of=self._run_date,
            all_tickers=self._all_tickers,
        )

    def run(self) -> None:
        self.run_capture_dei_symbols()
        self.run_capture_ftd_cusips()
        self.run_symbol_tenure()
        self.run_entity_lineage()
        self.run_security_master()
        self.run_propagate()
