"""
step_cube_text.py  (src/data_aggregate/transformers/step_cube_text.py)
------------------------------------------------------------------
Earnings-call TEXT analysis -> `cube_part_text`.

Two independent passes over the transcript archive:

SENTIMENT   local FinBERT-tone + Loughran-McDonald scoring (cached/incremental in
            `earnings_call_sentiment`, so the GPU pass runs once), then tone, momentum,
            Q&A-vs-scripted candor, hedging, and disclosure-length KPIs.
EMBEDDING   OpenAI embeddings (cached/incremental; a no-op without an API key), then
            Q&A coherence and consecutive-quarter narrative drift. The final contract
            contains raw levels plus four prior-only issuer-history scores; peer and
            cross-sectional variants are intentionally absent.

MEMORY: neither pass preloads `earnings_call_sections`. Scoring streams the text per ticker
and the KPIs stream back per ticker. Loading that table whole is precisely what OOM-killed
this work before, and it is the only reason the two passes can share one step at all.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PART_REFRESH_TRADING_DAYS, plan_window, write_part
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames,
    load_price_frames,
    load_trading_calendar,
)
from src.data_aggregate.utils.text.earnings_call_embeddings import (
    embed_earnings_calls,
    embedding_kpis_streamed,
)
from src.data_aggregate.utils.text.earnings_call_features import (
    attach_issuer_identity,
    build_earnings_call_feature_panel,
    score_earnings_calls,
    sentiment_kpis_streamed,
)
from src.data_store.schema import Tables
from src.utils.step import Step


class StepCubeText(Step):
    # The price fields this step reads back from `cube_part_prices`. Declared rather than
    # inlined so the projection stays introspectable (see test_part_registry.py).
    _FIELDS = ("close_split",)

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_text)
        self._store = context.store

    def run(self, full: bool = False) -> None:
        calendar = load_trading_calendar(self._store)
        window = plan_window(
            self._store,
            Tables.cube_part_text,
            full=full,
            warmup=self._warmup(),
            trading_index=calendar,
            refresh=PART_REFRESH_TRADING_DAYS,
        )
        # EC histories are sparse and five-year/prior-call based. Compute on the same full
        # calendar for both paths; `write_part` alone slices the incremental tail.
        frames = self._load_frames(None)
        panel, changed = self._feature_panel(frames)
        del frames
        refresh_from = min(changed) + pd.offsets.BDay(1) if changed else None
        n = write_part(self._store, Tables.cube_part_text, panel, window, refresh_from=refresh_from, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(self._store, peers=load_peers_or_raise(self._context, self._config), fields=self._FIELDS, since=since)

    def _feature_panel(self, frames: PriceFrames) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
        changed = [date for date in (score_earnings_calls(self._context), embed_earnings_calls(self._context)) if date is not None]
        per_call = sentiment_kpis_streamed(self._context)
        embedding, as_of = embedding_kpis_streamed(self._context)
        if embedding is not None and not embedding.empty and as_of is not None:
            dates = as_of.drop_duplicates(["ticker", "quarter"])
            embedding = embedding.merge(dates, on=["ticker", "quarter"], how="left")
            per_call = (
                embedding
                if per_call is None or per_call.empty
                else per_call.merge(embedding.drop(columns=["as_of"]), on=["ticker", "quarter"], how="outer")
            )
            if "as_of_x" in per_call.columns:
                per_call["as_of"] = per_call["as_of_x"].fillna(per_call["as_of_y"])
                per_call = per_call.drop(columns=["as_of_x", "as_of_y"])
        if per_call is None or per_call.empty:
            return pd.DataFrame(columns=["date", "ticker"]), changed

        tenure = self._store.load(Tables.symbol_tenure, columns=["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings"], optional=True)
        lineage = self._store.load(Tables.entity_lineage, columns=["cik", "entity_id"], optional=True)
        per_call = attach_issuer_identity(per_call, tenure, lineage)
        panel = build_earnings_call_feature_panel(
            None,
            frames.peers,
            frames.trading_index,
            per_call=per_call,
            availability=frames.availability,
        )
        return panel, changed
