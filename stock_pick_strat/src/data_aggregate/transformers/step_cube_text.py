"""
step_cube_text.py  (src/data_aggregate/transformers/step_cube_text.py)
------------------------------------------------------------------
Earnings-call TEXT analysis -> `cube_part_text`.

Two independent passes over the transcript archive:

  SENTIMENT   local FinBERT-tone + Loughran-McDonald scoring (cached/incremental in
              `earnings_call_sentiment`, so the GPU pass runs once), then the per-call KPIs:
              tone level and momentum, the Q&A-vs-scripted candor gap, the hedging
              (uncertainty) ratio, disclosure-length change and vocabulary novelty.
  EMBEDDING   OpenAI embeddings (cached/incremental; a no-op without an API key), then the
              Q&A-coherence (cosine of a question vs its answer) and quarter-to-quarter
              narrative-drift KPIs. Independent of the sentiment pass -- its call dates come
              from the embedding rows' own `as_of` -- so it needs no GPU tone model.

MEMORY: neither pass preloads `earnings_call_sections`. Scoring streams the text per ticker
and the KPIs stream back per ticker. Loading that table whole is precisely what OOM-killed
this work before, and it is the only reason the two passes can share one step at all.
"""
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import CUBE_PART_TEXT
from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.part_io import PartStore
from src.data_aggregate.utils.common.parts import PART_BY_NAME
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.text.earnings_call_embeddings import (
    embed_earnings_calls, embedding_kpis_streamed,
)
from src.data_aggregate.utils.text.earnings_call_features import (
    build_earnings_call_embedding_panel, build_earnings_call_feature_panel,
    score_earnings_calls, sentiment_kpis_streamed,
)
from src.utils.step import Step


class StepCubeText(Step):

    _FIELDS = ("close",)

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = PART_BY_NAME[CUBE_PART_TEXT]
        self._market_ticker = str(self._cfg.market_ticker)
        self._parts = PartStore(context.store, self._log)

    def run(self, full: bool = False) -> None:
        window = plan_window(self._parts, CUBE_PART_TEXT, full=full,
                             warmup=self._warmup(),
                             trading_index=load_trading_calendar(self._parts))
        frames = self._load_frames(window.since)

        merger = PanelMerger(self._log)
        merger.add(frames.skeleton().assign(_grid=1.0), "universe-grid")
        merger.add(self._sentiment_panel(frames), "earnings-call sentiment",
                   "No earnings-call sentiment cache -> sentiment features skipped.")
        merger.add(self._embedding_panel(frames), "earnings-call embedding",
                   "No earnings-call embeddings -> embedding features skipped "
                   "(no transcripts / model or API key absent).")

        panel = merger.to_long().drop(columns=["_grid"], errors="ignore")
        del frames
        n = write_part(self._parts, CUBE_PART_TEXT, panel, window, self._log)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(
            self._parts, peers=load_peers_or_raise(self._context, self._config),
            market_ticker=self._market_ticker, fields=self._FIELDS, since=since)

    def _sentiment_panel(self, frames: PriceFrames) -> pd.DataFrame | None:
        score_earnings_calls(self._context)                  # lazy, iterative, cache-incremental
        per_call = sentiment_kpis_streamed(self._context)     # per-ticker stream, bounded memory
        if per_call is None or per_call.empty:
            return None
        return build_earnings_call_feature_panel(
            None, frames.peers, frames.trading_index, embeddings=None, per_call=per_call)

    def _embedding_panel(self, frames: PriceFrames) -> pd.DataFrame | None:
        embed_earnings_calls(self._context)                  # lazy, no-op without an API key
        ekpi, asof = embedding_kpis_streamed(self._context)   # per-ticker stream, bounded memory
        if ekpi is None or ekpi.empty:
            return None
        return build_earnings_call_embedding_panel(
            None, frames.peers, frames.trading_index, sections=asof, ekpi=ekpi)
