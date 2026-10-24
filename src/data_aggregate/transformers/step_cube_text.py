"""Prepare earnings-call text, persist independent model outputs, then refresh text features."""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from typing import cast

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import EARNINGS_CALL_FEATURES
from src.context import Context
from src.data_aggregate.utils.common.incremental import PartWindow, write_part
from src.data_aggregate.utils.common.price_frames import load_price_frames, load_trading_calendar
from src.data_aggregate.utils.text.earnings_call_embeddings import (
    build_embedding_kpis,
    build_embedding_rows,
    completed_embedding_calls,
    feature_embedding_chunks,
)
from src.data_aggregate.utils.text.earnings_call_features import (
    SENTIMENT_COLUMNS,
    attach_issuer_identity,
    build_earnings_call_feature_panel,
    canonical_sentiment_rows,
    completed_sentiment_calls,
    feature_history_scope,
    load_issuer_identity,
    sentiment_kpis,
    valid_sentiment_rows,
)
from src.data_aggregate.utils.text.earnings_call_preparation import SENTIMENT_TAGS, TextPlan, sentiment_rows
from src.data_aggregate.utils.text.nlp_sentiment import get_sentiment_engine
from src.data_store.schema import Tables
from src.gpt_extract import embed_texts, openai_api_key
from src.utils.step import Step
from src.utils.text.earnings_call_split import CALL_SECTIONS, split_call
from src.utils.text.text_metrics import assess_earnings_call_sections, clean_earnings_call_paragraphs, word_count


class StepCubeText(Step):
    _FIELDS = ("close_split",)

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)

        # Keep configured model and window settings on the step.
        self._store = context.store
        settings = config.build_cube.earnings_calls
        embedding = config.gpt.embedding
        self._sentiment_model = settings.sentiment_model
        self._embedding_model = embedding.model
        self._lag = settings.availability_lag_days
        self._refresh_days = settings.refresh_calendar_days
        self._embedding_batch = embedding.batch_size
        self._embedding_max_chars = embedding.max_chars

        # Reject invalid settings before reading data or starting inference.
        for name, value in (("sentiment_model", self._sentiment_model), ("embedding.model", self._embedding_model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty model name")
        for name, value, minimum in (
            ("availability_lag_days", self._lag, 0),
            ("refresh_calendar_days", self._refresh_days, 1),
            ("embedding.batch_size", self._embedding_batch, 1),
            ("embedding.max_chars", self._embedding_max_chars, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be a {'non-negative' if minimum == 0 else 'positive'} integer")
        cleaning_version = "speaker-clean-v3"
        self._sentiment_cache = f"{self._sentiment_model}:{cleaning_version}"
        self._embedding_cache = f"{self._embedding_model}:{cleaning_version}"

    def run(self, full: bool = False) -> None:
        # Plan once and share preparation between the independent model stages.
        calendar = load_trading_calendar(self._store)
        plan = self.plan_window(calendar, full=full)
        prepared = self.prepare_sections(plan)
        self.run_sentiments(prepared, plan)
        self.run_embeddings(prepared, plan)
        del prepared
        # Build features from the saved outputs after both models finish.
        self.run_features(calendar, plan)

    def plan_window(self, calendar: pd.DatetimeIndex, *, full: bool) -> TextPlan:
        """Compare all source keys with each model table; plan the separate cube tail."""
        # Load compact source/model metadata once.
        source = self._store.load(Tables.earnings_call_sections, columns=["ticker", "quarter", "as_of"], where={"paragraph": 1}, optional=True)
        sentiment = self._store.load(Tables.earnings_call_sentiment, columns=SENTIMENT_COLUMNS, optional=True)
        embedding = self._store.load(Tables.earning_calls_embedding, columns=["ticker", "quarter", "seq", "as_of", "model"], optional=True)
        # Resolve the output window, including schema-triggered full rebuilding.
        columns = self._store.columns(Tables.cube_part_text)
        last = None if full or not columns else self._store.max_date(Tables.cube_part_text)
        expected = {"date", "ticker", *[f"f_{name}" for name in EARNINGS_CALL_FEATURES]}
        if columns and set(columns) != expected:
            self._log.info("Text feature columns changed -> full feature rebuild, retaining valid model outputs.")
            last = None
        cutoff = None if last is None else pd.Timestamp(last).normalize() - pd.Timedelta(days=self._refresh_days)
        window = PartWindow(None if last is None else pd.Timestamp(last), cutoff, cutoff)
        # Validate cached scores and match each saved model to the source call.
        source = source if source is not None else pd.DataFrame(columns=["ticker", "quarter", "as_of"])
        source["as_of"] = pd.to_datetime(source["as_of"], errors="coerce")
        sentiment = sentiment if sentiment is not None else pd.DataFrame(columns=SENTIMENT_COLUMNS)
        valid = valid_sentiment_rows(sentiment, self._sentiment_cache)
        if not valid.empty:
            valid = valid.merge(source, on=["ticker", "quarter"], suffixes=("", "_source"))
            valid = valid[valid["as_of"].eq(valid["as_of_source"])].drop(columns="as_of_source")
        embedding = embedding if embedding is not None else pd.DataFrame(columns=["ticker", "quarter", "seq", "as_of", "model"])
        source_keys = set(map(tuple, source[["ticker", "quarter"]].to_numpy()))
        scored = completed_sentiment_calls(valid)
        embedded = completed_embedding_calls(embedding, self._embedding_cache, source)
        plan = TextPlan(
            window=window,
            source=source,
            sentiment=valid,
            embedding_keys=embedding,
            sentiment_work=source_keys - scored,
            embedding_work=source_keys - embedded,
            eligible=scored.copy(),
        )
        # Report inference work separately from the feature rewrite range.
        self._log.info(
            "Text plan: %d source calls; sentiment %d to do/%d reused; embedding %d to do/%d reused; feature rewrite %s..%s.",
            len(source),
            len(plan.sentiment_work),
            len(scored),
            len(plan.embedding_work),
            len(embedded),
            cutoff.date() if cutoff is not None else "FULL",
            calendar[-1].date(),
        )
        if cutoff is not None:
            pending = plan.sentiment_work | plan.embedding_work
            old = source[pd.MultiIndex.from_frame(source[["ticker", "quarter"]]).isin(pending) & source["as_of"].lt(cutoff)]
            if not old.empty:
                self._log.info("%d older NLP calls need work; rebuilding historical cube rows outside the overlap requires build-text -F.", len(old))
        return plan

    def prepare_sections(self, plan: TextPlan) -> pd.DataFrame:
        """Read selected paragraphs once, then visibly split, clean and assess every call."""

        # Load the union of calls needed by either model.
        requested = plan.sentiment_work | plan.embedding_work
        paragraph_columns = ["ticker", "quarter", "paragraph", "as_of", "speaker", "content"]
        columns = [
            "ticker",
            "quarter",
            "as_of",
            "coverage",
            *SENTIMENT_TAGS,
            *[f"{tag}_words" for tag in SENTIMENT_TAGS],
            "turns",
        ]
        paragraphs = (
            self._store.load(
                Tables.earnings_call_sections,
                columns=paragraph_columns,
                where={"ticker": sorted({key[0] for key in requested}), "quarter": sorted({key[1] for key in requested})},
                order_by=["ticker", "quarter", "paragraph"],
                optional=True,
            )
            if requested
            else None
        )
        paragraphs = pd.DataFrame(columns=paragraph_columns) if paragraphs is None else paragraphs
        paragraphs = paragraphs[pd.MultiIndex.from_frame(paragraphs[["ticker", "quarter"]]).isin(requested)]

        # Clean and assess each complete call before assembling model documents.
        rows: list[dict[str, object]] = []
        reasons: Counter[str] = Counter()
        cleaning: Counter[str] = Counter()
        excluded_words = 0
        turn_count = 0
        plan.eligible.difference_update(requested)
        for (ticker, quarter), call in paragraphs.groupby(["ticker", "quarter"], sort=False):
            dates = pd.to_datetime(call["as_of"], errors="coerce")
            if dates.isna().any() or dates.nunique() != 1:
                reasons["invalid call date"] += 1
                continue
            records = cast(list[dict[str, object]], call[["paragraph", "speaker", "content"]].to_dict("records"))
            records, removed = clean_earnings_call_paragraphs(records)
            cleaning.update(removed)
            split = split_call(records)
            quality = assess_earnings_call_sections(dict(zip(CALL_SECTIONS, (split.prepared_remarks, split.qa), strict=True)))
            if not quality.valid:
                reasons[quality.reason or "unusable split"] += 1
                excluded_words += quality.combined_word_count
                continue
            row: dict[str, object] = {"ticker": ticker, "quarter": quarter, "as_of": dates.iloc[0], "turns": split.turns}
            documents = (quality.cleaned_sections[CALL_SECTIONS[0]], split.questions, quality.cleaned_sections[CALL_SECTIONS[1]])
            for tag, text in zip(SENTIMENT_TAGS, documents, strict=True):
                row[tag] = text
                row[f"{tag}_words"] = word_count(text)
            row["coverage"] = int(bool(documents[0])) + 2 * int(bool(documents[2]))
            rows.append(row)
            turn_count += len(split.turns)
            plan.eligible.add((str(ticker), str(quarter)))

        # Return shared preparation and summarize exclusions/cleaning.
        prepared = pd.DataFrame(rows, columns=columns)
        self._log.info(
            "Text preparation: %d paragraphs, %d accepted calls (%d partial), %d turns; %d excluded calls/%d cleaned words; reasons=%s; cleaning=%s.",
            len(paragraphs),
            len(prepared),
            int(prepared["coverage"].ne(3).sum()),
            turn_count,
            sum(reasons.values()),
            excluded_words,
            dict(reasons),
            dict(cleaning),
        )
        return prepared

    def run_sentiments(self, prepared: pd.DataFrame, plan: TextPlan) -> None:
        """Score missing role documents, reuse valid partners and save three canonical roles."""

        # Select pending calls and index their reusable canonical scores.
        selected = prepared[pd.MultiIndex.from_frame(prepared[["ticker", "quarter"]]).isin(plan.sentiment_work)]
        reusable = canonical_sentiment_rows(plan.sentiment)
        previous_calls = {key: group for key, group in reusable.groupby(["ticker", "quarter"], sort=False)}
        empty = reusable.iloc[0:0]
        legacy_keys = set(map(tuple, plan.sentiment.loc[plan.sentiment["tag"].eq("qa"), ["ticker", "quarter"]].to_numpy()))
        existing = {(str(row.ticker), str(row.quarter), str(row.tag)) for row in reusable.itertuples(index=False)}
        needs_model = any(
            row[tag] and (str(row.ticker), str(row.quarter), tag) not in existing for _, row in selected.iterrows() for tag in SENTIMENT_TAGS
        )

        # Load one model only when there is nonempty text left to score.
        engine = get_sentiment_engine(self._sentiment_model) if needs_model else None
        if needs_model and engine is None:
            self._log.warning("Sentiment engine unavailable; missing scores stay retryable and supported embedding features remain available.")
        completed = inferred = unavailable = 0
        saved: list[pd.DataFrame] = []

        # Validate and save each call independently, then update the in-memory snapshot.
        for _, call in selected.iterrows():
            key = (str(call.ticker), str(call.quarter))
            previous = previous_calls.get(key, empty)
            tags = [tag for tag in SENTIMENT_TAGS if call[tag] and (str(call.ticker), str(call.quarter), tag) not in existing]
            if not tags:
                probabilities = []
            elif engine is None:
                continue
            else:
                probabilities = engine.score_texts([str(call[tag]) for tag in tags])
            scored = sentiment_rows(call, previous, probabilities, self._sentiment_cache)
            self._store.save(Tables.earnings_call_sentiment, scored)
            saved.append(scored)
            if key in legacy_keys:
                self._store.delete(Tables.earnings_call_sentiment, {"ticker": str(call.ticker), "quarter": str(call.quarter), "tag": "qa"})
            completed += 1
            inferred += len(tags)
            unavailable += int(scored["n_words"].eq(0).sum())
        frames = [frame for frame in [reusable, *saved] if not frame.empty]
        plan.sentiment = pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "quarter", "tag"], keep="last") if frames else reusable
        self._log.info(
            "Sentiment: %d/%d calls saved, %d sections inferred; %d unavailable sections recorded.",
            completed,
            len(selected),
            inferred,
            unavailable,
        )

    def run_embeddings(self, prepared: pd.DataFrame, plan: TextPlan) -> None:
        """Embed eligible stored turns and save/reconcile one complete call at a time."""
        # Keep pending calls with actual turns; empty calls never reach the API.
        selected = prepared[pd.MultiIndex.from_frame(prepared[["ticker", "quarter"]]).isin(plan.embedding_work)]
        empty_views = int((~selected["turns"].map(bool)).sum())
        if empty_views:
            self._log.info("Embedding: %d eligible calls have no mapped turns; unsupported embedding KPIs stay null.", empty_views)
        selected = selected[selected["turns"].map(bool)]
        if selected.empty:
            self._log.info("Embedding: no eligible missing turns.")
            return
        if not openai_api_key():
            self._log.warning("Embedding API unavailable; missing vectors stay retryable and supported sentiment features remain available.")
            return

        # Validate and persist each complete call, removing any stale saved tail.
        completed = turns = removed = 0
        for _, call in selected.iterrows():
            vectors = embed_texts(
                [turn["text"] for turn in call.turns],
                model=self._embedding_model,
                batch_size=self._embedding_batch,
                max_chars=self._embedding_max_chars,
            )
            rows = build_embedding_rows(call, vectors, self._embedding_cache)
            stale = plan.embedding_keys[
                plan.embedding_keys["ticker"].eq(call.ticker)
                & plan.embedding_keys["quarter"].eq(call.quarter)
                & plan.embedding_keys["seq"].ge(len(rows))
            ]
            if not stale.empty:
                removed += self._store.delete(
                    Tables.earning_calls_embedding, {"ticker": str(call.ticker), "quarter": str(call.quarter), "seq": stale["seq"].tolist()}
                )
            self._store.save(Tables.earning_calls_embedding, rows)
            plan.embedding_work.discard((str(call.ticker), str(call.quarter)))
            completed += 1
            turns += len(rows)
        self._log.info("Embedding: %d calls/%d turns saved, %d stale turns removed.", completed, turns, removed)

    def run_features(self, calendar: pd.DatetimeIndex, plan: TextPlan) -> None:
        """Read only required saved history and tail prices, calculate KPIs and persist output."""

        # Resolve active issuer history before loading feature inputs.
        roster, lineage, tenure = load_issuer_identity(self._context)
        source = plan.source[pd.MultiIndex.from_frame(plan.source[["ticker", "quarter"]]).isin(plan.eligible)]
        identified = attach_issuer_identity(source, roster, lineage, tenure)
        scope = feature_history_scope(identified, calendar, plan.window.since, self._lag)
        where = {"ticker": sorted(scope["ticker"].unique()), "quarter": sorted(scope["quarter"].unique())}
        history_from = None if scope.empty else pd.Timestamp(scope["as_of"].min())
        history_until = None if scope.empty else pd.Timestamp(scope["as_of"].max())

        # Reuse sentiment, materialize prices once, and open one embedding stream.
        sentiment = plan.sentiment
        frames = load_price_frames(self._store, peers={}, fields=self._FIELDS, since=plan.window.since) if not scope.empty else None
        embeddings = self._store.iter_load(
            Tables.earning_calls_embedding,
            chunksize=5_000,
            columns=["ticker", "quarter", "seq", "as_of", "section", "tag", "exchange_idx", "embedding", "model"],
            where={**where, "model": self._embedding_cache},
            since=history_from,
            until=history_until,
            date_col="as_of",
            order_by=["ticker", "quarter", "seq"],
        )

        # Join matching sentiment dates and reduce complete embedding calls to KPIs.
        per_call = scope.copy()
        if not sentiment.empty:
            sentiment = sentiment.merge(scope[["ticker", "quarter", "as_of"]], on=["ticker", "quarter"], suffixes=("", "_source"))
            sentiment = sentiment[sentiment["as_of"].eq(sentiment["as_of_source"])].drop(columns="as_of_source")
            metrics = sentiment_kpis(sentiment).drop(columns="as_of")
            per_call = per_call.merge(metrics, on=["ticker", "quarter"], how="left")

        with closing(embeddings):
            vectors = feature_embedding_chunks(embeddings, scope, plan.embedding_work)
            per_call = per_call.merge(build_embedding_kpis(vectors), on=["ticker", "quarter"], how="left")

        # Project the ten call KPIs onto the output trading sessions.
        panel = build_earnings_call_feature_panel(
            per_call, calendar, availability=None if frames is None else frames.availability, availability_lag_days=self._lag, since=plan.window.since
        )

        # Save once, or clear stale output when no eligible features remain.
        if not panel.empty:
            write_part(self._store, Tables.cube_part_text, panel, plan.window, drop_empty=True)
        elif self._store.exists(Tables.cube_part_text):
            if plan.window.is_full:
                self._store.delete(Tables.cube_part_text, {"date": self._store.NOT_NULL})
            else:
                assert plan.window.refresh_from is not None
                self._store.append_tail(Tables.cube_part_text, panel, plan.window.refresh_from, inclusive=True)
        self._log.info(
            "Text features: history %s..%s, %d calls; output %s..%s, %d rows.",
            history_from,
            history_until,
            len(scope),
            plan.window.since if plan.window.since is not None else "FULL",
            calendar[-1].date(),
            len(panel),
        )
