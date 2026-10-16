"""Read-only bounded PostgreSQL snapshot replayed through the candidate into SQLite."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv

from src.context import Context
from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.utils.common.parts import CUBE_PARTS
from src.data_store.schema import Tables
from src.modelling.utils.panel import load_frame
from src.utils.config import read_config
from src.utils.universe import load_universe_tickers
from src.validate.checks.grain import check_grain
from src.validate.result import jsonable
from tests.conftest import _store


def _record(root: Path, result: dict[str, Any]) -> None:
    out = root / "reports/validate/2026-10-16-cube-assembly-selection/_out/real_replay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")


def _metadata(store: Any) -> dict[str, Any]:
    return {part.name: {"columns": store.columns(part.table), "max_date": str(store.max_date(part.table))} for part in CUBE_PARTS}


def test_real_source_assembly_replay(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path)
    root = Path(env_path).parent if env_path else Path.cwd()
    config = read_config("configs")
    source = _store()
    optional = set(config.build_cube.assembly.optional_parts)
    missing = [
        part.name for part in CUBE_PARTS if part.name not in optional and (not source.exists(part.table) or source.max_date(part.table) is None)
    ]
    if missing:
        _record(root, {"status": "abstain", "reason": "missing compulsory sources", "parts": missing, "metadata": _metadata(source)})
        pytest.skip(f"real candidate replay incomplete: compulsory parts absent/empty: {missing}")
    live_context = SimpleNamespace(store=source, config=config)
    universe = set(load_universe_tickers(cast(Context, live_context)))
    tickers = [ticker for ticker in ("AAPL", "MSFT", "JPM") if ticker in universe]
    assert len(tickers) == 3, "named source-backed sample must be available"
    horizon = max(config.build_cube.targets.horizons)
    sessions = source.distinct(Tables.cube_part_prices, "date", where={"ticker": tickers}, order="desc", limit=horizon + 21)
    assert sessions
    since, until = min(sessions), max(sessions)
    snapshot: dict[str, pd.DataFrame] = {}
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    for _attempt in range(2):
        before = _metadata(source)
        snapshot = {}
        for part in CUBE_PARTS:
            columns = before[part.name]["columns"]
            if not columns:
                continue
            chunks = list(source.iter_load(part.table, columns=columns, where={"ticker": tickers}, since=since, until=until))
            snapshot[part.name] = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
        after = _metadata(source)
        if before == after:
            break
    else:
        _record(root, {"status": "abstain", "reason": "source metadata changed during both bounded captures", "before": before, "after": after})
        pytest.skip("real candidate replay incomplete: concurrent source rebuild")
    peers_path = (
        Path(os.getenv("ROOT_PATH", config.local.paths.root)) / config.local.paths.get("data_store", "data_store") / "output/sector_peers.json"
    )
    if not peers_path.exists():
        _record(root, {"status": "abstain", "reason": "existing peer artifact absent; recomputation blocked", "metadata": after})
        pytest.skip("existing peer artifact unavailable; no production peer recomputation allowed")
    peers = json.loads(peers_path.read_text(encoding="utf-8"))
    monkeypatch.setattr("src.data_aggregate.transformers.step_assemble_cube.load_peers_or_raise", lambda *_: peers)
    reference = source.load(Tables.sp500_tickers, columns=["ticker", "sector", "industry_group"], where={"ticker": tickers})
    assert reference is not None
    sqlite_store.replace(Tables.sp500_tickers, reference)
    for part in CUBE_PARTS:
        frame = snapshot.get(part.name)
        if frame is not None and not frame.empty:
            sqlite_store.replace(part.table, frame)
    sqlite_store.replace(Tables.prices, snapshot[Tables.cube_part_prices.name][["date", "ticker"]])
    context = cast(Context, SimpleNamespace(store=sqlite_store, config=config, log=logging.getLogger(__name__)))
    StepAssembleCube(context, config).run()
    cube = sqlite_store.load(Tables.cube, columns=sqlite_store.columns(Tables.cube))
    assert cube is not None
    keys = ["date", "ticker"]
    actual = cube.assign(date=pd.to_datetime(cube["date"])).set_index(keys).sort_index()
    prices = snapshot[Tables.cube_part_prices.name].assign(date=lambda d: pd.to_datetime(d["date"])).set_index(keys).sort_index()
    assert actual.index.equals(prices.index)
    checked = 0
    for part in CUBE_PARTS:
        frame = snapshot.get(part.name)
        if frame is None or frame.empty:
            continue
        expected = frame.assign(date=pd.to_datetime(frame["date"])).set_index(keys).reindex(actual.index)
        for column in expected:
            values = expected[column].to_numpy()
            if pd.api.types.is_float_dtype(expected[column]) and part.kind != "prices":
                values = values.astype("float32").astype("float64")
            pd.testing.assert_series_equal(actual[column].isna(), expected[column].isna(), check_names=False)
            if pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(actual[column].to_numpy(float), values.astype(float), rtol=0, atol=0, equal_nan=True)
            else:
                pd.testing.assert_series_equal(actual[column], expected[column], check_dtype=False, check_names=False)
            checked += len(expected)
    latest = load_frame(
        sqlite_store, Tables.cube, columns=["date", "ticker", *[c for c in prices.columns if c != "date"]], since=until, downcast=False
    )
    assert latest is not None and len(latest) == len(prices.loc[pd.Timestamp(until)])
    wide_columns = set(sqlite_store.columns(Tables.cube))
    selected: dict[str, list[str]] = {}
    for table in (Tables.cube_part_fundamentals, Tables.cube_part_governance, Tables.cube_part_institutionals):
        payload = [col for col in sqlite_store.columns(table) if col not in keys]
        if payload:
            selected[table.name] = payload[:1]
    config.build_cube.assembly.columns = selected
    StepAssembleCube(context, config).run()
    narrowed = sqlite_store.load(Tables.cube, columns=sqlite_store.columns(Tables.cube))
    assert narrowed is not None
    removed = {col for name, keep in selected.items() for col in snapshot[name].columns if col not in [*keys, *keep]}
    assert not removed & set(narrowed.columns)
    assert len(narrowed) == len(cube)
    common = list(narrowed.columns)
    pd.testing.assert_frame_equal(narrowed[common].sort_values(keys).reset_index(drop=True), cube[common].sort_values(keys).reset_index(drop=True))
    grain = check_grain(context, Tables.cube)
    assert grain.status == "pass", grain
    _record(
        root,
        {
            "status": "pass",
            "scope": "bounded real-source candidate replay, not production adoption",
            "tickers": tickers,
            "since": since,
            "until": until,
            "rows": len(cube),
            "source_cells_checked": checked,
            "wide_columns": len(wide_columns),
            "narrow_columns": len(narrowed.columns),
            "metadata": after,
            "grain": asdict(grain),
        },
    )
    print(
        f"SANITY: {tickers}, {len(cube)} real price rows; {checked} selected source cells/nulls exact; schema {len(wide_columns)} -> {len(narrowed.columns)}; captured candidate grain PASS."
    )
