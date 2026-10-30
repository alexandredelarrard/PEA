"""Known-truth contracts for selected reads, price rows and physical output schema."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.context import Context
from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.utils.common.panel_merge import DuplicateKeyError, FeatureCollisionError
from src.data_aggregate.utils.common.parts import CUBE_PARTS
from src.data_store.errors import UnknownTableError
from src.data_store.schema import Tables, name_of
from src.modelling.utils.panel import load_frame
from src.utils.config import read_config


def _parts() -> dict[str, pd.DataFrame]:
    grid = pd.DataFrame({"date": pd.to_datetime(["2023-01-02", "2023-01-03"]), "ticker": ["AAA", "AAA"]})
    tables = {part.name: grid.assign(**{f"v_{i}": np.array([1.2, 2.3])}) for i, part in enumerate(CUBE_PARTS)}
    tables[Tables.cube_part_prices.name] = grid.assign(close_split=[10.123456789, 12.123456789])
    tables[Tables.cube_part_targets.name] = grid.iloc[:1].assign(target_zscore_h60=0.5)
    tables[Tables.sp500_tickers.name] = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"], "industry_group": ["Software"]})
    tables[Tables.prices.name] = grid.copy()
    return tables


_OPTIONAL = (Tables.cube_part_text, Tables.cube_part_institutionals)
_ASSEMBLED = tuple(p for p in CUBE_PARTS if p.table != Tables.cube_part_betas)


def _config(**assembly: Any) -> Any:
    policy = {
        "optional_parts": [t.name for t in _OPTIONAL],
        "compulsory_parts": [p.name for p in _ASSEMBLED if p.table not in _OPTIONAL],
        **assembly,
    }
    return OmegaConf.create({"build_cube": {"assembly": policy}, "data_extract": {"redundant_ticks": []}})


def _step(store: Any, monkeypatch: pytest.MonkeyPatch, config: Any = None) -> StepAssembleCube:
    config = config if config is not None else _config()
    context = SimpleNamespace(store=store, config=config, log=logging.getLogger(__name__), config_dir="configs")
    step = StepAssembleCube(context=context, config=config)
    step._context = cast(Context, context)
    step._config = config
    step._cfg = config.build_cube
    step._log = context.log
    monkeypatch.setattr("src.data_aggregate.transformers.step_assemble_cube.load_peers_or_raise", lambda *_: {"AAA": {"BBB": 1.0}})
    monkeypatch.setattr("src.data_aggregate.transformers.step_assemble_cube._CHUNK_ROWS", 1)
    return step


class _SpyStore:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self.tables = {k: v.copy() for k, v in tables.items()}
        self.writes: list[str] = []
        self.reads: list[tuple[str, Any, dict[str, Any]]] = []

    def exists(self, table: Any) -> bool:
        return name_of(table) in self.tables

    def columns(self, table: Any) -> list[str]:
        return list(self.tables[name_of(table)].columns) if self.exists(table) else []

    def load(self, table: Any, columns: Any = None, **kwargs: Any) -> pd.DataFrame | None:
        self.reads.append((name_of(table), columns, kwargs))
        frame = self.tables.get(name_of(table))
        if frame is None:
            return None
        out = frame.copy()
        for key, values in (kwargs.get("where") or {}).items():
            out = out[out[key].isin(values if isinstance(values, list) else [values])]
        for key, operator in (("since", "ge"), ("until", "le")):
            if kwargs.get(key) is not None:
                out = out[getattr(pd.to_datetime(out["date"]), operator)(pd.Timestamp(kwargs[key]))]
        out = out[list(columns)] if columns is not None else out
        out = out.head(kwargs.get("limit", len(out)))
        return None if out.empty else out

    def iter_load(self, table: Any, **kwargs: Any) -> Iterator[pd.DataFrame]:
        size = kwargs.pop("chunksize", 200_000)
        frame = self.load(table, **kwargs)
        if frame is not None:
            for i in range(0, len(frame), size):
                yield frame.iloc[i : i + size].copy()

    def max_date(self, table: Any) -> pd.Timestamp | None:
        frame = self.tables.get(name_of(table))
        return (
            pd.to_datetime(frame["date"], errors="coerce", format="mixed").max()
            if frame is not None and not frame.empty and "date" in frame
            else None
        )

    def max_date_by(self, table: Any, key_col: str) -> dict[str, pd.Timestamp]:
        return cast(dict[str, pd.Timestamp], self.tables[name_of(table)].groupby(key_col)["date"].max().to_dict()) if self.exists(table) else {}

    def row_count(self, table: Any) -> int:
        return len(self.tables[name_of(table)]) if self.exists(table) else 0

    def drop(self, table: Any) -> None:
        self.writes.append("drop")
        del self.tables[name_of(table)]

    def replace(self, table: Any, frame: pd.DataFrame) -> int:
        self.writes.append("replace")
        self.tables[name_of(table)] = frame.copy()
        return len(frame)


def test_price_grid_values_precision_and_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _parts()
    tables[Tables.cube_part_fundamentals.name] = pd.concat(
        [
            tables[Tables.cube_part_fundamentals.name].iloc[:1],
            pd.DataFrame({"date": [pd.Timestamp("2023-01-02")], "ticker": ["ORPHAN"], "v_3": [99.0]}),
        ]
    )
    store = _SpyStore(tables)
    _step(store, monkeypatch).run()
    cube = store.tables[Tables.cube.name]
    assert store.writes == ["replace"]
    assert cube[["date", "ticker"]].equals(tables[Tables.cube_part_prices.name][["date", "ticker"]])
    assert pd.isna(cube.iloc[-1]["target_zscore_h60"]) and pd.isna(cube.iloc[-1]["v_3"])
    np.testing.assert_array_equal(cube["close_split"], tables[Tables.cube_part_prices.name]["close_split"])
    assert all(cube[col].dtype == np.float32 for col in cube if str(col).startswith(("v_", "target_")))
    assert cube["peers"].eq('{"BBB": 1.0}').all()
    assert {"sector", "industry_group"}.issubset(cube)
    print("SANITY: exact price grid, sparse cells and null latest labels survive; precision preserved.")


@pytest.mark.parametrize("selection", [None, [], ["v_3"], ["date", "ticker", "v_3"]])
def test_projected_reads(monkeypatch: pytest.MonkeyPatch, selection: Any) -> None:
    tables = _parts()
    tables[Tables.cube_part_fundamentals.name]["extra"] = 9.0
    store = _SpyStore(tables)
    _step(store, monkeypatch, _config(columns={Tables.cube_part_fundamentals.name: selection})).run()
    reads = [cols for name, cols, kw in store.reads if name == Tables.cube_part_fundamentals.name and not kw.get("limit")]
    expected = {"date", "ticker", "v_3"} | ({"extra"} if not selection else set())
    assert reads and all(cols is not None and set(cols) in (expected, {"date", "ticker"}) for cols in reads)
    assert any(set(cols) == expected for cols in reads)
    assert ("extra" in store.tables[Tables.cube.name]) == (not selection)
    print(f"SANITY: selection {selection!r} projects payload before reading; only key scans may precede it.")


@pytest.mark.parametrize("table", [Tables.cube_part_prices, Tables.cube_part_targets, Tables.cube_part_fundamentals])
def test_duplicate_keys_are_refused(monkeypatch: pytest.MonkeyPatch, table: Any) -> None:
    tables = _parts()
    tables[table.name] = pd.concat([tables[table.name], tables[table.name].iloc[:1]], ignore_index=True)
    store = _SpyStore(tables)
    with pytest.raises(DuplicateKeyError, match="duplicate|repeats"):
        _step(store, monkeypatch).run()
    assert not store.writes
    print(f"SANITY: {table.name} duplicate spans one-row read chunks and is refused before writes.")


def test_keys_only_selections_preserve_all_price_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _SpyStore(_parts())
    cfg = _config(columns={part.name: ["date", "ticker"] for part in _ASSEMBLED if part.kind != "prices"})
    _step(store, monkeypatch, cfg).run()
    assert len(store.tables[Tables.cube.name]) == 2
    assert set(store.tables[Tables.cube.name]) == {"date", "ticker", "close_split", "peers", "sector", "industry_group"}
    print("SANITY: keys-only projections cannot remove rows from the price root.")


def test_bad_input_preserves_physical_cube(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, frame in _parts().items():
        sqlite_store.replace(name, frame)
    _step(sqlite_store, monkeypatch).run()
    before = sqlite_store.load(Tables.cube, columns=sqlite_store.columns(Tables.cube))
    cfg = _config(columns={Tables.cube_part_text.name: ["missing"]})
    with pytest.raises(ValueError, match="missing"):
        _step(sqlite_store, monkeypatch, cfg).run()
    after = sqlite_store.load(Tables.cube, columns=sqlite_store.columns(Tables.cube))
    pd.testing.assert_frame_equal(before, after)
    print("SANITY: preflight failure preserves the real destination's existing cells and physical schema.")


@pytest.mark.parametrize("part", [p for p in _ASSEMBLED if p.table not in _OPTIONAL and p.kind != "prices"])
@pytest.mark.parametrize("empty", [False, True])
def test_required_inputs_preserve_existing_cube(monkeypatch: pytest.MonkeyPatch, part: Any, empty: bool) -> None:
    tables = _parts()
    tables[Tables.cube.name] = pd.DataFrame({"old": [1]})
    if empty:
        tables[part.name] = tables[part.name].iloc[:0]
    else:
        del tables[part.name]
    store = _SpyStore(tables)
    with pytest.raises(RuntimeError, match=f"{part.name}.*{part.command}"):
        _step(store, monkeypatch).run()
    assert not store.writes and store.tables[Tables.cube.name].equals(tables[Tables.cube.name])
    print(f"SANITY: required {part.name}, empty={empty}, fails without touching existing output.")


@pytest.mark.parametrize("table", [Tables.cube_part_text, Tables.cube_part_institutionals])
@pytest.mark.parametrize("empty", [False, True])
def test_optional_absent_or_empty_skips_before_selection(monkeypatch: pytest.MonkeyPatch, table: Any, empty: bool) -> None:
    tables = _parts()
    if empty:
        tables[table.name] = tables[table.name].iloc[:0]
    else:
        del tables[table.name]
    store = _SpyStore(tables)
    _step(store, monkeypatch, _config(columns={table.name: ["not_yet_built"]})).run()
    assert "not_yet_built" not in store.tables[Tables.cube.name]
    print(f"SANITY: optional {table}, empty={empty}, skips before requested-column validation.")


def test_unknown_part_name(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _SpyStore(_parts())
    original_cube = pd.DataFrame({"preserved": [1.0]})
    store.tables[Tables.cube.name] = original_cube.copy()
    config = _config()
    config.build_cube.assembly.optional_parts.append("cube_part_typo")
    with pytest.raises(UnknownTableError):
        _step(store, monkeypatch, config).run()
    assert not store.writes
    pd.testing.assert_frame_equal(store.tables[Tables.cube.name], original_cube)
    print("SANITY: a mistyped part name fails loudly, makes zero writes and preserves existing output.")


@pytest.mark.parametrize("defect", ["duplicate", "bad_date", "collision", "metadata", "missing_column", "read_error"])
def test_broken_present_inputs(monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    tables = _parts()
    part = Tables.cube_part_text
    frame = tables[part.name]
    if defect == "duplicate":
        tables[part.name] = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif defect == "bad_date":
        frame["date"] = frame["date"].astype(object)
        frame.loc[0, "date"] = "bad"
    elif defect == "collision":
        frame["target_zscore_h60"] = 7.0
    elif defect == "metadata":
        frame["peers"] = "bad"
    cfg = _config(columns={part.name: ["missing"]}) if defect == "missing_column" else _config()
    tables[Tables.cube.name] = pd.DataFrame({"preserved": [1.0]})
    store = _SpyStore(tables)
    if defect == "read_error":
        original = store.iter_load

        def failing(table: Any, **kwargs: Any) -> Iterator[pd.DataFrame]:
            if name_of(table) == part.name:
                raise RuntimeError(f"{part.name} read failed")
            yield from original(table, **kwargs)

        monkeypatch.setattr(store, "iter_load", failing)
    with pytest.raises((ValueError, RuntimeError, FeatureCollisionError, DuplicateKeyError)):
        _step(store, monkeypatch, cfg).run()
    assert not store.writes
    pd.testing.assert_frame_equal(store.tables[Tables.cube.name], tables[Tables.cube.name])
    print(f"SANITY: populated {defect} fails before replacement.")


def test_physical_schema_narrows(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _parts()
    tables[Tables.cube_part_fundamentals.name]["extra"] = 9.0
    for name, frame in tables.items():
        sqlite_store.replace(name, frame)
    _step(sqlite_store, monkeypatch).run()
    assert {"extra", "v_5"}.issubset(sqlite_store.columns(Tables.cube))
    sqlite_store.drop(Tables.cube_part_text)
    _step(sqlite_store, monkeypatch, _config(columns={Tables.cube_part_fundamentals.name: ["v_3"]})).run()
    assert not {"extra", "v_5"} & set(sqlite_store.columns(Tables.cube))
    assert sqlite_store.row_count(Tables.cube) == 2
    print("SANITY: actual SQLite output schema removes deselected and skipped optional columns; price rows remain.")


def test_committed_config() -> None:
    assembly = read_config("configs").build_cube.assembly
    listed = [*assembly.compulsory_parts, *assembly.optional_parts]
    assert set(assembly.optional_parts) == {Tables.cube_part_text.name, Tables.cube_part_institutionals.name, Tables.cube_part_sec_thirteen.name}
    assert len(listed) == len(set(listed))
    missing = {p.name for p in _ASSEMBLED} - set(listed)
    assert not missing, f"registered parts silently excluded from the cube: {sorted(missing)}"
    assert all(value is None or len(value) == 0 for value in assembly.columns.values())
    print("SANITY: every registered part except betas is listed once in the config; optional parts and keep-all defaults match.")


def test_optional_no_overlap_and_selected_out_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _parts()
    tables[Tables.cube_part_text.name]["date"] = pd.to_datetime(["2020-01-01", "2020-01-02"])
    tables[Tables.cube_part_fundamentals.name]["close_split"] = 99.0
    store = _SpyStore(tables)
    _step(store, monkeypatch, _config(columns={Tables.cube_part_fundamentals.name: ["v_3"]})).run()
    assert store.tables[Tables.cube.name]["v_5"].isna().all()
    assert store.tables[Tables.cube.name]["v_5"].dtype == np.float32
    assert store.tables[Tables.cube.name]["close_split"].max() < 13.0
    print("SANITY: populated out-of-grid optional payload contributes null columns; selected-out collisions are harmless.")


def test_assembled_cube_feeds_the_modelling_reader(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, frame in _parts().items():
        sqlite_store.replace(name, frame)
    _step(sqlite_store, monkeypatch).run()
    raw = load_frame(sqlite_store, Tables.cube, columns=["date", "ticker", "v_3", "target_zscore_h60"])
    assert raw is not None and raw["target_zscore_h60"].notna().sum() == 1
    latest = load_frame(sqlite_store, Tables.cube, columns=["date", "ticker", "v_3"], since="2023-01-03", downcast=False)
    assert latest is not None and len(latest) == 1 and latest["v_3"].notna().all()
    assert "target_zscore_h60" not in latest
    print("SANITY: the modelling reader keeps the one labelled row; the feature-only reader retains the newest unlabelled price row.")


def test_existing_cube_rewrites_only_the_overlap_window(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, frame in _parts().items():
        sqlite_store.replace(name, frame)
    _step(sqlite_store, monkeypatch).run()

    # a new session, a restated value inside the window (last stored date) and one before it
    grid = pd.DataFrame({"date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]), "ticker": "AAA"})
    for name in (p.name for p in _ASSEMBLED):
        old = sqlite_store.load(name)
        assert old is not None
        value_cols = [c for c in old.columns if c not in ("date", "ticker")]
        new = grid.assign(**{c: np.float64(1.2) for c in value_cols})
        if name == Tables.cube_part_fundamentals.name:
            new["v_3"] = [7.0, 5.0, 6.0]
        sqlite_store.replace(name, new)
    _step(sqlite_store, monkeypatch).run()

    cube = sqlite_store.load(Tables.cube, columns=["date", "ticker", "v_3"])
    assert cube is not None
    cube = cube.assign(date=pd.to_datetime(cube["date"])).sort_values("date")
    assert len(cube) == 3
    np.testing.assert_allclose(cube["v_3"].to_numpy(float), [1.2, 5.0, 6.0], rtol=1e-6)
    print("SANITY: rows before the overlap start keep their stored value; the last stored date is restated and the new session appended.")
