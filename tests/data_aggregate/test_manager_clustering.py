"""Manager clustering: the pure labelling functions and `StepManagerProfile.run_clustering` / the CLI split (AC-003, AC-004 synthetic).

The pure functions are tested on hand-built frames. The step runs on an in-memory SQLite `sec_13f_manager_details` seeded with four planted
manager groups (small-cap US, large-cap US, mid-cap blend, ETF book) over six past report quarters plus one quarter that is not due yet, plus
ineligible funds and the category managers, so a fit, the nightly prediction, a new CIK and a full relabel are checked on known groups.
No DAG, cube build, PostgreSQL or configs/ write is made: the centroid JSON lives in a temporary config directory.
"""

from __future__ import annotations

import json
import logging
import shutil
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from omegaconf import DictConfig, OmegaConf

from src.data_aggregate import cli as aggregate_cli
from src.data_aggregate.transformers.step_manager_profile import StepManagerProfile
from src.data_aggregate.utils.manager_profile.clustering import (
    assemble_labels,
    due_rows,
    eligible_fund_rows,
    keep_clusters,
    read_mapping,
    write_centroids,
)
from src.data_store.schema import Tables
from src.modelling.transformers.cluster import ClusterTransformer
from src.utils.config import read_config

ROOT = Path(__file__).resolve().parents[2]
FILTERS = {"category": "fund", "min_positions": 4, "min_book_usd": 20_000_000}
PERIODS = [pd.Timestamp(p) for p in ("2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30")]
RECENT = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)  # filed, but its label is not due for another 60 days
CATEGORIES = {"bank": 7, "insurance": 8, "broker_dealer": 9, "pension": 10, "sovereign": 14, "other": 15}
TRACKED = {11: "pct_etf", 12: "pct_etf", 13: "pct_equity_foreign"}  # tracked fund id -> the exclude key that drops its fitted cluster
EXCLUDE = {"pct_etf": 0.40, "pct_equity_foreign": 0.40}
STATS = ["pct_cap_small", "pct_cap_mid", "pct_cap_large", "pct_equity_us", "pct_equity_foreign", "pct_etf", "pct_bond", "active_share"]
# group: rows per quarter, the eight composition stats, then book value, hhi and turnover (the three ranked features)
GROUPS: dict[str, tuple[int, list[float], float, float, float]] = {
    "small_us": (60, [0.70, 0.20, 0.10, 0.90, 0.02, 0.03, 0.01, 0.80], 3e8, 0.10, 0.60),
    "large_us": (40, [0.05, 0.15, 0.80, 0.85, 0.05, 0.05, 0.02, 0.15], 5e10, 0.04, 0.10),
    "mid_blend": (25, [0.20, 0.60, 0.20, 0.50, 0.30, 0.10, 0.05, 0.45], 4e9, 0.06, 0.30),
    "etf_book": (20, [0.10, 0.30, 0.60, 0.20, 0.10, 0.85, 0.05, 0.30], 1e9, 0.02, 0.45),
}
EXPECTED_NUMBER = {
    "small_us": 0,
    "large_us": 1,
    "mid_blend": 2,
    "etf_book": 11,
}  # kept ones numbered by descending fit size; the ETF group is dropped, tracked as 11


SMALL_WORLD_LABELS = [  # the registry of the planted world: k=4 fitted, 3 kept (0-2), one dropped (tracked 11)
    {"id": -2, "name": "not_computable", "features": False},
    {"id": -1, "name": "below_size_floor", "features": False},
    *({"id": number, "name": f"fund_{number}", "features": True} for number in range(3)),
    *({"id": cluster_id, "name": kind, "features": True} for kind, cluster_id in CATEGORIES.items() if cluster_id < 14),
    {"id": 11, "name": "fund_etf_heavy", "features": False, "dropped_by": "pct_etf"},
    {"id": 14, "name": "sovereign", "features": False, "category": "sovereign"},
    {"id": 15, "name": "other", "features": False, "category": "other"},
]


def _centroid_transformer(raw: np.ndarray, sizes: list[int], features: list[str]) -> ClusterTransformer:
    """A transformer whose clusters sit at the raw-unit centroids `raw` (standardisation mean 0.3, std 0.2), nothing fitted."""
    block = {"features": features, "group_by": "period", "k": len(sizes), "seed": 0, "n_init": 1, "max_rows_per_group": 10}
    transformer = ClusterTransformer(OmegaConf.create(block))
    mean, std = np.full(len(features), 0.3), np.full(len(features), 0.2)
    transformer.centroids = ((raw - mean) / std).tolist()
    transformer.mean, transformer.std = mean.tolist(), std.tolist()
    transformer.sizes = sizes
    transformer.version = "unit-test"
    return transformer


def _ten_clusters() -> tuple[ClusterTransformer, dict[int, int]]:
    """Ten clusters of which idx 2 (ETF 0.45), idx 7 (ETF 0.50) and idx 9 (foreign 0.60) are heavy; the seven others are kept."""
    etf = np.array([0.10, 0.10, 0.45, 0.10, 0.10, 0.10, 0.10, 0.50, 0.10, 0.10])
    foreign = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.00, 0.60])
    small = np.linspace(0.1, 0.9, 10)
    sizes = [50, 10, 5, 30, 40, 20, 60, 99, 70, 98]
    transformer = _centroid_transformer(np.column_stack([etf, foreign, small]), sizes, ["pct_etf", "pct_equity_foreign", "pct_cap_small"])
    return transformer, {8: 0, 6: 1, 0: 2, 4: 3, 3: 4, 5: 5, 1: 6, 7: 11, 2: 12, 9: 13}


def test_keep_clusters_keeps_exactly_the_wanted_number_numbered_by_descending_size_else_raises() -> None:
    transformer, expected = _ten_clusters()

    mapping = keep_clusters(transformer, EXCLUDE, 7, TRACKED)

    assert mapping == expected, "7 kept 0-6 by descending fit size (70, 60, 50, 40, 30, 20, 10); the 3 dropped take 11-13 by their centroid"
    with pytest.raises(ValueError, match="7"):
        keep_clusters(transformer, {"pct_etf": 0.48, "pct_equity_foreign": 0.40}, 7, TRACKED)  # idx 2 survives: 8 remain
    with pytest.raises(ValueError, match="7"):
        keep_clusters(transformer, {"pct_etf": 0.05}, 7, {11: "pct_etf"})  # every cluster is dropped: 0 remain
    print("\n=== SANITY CHECK: keep_clusters ===")
    print(f"10 fitted -> 7 kept and 3 tracked {mapping}; the rule leaving 8 or 0 clusters raises instead of guessing. Validated.")


def test_tracked_ids_follow_the_rule_value_never_the_fit_size() -> None:
    transformer, _ = _ten_clusters()
    first = keep_clusters(transformer, EXCLUDE, 7, TRACKED)
    transformer.sizes = [50, 10, 99, 30, 40, 20, 60, 5, 70, 98]  # a refit that swaps the sizes of the two ETF clusters (idx 2 and 7)
    second = keep_clusters(transformer, EXCLUDE, 7, TRACKED)

    assert {index: second[index] for index in (7, 2, 9)} == {7: 11, 2: 12, 9: 13} == {index: first[index] for index in (7, 2, 9)}
    assert sorted(second.values()) == [*range(7), 11, 12, 13]
    both = _centroid_transformer(
        np.column_stack(
            [
                [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.50, 0.10, 0.45],  # idx 7 and 9 are above the ETF limit
                [0.10, 0.10, 0.55, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.60],  # idx 2 and 9 are above the foreign limit
                np.linspace(0.1, 0.9, 10),
            ]
        ),
        [50, 10, 5, 30, 40, 20, 60, 99, 70, 98],
        ["pct_etf", "pct_equity_foreign", "pct_cap_small"],
    )
    overlap = keep_clusters(both, EXCLUDE, 7, TRACKED)
    assert (overlap[7], overlap[9], overlap[2]) == (11, 12, 13), (
        "idx 9 is over both limits and goes by the first rule (pct_etf); idx 2 is the foreign one"
    )
    with pytest.raises(ValueError, match="tracked"):
        keep_clusters(transformer, EXCLUDE, 7, {11: "pct_etf", 12: "pct_equity_foreign", 13: "pct_equity_foreign"})  # the fit drops 2 ETF + 1 foreign
    print("\n=== SANITY CHECK: tracked ids ===")
    print(
        f"dropped clusters take 11, 12 (pct_etf 0.50, 0.45) and 13 (foreign 0.60) whatever the fit sizes {second}; a registry that disagrees with the fit raises. Validated."
    )


def test_the_centroid_file_carries_ten_centroids_the_kept_flags_the_numbers_and_the_tracked_ids(tmp_path: Path) -> None:
    transformer, mapping = _ten_clusters()
    path = tmp_path / "centroids.json"

    write_centroids(transformer, mapping, 7, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["centroids"]) == 10 and sum(payload["kept"]) == 7 and payload["kept"][8] and not payload["kept"][7]
    assert payload["number"] == [2, 6, None, 4, 3, 5, 1, None, 0, None]
    assert payload["tracked"] == [None, None, 12, None, None, None, None, 11, None, 13]
    assert read_mapping(path) == mapping, "the saved ids of all ten fitted clusters come back"
    assert {"mean", "std", "features", "version", "sizes"} <= set(payload), "the transformer's own keys are kept"
    path.write_text(json.dumps({k: v for k, v in payload.items() if k not in ("kept", "number", "tracked")}), encoding="utf-8")
    with pytest.raises(ValueError, match="fit"):
        read_mapping(path)
    print("\n=== SANITY CHECK: centroid file ===")
    print(
        "10 centroids, 7 kept flags, numbers [2, 6, -, 4, 3, 5, 1, -, 0, -] and tracked ids [-, -, 12, -, -, -, -, 11, -, 13] round-trip; a file without them asks for a refit. Validated."
    )


def _rows(**columns: Any) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_due_rows_are_unlabelled_rows_whose_period_plus_the_delay_has_passed() -> None:
    today = pd.Timestamp("2026-06-29")
    details = _rows(
        cik=list("abcdef"),
        period=pd.to_datetime(["2026-04-01", "2026-03-31", "2026-03-30", "2026-06-29", "2026-03-30", "2025-12-31"]),
        category=["fund", "fund", "fund", "fund", None, "bank"],
        cluster=[np.nan, np.nan, np.nan, np.nan, np.nan, 7.0],
    )

    nightly = due_rows(details, 90, today, relabel=False)
    everything = due_rows(details, 90, today, relabel=True)

    assert nightly["cik"].tolist() == ["b", "c"], "2026-03-31 + 90 days is the run date (due); 2026-04-01 is not; a labelled row is never due"
    assert everything["cik"].tolist() == ["b", "c", "f"], (
        "a relabel takes the labelled row too; a row with no profile yet (category null) is never due"
    )
    print("\n=== SANITY CHECK: due_rows ===")
    print(f"nightly due {nightly['cik'].tolist()}, relabel due {everything['cik'].tolist()}; not-yet-old and unprofiled rows wait. Validated.")


def test_eligible_fund_rows_need_the_category_the_positions_and_the_book() -> None:
    details = _rows(
        category=["fund", "fund", "fund", "bank", "fund", "fund", "fund"],
        n_positions=[4, 3, 50, 50, np.nan, 50, 9],
        book_value_usd=[2e7, 5e9, 1.9e7, 5e9, 5e9, np.nan, 3e7],
    )

    eligible = eligible_fund_rows(details, FILTERS)

    assert eligible.index.tolist() == [0, 6], (
        "the boundary row (4 positions, 20M) and a 9-position fund pass; 3 positions, a smaller book, a bank or a null fail"
    )
    print("\n=== SANITY CHECK: eligible_fund_rows ===")
    print(
        "a fund with 4 positions and a 20M book is eligible (the floor is 4), so is one with 9; 3 positions, a smaller book, null positions or book fail. Validated."
    )


def test_assemble_labels_gives_every_row_an_id_a_fund_cluster_a_tracked_cluster_a_category_the_floor_or_not_computable() -> None:
    due = _rows(
        cik=[f"c{n}" for n in range(16)],
        period=pd.Timestamp("2023-03-31"),
        category=["fund"] * 6 + ["bank", "insurance", "broker_dealer", "pension", "bank", "sovereign", "other", "sovereign", "other", "fund"],
        n_positions=[50, 50, 3, 50, 50, 4, 50, 50, 50, 50, 3, 50, 50, 50, 3, np.nan],
        book_value_usd=[1e9, 1e9, 1e9, 1e6, 1e9, 2e7, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e6, 1e9, 1e9],
    )
    eligible = [0, 1, 4, 5]  # the fund rows at or above the floor, the only ones the transformer labels
    raw = pd.Series(pd.array([1, 3, pd.NA, 2], dtype="Int64"), index=due.index[eligible])  # fitted 1 -> 0, 2 -> 1, 3 was dropped and tracked as 11
    mapping = {1: 0, 2: 1, 3: 11}

    labels = assemble_labels(due, raw, mapping, CATEGORIES, FILTERS)

    assert labels.columns.tolist() == ["cik", "period", "cluster"] and labels["cik"].tolist() == due["cik"].tolist()
    assert labels["cluster"].tolist() == [0, 11, -1, -1, -2, 1, 7, 8, 9, 10, -1, 14, 15, -1, -1, -1]
    assert labels["cluster"].notna().all(), "every due row is assigned an id"
    assert set(labels["cluster"]) <= {-2, -1, 0, 1, 7, 8, 9, 10, 11, 14, 15}
    with pytest.raises(ValueError, match="mystery"):
        assemble_labels(due.assign(category="mystery"), raw, mapping, CATEGORIES, FILTERS)
    print("\n=== SANITY CHECK: assemble_labels ===")
    print(
        "fund -> 0 / 1 (kept), 11 (dropped, tracked), -2 (null required feature), -1 (3 positions, 1M book, null positions); "
        f"bank, insurance, broker_dealer, pension -> 7-10, sovereign, other -> 14, 15, each -1 below the floor; labels {labels['cluster'].tolist()}"
    )


def _planted_details() -> tuple[pd.DataFrame, dict[str, str]]:
    """Manager rows of the four planted groups, ineligible funds and category managers over every period; returns (rows, group of each cik)."""
    rng = np.random.default_rng(3)
    rows: list[dict[str, Any]] = []
    group_of: dict[str, str] = {}

    def add(cik: str, group: str, category: str, n_positions: int, stats: list[float], book: float, hhi: float, turnover: float) -> None:
        group_of[cik] = group
        for period in [*PERIODS, RECENT]:
            noisy = np.clip(np.array(stats) + rng.normal(0, 0.01, len(stats)), 0, 1)
            row = dict(zip(STATS, noisy, strict=True))
            rows.append(
                {"cik": cik, "period": period, "category": category, "n_positions": n_positions, **row}
                | {"book_value_usd": book * rng.uniform(0.9, 1.1), "hhi": hhi * rng.uniform(0.9, 1.1), "turnover": turnover * rng.uniform(0.9, 1.1)}
            )

    number = 0
    for group, (count, stats, book, hhi, turnover) in GROUPS.items():
        for _ in range(count):
            add(f"{number:010d}", group, "fund", 80, stats, book, hhi, turnover)
            number += 1
    small = GROUPS["small_us"]
    for group, n_positions, book in (("few_positions", 3, 3e8), ("small_book", 80, 1e6), ("four_positions", 4, 3e8)):
        for _ in range(4):
            add(f"{number:010d}", group, "fund", n_positions, small[1], book, small[3], small[4])
            number += 1
    for _ in range(4):  # a first row: turnover is null, everything else is the small-cap US book
        add(f"{number:010d}", "first_row", "fund", 80, small[1], small[2], small[3], np.nan)
        number += 1
    for _ in range(4):  # a required feature (the small-cap share) is missing
        add(f"{number:010d}", "missing_cap", "fund", 80, [np.nan, *small[1][1:]], small[2], small[3], small[4])
        number += 1
    for category, count in (("bank", 5), ("insurance", 5), ("broker_dealer", 3), ("pension", 3), ("sovereign", 2), ("other", 2)):
        for _ in range(count):
            add(f"{number:010d}", category, category, 50, small[1], 1e9, small[3], small[4])
            number += 1
    return pd.DataFrame(rows), group_of


class Workspace:
    """The seeded store, a temporary config directory for the centroid JSON, and the steps built on them."""

    def __init__(self, store: Any, config_dir: Path, cfg: DictConfig, group_of: dict[str, str]) -> None:
        self.store, self.config_dir, self.cfg, self.group_of = store, config_dir, cfg, group_of
        self.calls: list[tuple[str, str, Any]] = []
        load, save = store.load, store.save

        def _load(table: Any, *args: Any, **kwargs: Any) -> Any:
            self.calls.append(("load", getattr(table, "name", table), kwargs.get("columns")))
            return load(table, *args, **kwargs)

        def _save(table: Any, df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
            self.calls.append(("save", getattr(table, "name", table), list(df.columns)))
            return save(table, df, *args, **kwargs)

        store.load, store.save = _load, _save

    @property
    def path(self) -> Path:
        return self.config_dir / str(self.cfg.cluster.centroids_path)

    def step(self, **cluster: Any) -> StepManagerProfile:
        cfg = OmegaConf.merge(self.cfg, {"cluster": cluster})
        context = types.SimpleNamespace(store=self.store, config_dir=self.config_dir)
        return StepManagerProfile(context=context, config=cfg)  # type: ignore[arg-type]

    def table(self) -> pd.DataFrame:
        df = self.store.load(Tables.sec_13f_manager_details)
        df["period"] = pd.to_datetime(df["period"])
        df["group"] = df["cik"].map(self.group_of)
        return df.sort_values(["cik", "period"], ignore_index=True)

    def writes(self) -> list[list[str]]:
        return [columns for kind, name, columns in self.calls if kind == "save" and name == Tables.sec_13f_manager_details.name]


@pytest.fixture
def workspace(sqlite_store: Any, tmp_path: Path) -> Workspace:
    """Planted managers in `sec_13f_manager_details`; k=4 fitted, 3 kept; configs live in a temporary directory."""
    details, group_of = _planted_details()
    sqlite_store.create_declared(Tables.sec_13f_manager_details)
    sqlite_store.save(Tables.sec_13f_manager_details, details)
    (tmp_path / "superinvestors").mkdir()
    shutil.copy(ROOT / "configs" / "superinvestors" / "manager_categories.json", tmp_path / "superinvestors" / "manager_categories.json")
    cfg = read_config(str(ROOT / "configs"))
    cfg = OmegaConf.merge(cfg, {"cluster": {"k": 4, "keep": 3, "n_init": 5, "fit": False, "labels": SMALL_WORLD_LABELS}})
    return Workspace(sqlite_store, tmp_path, cfg, group_of)  # type: ignore[arg-type]


def _due(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["period"].isin(PERIODS)]


def test_a_fit_writes_the_centroids_labels_every_due_row_and_warns_that_fit_is_still_true(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)

    written = workspace.step(fit=True).run_clustering()

    table, due = workspace.table(), _due(workspace.table())
    payload = json.loads(workspace.path.read_text(encoding="utf-8"))
    assert len(payload["centroids"]) == 4 and sum(payload["kept"]) == 3 and sorted(n for n in payload["number"] if n is not None) == [0, 1, 2]
    assert ClusterTransformer.load(workspace.path).version == payload["version"], "the file still loads (its hash covers only the fit)"
    assert written == len(due) and due["cluster"].notna().all(), "every due row is labelled"
    assert table.loc[~table["period"].isin(PERIODS), "cluster"].isna().all(), "the not-yet-due quarter stays null"
    by_group = due.groupby("group")["cluster"].agg(["min", "max"])
    expected = {**EXPECTED_NUMBER, "few_positions": -1, "small_book": -1, "four_positions": 0, "first_row": 0, "missing_cap": -2, **CATEGORIES}
    for group, number in expected.items():
        assert list(by_group.loc[group]) == [number, number], group
    assert payload["tracked"].count(11) == 1 and sum(value is not None for value in payload["tracked"]) == 1, "one dropped cluster, tracked as 11"
    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("still true" in m for m in messages) and any("build-sec-thirteen -F" in m for m in messages)
    print("\n=== SANITY CHECK: fit and first labelling ===")
    print(
        f"4 fitted, 3 kept {payload['number']}, 1 tracked {payload['tracked']}; {written} due rows labelled: groups {by_group['min'].astype(int).to_dict()}; "
        "the 'fit still true' and -F warnings are logged."
    )


def test_a_missing_turnover_falls_back_to_the_nearest_cluster_and_a_label_depends_on_its_own_quarter_alone(workspace: Workspace) -> None:
    workspace.step(fit=True).run_clustering()
    before = _due(workspace.table())
    first_row, missing_cap = before[before["group"] == "first_row"], before[before["group"] == "missing_cap"]
    assert first_row["turnover"].isna().all() and (first_row["cluster"] == 0).all(), (
        "no turnover: nearest cluster on the other 9 features, the small-cap US one"
    )
    assert missing_cap["pct_cap_small"].isna().all() and (missing_cap["cluster"] == -2).all(), "a required feature missing: not computable"

    # Point in time: every other quarter changes (turnover, size and the small-cap share), the target quarter's labels must not.
    target = PERIODS[2]
    others = [period.strftime("%Y-%m-%d") for period in PERIODS if period != target]
    workspace.store.update(Tables.sec_13f_manager_details, {"turnover": None, "pct_cap_small": 0.55, "book_value_usd": 5e8}, {"period": others})
    workspace.step(fit=False).run_clustering(full=True)

    after = _due(workspace.table())
    same_quarter = before["period"] == target
    pd.testing.assert_series_equal(after.loc[same_quarter].set_index("cik")["cluster"], before.loc[same_quarter].set_index("cik")["cluster"])
    moved = int(after.loc[~same_quarter, "cluster"].ne(before.loc[~same_quarter, "cluster"]).sum())
    assert moved > 0, "control: the edited quarters do relabel, so an unchanged target quarter is evidence"
    print("\n=== SANITY CHECK: optional turnover and point in time ===")
    print(
        f"{len(first_row)} first-row funds (turnover null) -> cluster 0, {len(missing_cap)} funds missing the small-cap share -> -2; "
        f"after editing the other 5 quarters the {int(same_quarter.sum())} labels of {target.date()} are unchanged while {moved} edited-quarter labels moved."
    )


def test_fit_false_fits_nothing_writes_no_json_and_labels_only_due_unlabelled_rows(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.step(fit=True).run_clustering()
    expected = _due(workspace.table()).set_index(["cik", "period"])["cluster"]
    everyone = workspace.table()["cik"].unique().tolist()
    workspace.store.update(Tables.sec_13f_manager_details, {"cluster": None}, {"cik": everyone})  # the nightly state: no label yet
    first = workspace.table()
    workspace.store.update(
        Tables.sec_13f_manager_details, {"cluster": 5}, {"cik": [first.loc[0, "cik"]], "period": [PERIODS[0].strftime("%Y-%m-%d")]}
    )
    before = workspace.path.read_bytes()
    monkeypatch.setattr(ClusterTransformer, "fit", lambda *a, **k: pytest.fail("fit: false must not fit"))
    workspace.calls.clear()

    written = workspace.step(fit=False).run_clustering()

    calls, writes = list(workspace.calls), workspace.writes()
    table = workspace.table()
    labels = _due(table).set_index(["cik", "period"])["cluster"]
    kept = table[(table["cik"] == first.loc[0, "cik"]) & (table["period"] == PERIODS[0])]["cluster"].iloc[0]
    assert workspace.path.read_bytes() == before, "the centroid JSON is only read"
    assert kept == 5 and written == len(labels) - 1, "a labelled row is never touched"
    pd.testing.assert_series_equal(labels.drop((first.loc[0, "cik"], PERIODS[0])), expected.drop((first.loc[0, "cik"], PERIODS[0])))
    assert [name for kind, name, _ in calls if kind == "load"] == [Tables.sec_13f_manager_details.name], "one table read, nothing else"
    assert writes == [["cik", "period", "cluster"]], "one upsert of the key and the label"
    print("\n=== SANITY CHECK: fit false ===")
    print(
        f"fit never called, JSON unchanged, {written} due null rows relabelled to the fit's labels, a labelled row kept; {len(calls)} store calls (1 read, 1 upsert)."
    )


def test_nothing_due_reads_no_centroids_writes_nothing_and_exits_cleanly(workspace: Workspace) -> None:
    workspace.step(fit=True).run_clustering()
    workspace.path.unlink()
    workspace.calls.clear()

    written = workspace.step(fit=False).run_clustering()

    assert written == 0 and workspace.writes() == [], "no due row: no centroid read (the file is gone), no write"
    print("\n=== SANITY CHECK: no-op night ===")
    print("a second night with every due row labelled returns 0 without the centroid file or any write. Validated.")


def test_a_new_cik_in_a_due_quarter_gets_the_label_a_full_relabel_gives(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.step(fit=True).run_clustering()
    donor = workspace.table().query("group == 'small_us'").iloc[0]
    newcomer = pd.DataFrame([{**donor.drop(["group", "cluster"]).to_dict(), "cik": "9999999999", "period": PERIODS[2]}])
    workspace.store.save(Tables.sec_13f_manager_details, newcomer)
    workspace.calls.clear()

    written = workspace.step(fit=False).run_clustering()

    table = workspace.table()
    label = table[table["cik"] == "9999999999"]["cluster"].iloc[0]
    monkeypatch.setattr(ClusterTransformer, "fit", lambda *a, **k: pytest.fail("a full relabel never refits"))
    workspace.step(fit=False).run_clustering(full=True)
    assert written == 1 and label == 0, "only the new manager is labelled, as the small-cap US group (cluster 0)"
    assert workspace.table().set_index(["cik", "period"])["cluster"].loc[("9999999999", PERIODS[2])] == label
    print("\n=== SANITY CHECK: new CIK ===")
    print(
        f"the unlabelled newcomer got cluster {int(label)} on its own; a full relabel gives the same label (ranks taken among the quarter's funds)."
    )


def test_a_rule_that_does_not_leave_the_wanted_clusters_raises_and_leaves_the_file_and_the_table_unchanged(workspace: Workspace) -> None:
    workspace.path.write_text('{"previous": "fit"}', encoding="utf-8")

    with pytest.raises(ValueError, match="3"):
        workspace.step(fit=True, exclude={"pct_etf": 5.0}).run_clustering()  # nothing is dropped: 4 clusters remain, 3 wanted

    assert workspace.path.read_text(encoding="utf-8") == '{"previous": "fit"}', "the saved centroids are untouched"
    assert workspace.writes() == [] and workspace.table()["cluster"].isna().all(), "no label is written"
    print("\n=== SANITY CHECK: keep rule fails ===")
    print("a rule leaving 4 clusters instead of 3 raised; the old JSON and the all-null label column are unchanged. Validated.")


def test_full_relabels_every_due_row_with_the_saved_centroids_and_never_refits(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.step(fit=True).run_clustering()
    expected = workspace.table().set_index(["cik", "period"])["cluster"]
    small_ciks = workspace.table().query("group == 'small_us'")["cik"].unique()
    smalls, banker = small_ciks[:3].tolist(), small_ciks[3]
    workspace.store.update(Tables.sec_13f_manager_details, {"cluster": 9}, {"cik": smalls})  # wrong labels, also on the not-due quarter
    workspace.store.update(Tables.sec_13f_manager_details, {"category": "bank"}, {"cik": [banker]})  # a category edit changes the row's label
    monkeypatch.setattr(ClusterTransformer, "fit", lambda *a, **k: pytest.fail("a full relabel never refits"))
    before = workspace.path.read_bytes()

    written = workspace.step(fit=False).run_clustering(full=True)

    after = workspace.table().set_index(["cik", "period"])["cluster"]
    due_index = _due(workspace.table()).set_index(["cik", "period"]).index
    assert workspace.path.read_bytes() == before and written == len(due_index)
    changed = after.loc[due_index].ne(expected.loc[due_index])
    assert set(changed[changed].index.get_level_values("cik")) == {banker}, "the wrong labels are repaired; only the re-categorised manager differs"
    assert (after.loc[(banker, PERIODS[0])]) == 7 and after.loc[[(c, RECENT) for c in smalls]].eq(9).all(), "not-due rows are not touched"
    print("\n=== SANITY CHECK: full relabel ===")
    print(f"{written} due rows relabelled from the saved JSON, 3 corrupted managers repaired, the re-categorised fund became cluster 7, no refit.")


def test_run_is_the_profile_then_the_labels_and_the_cli_calls_one_sub_run_each(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(StepManagerProfile, "run_profile", lambda self, full=False: order.append(f"profile:{full}") or 2)
    monkeypatch.setattr(StepManagerProfile, "run_clustering", lambda self, full=False: order.append(f"cluster:{full}") or 3)

    total = workspace.step().run()

    assert order == ["profile:False", "cluster:False"] and total == 5
    order.clear()
    fake = types.SimpleNamespace(
        run_profile=lambda full=False: order.append(f"cli-profile:{full}"), run_clustering=lambda full=False: order.append(f"cli-cluster:{full}")
    )
    monkeypatch.setattr(aggregate_cli, "_step", lambda cls, config_path: fake)
    runner = CliRunner()
    for args in (["build-manager-profile"], ["build-manager-profile", "-F"], ["build-cluster"], ["build-cluster", "-F"]):
        assert runner.invoke(aggregate_cli.cli, args).exit_code == 0, args
    assert order == ["cli-profile:False", "cli-profile:True", "cli-cluster:False", "cli-cluster:True"]
    print("\n=== SANITY CHECK: run and CLI ===")
    print(f"run() = profile then clustering; CLI build-manager-profile -> run_profile, build-cluster (-c, -F) -> run_clustering: {order}")
