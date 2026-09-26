"""Canonical quarterly/live insider source selection."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.utils.institutionals import frontiers as institutional_frontiers
from src.data_store.schema import Tables


def test_bulk_replaces_a_whole_accession_while_live_only_accessions_fill_the_gap():
    bulk = pd.DataFrame(
        [
            {"accession_number": "A", "filing_date": "2026-05-01", "transaction_code": "S", "price_per_share": 91.67},
            {"accession_number": "A", "filing_date": "2026-05-01", "transaction_code": "M", "price_per_share": 20.00},
        ]
    )
    live = pd.DataFrame(
        [
            {"accession_number": "A", "filing_date": "2026-05-01", "transaction_code": "S", "price_per_share": 91.6725},
            {"accession_number": "A", "filing_date": "2026-05-01", "transaction_code": "M", "price_per_share": 20.00},
            {"accession_number": "B", "filing_date": "2026-07-02", "transaction_code": "P", "price_per_share": 10.25},
        ]
    )
    canonical = StepCubeInstitutionals._overlay_insider_sources(
        bulk,
        live,
        bulk_authoritative_through="2026Q2",
    )
    assert canonical is not None
    assert canonical.groupby("accession_number").size().to_dict() == {"A": 2, "B": 1}
    assert (
        canonical.loc[
            canonical["accession_number"].eq("A") & canonical["transaction_code"].eq("S"),
            "price_per_share",
        ].iloc[0]
        == 91.67
    )
    print(
        "SANITY: bulk accession A replaced both provisional live rows as one unit, while "
        "live-only accession B remained; no filing contains mixed source rows."
    )


def test_unpromoted_quarter_keeps_live_accessions_until_parity_passes():
    bulk = pd.DataFrame([{"accession_number": "A", "filing_date": "2026-08-01", "price_per_share": 91.67}])
    live = pd.DataFrame([{"accession_number": "A", "filing_date": "2026-08-01", "price_per_share": 91.6725}])

    canonical = StepCubeInstitutionals._overlay_insider_sources(
        bulk,
        live,
        bulk_authoritative_through="2026Q2",
    )
    assert canonical is not None
    assert canonical["price_per_share"].tolist() == [91.6725]
    capped = StepCubeInstitutionals._insider_bulk_complete_through(
        "2026Q3",
        bulk,
        live,
        bulk_authoritative_through="2026Q2",
    )
    assert capped == pd.Timestamp("2026-06-30")
    print(
        "SANITY: an overlapping Q3 ZIP cannot replace Q3 EDGAR before promotion; the bulk "
        "frontier remains capped at Q2 until the parity report passes."
    )


def test_complete_through_uses_the_later_valid_source_frontier():
    insider = pd.DataFrame({"filing_date": [pd.Timestamp("2026-07-15")]})
    got = StepCubeInstitutionals._insider_complete_through(pd.Timestamp("2026-06-30"), insider, pd.Timestamp("2026-09-22"))
    assert got == pd.Timestamp("2026-09-22")
    bulk_only = StepCubeInstitutionals._insider_complete_through(pd.Timestamp("2026-06-30"), insider)
    assert bulk_only == pd.Timestamp("2026-06-30")
    print(
        "SANITY: Q2 bulk alone is complete through 2026-06-30; a successful all-ticker " "daily scan advances the canonical frontier to 2026-09-22."
    )


def test_live_frontier_requires_coverage_for_every_universe_ticker():
    calls = []

    class _Store:
        coverage = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "complete_through": ["2026-09-24", "2026-09-22"],
            }
        )

        def load(self, table, *, columns, where, optional):
            calls.append((table, columns, where, optional))
            return self.coverage.copy()

    warnings = []
    step = object.__new__(StepCubeInstitutionals)
    step._store = _Store()
    step._log = SimpleNamespace(warning=lambda message, *args: warnings.append(message % args))

    got = institutional_frontiers.insider_live_complete_through(
        step._store,
        step._log,
        ["BBB", "AAA"],
    )
    assert got == pd.Timestamp("2026-09-22")
    assert calls == [
        (
            Tables.insider_transactions_live_coverage,
            ("ticker", "complete_through"),
            {"ticker": ["AAA", "BBB"]},
            True,
        )
    ]

    step._store.coverage = step._store.coverage.loc[lambda frame: frame["ticker"].eq("AAA")]
    assert institutional_frontiers.insider_live_complete_through(
        step._store,
        step._log,
        ["AAA", "BBB"],
    ) is None
    assert "missing 1/2 universe ticker(s)" in warnings[-1]
    print(
        "SANITY: the live insider frontier is the minimum all-ticker scan date and becomes "
        "unavailable when any requested ticker lacks coverage."
    )
