"""
The say-on-pay proxy fallback in the governance cleaning stage (REQ-008, ledger P4).

The 8-K tally is the single say-on-pay source. A proxy whose FILED support has no 8-K
say-on-pay tally in the prior 450 days adds one synthetic vote row (for = support,
against = 1 - support, abstain 0), dated at the proxy and marked `source = proxy`.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.clean import GovernanceInputs, clean_governance_inputs
from src.data_aggregate.utils.governance.vote_dissent_features import _say_on_pay_history


def _proxies(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "as_of": pd.to_datetime([r[1] for r in rows]),
            "accession_number": [r[2] for r in rows],
            "say_on_pay_support_pct": [r[3] for r in rows],
        }
    )


def _votes(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    """8-K say-on-pay tallies; `filing_date` as `datetime.date`, as Postgres DATE returns it."""
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "accession_number": [r[2] for r in rows],
            "proposal_seq": [2.0] * len(rows),
            "filing_date": [dt.date.fromisoformat(r[1]) for r in rows],
            "proposal_type": ["say_on_pay"] * len(rows),
            "votes_for": [r[3] for r in rows],
            "votes_against": [r[4] for r in rows],
            "votes_abstain": [0.0] * len(rows),
        }
    )


def _clean(proxies: pd.DataFrame, votes: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, int]]:
    out, stats = clean_governance_inputs(GovernanceInputs(def14a=proxies, votes=votes), None)
    assert out.votes is not None
    return out.votes, stats


def test_fallback_row_when_no_8k() -> None:
    """Support 0.62 and no 8-K -> one proxy row, dissent 0.38, dated at the proxy."""
    proxies = _proxies([("AAA", "2019-04-01", "PX-AAA-2019", 0.62)])
    votes, stats = _clean(proxies, _votes([("BBB", "2019-01-10", "8K-BBB", 90.0, 10.0)]))
    added = votes.loc[votes["source"] == "proxy"]
    assert len(added) == 1
    row = added.iloc[0]
    assert row["ticker"] == "AAA" and row["accession_number"] == "PX-AAA-2019"
    assert pd.Timestamp(row["filing_date"]) == pd.Timestamp("2019-04-01")
    assert row["proposal_type"] == "say_on_pay"
    assert (votes.loc[votes["source"] != "proxy", "source"] == "8k").all()

    hist = _say_on_pay_history(votes, {})
    assert hist is not None
    got = hist.loc[hist["ticker"] == "AAA", "sop_dissent"]
    assert got.tolist() == pytest.approx([0.38])
    assert any(k.startswith("votes_fallback|") and v == 1 for k, v in stats.items())
    print(f"\n[sop fallback] AAA support 0.62, no 8-K -> 1 proxy row, dissent {got.iloc[0]:.2f}. Validated.")


def test_no_fallback_when_8k_within_window() -> None:
    """An 8-K tally 300 days before the proxy suppresses the fallback; one 500 days before does not."""
    proxies = _proxies([("AAA", "2019-04-01", "PX-AAA-2019", 0.95), ("CCC", "2019-04-01", "PX-CCC-2019", 0.80)])
    votes, _ = _clean(
        proxies,
        _votes([("AAA", "2018-06-05", "8K-AAA", 95.0, 5.0), ("CCC", "2017-11-17", "8K-CCC", 85.0, 15.0)]),
    )
    added = votes.loc[votes["source"] == "proxy"]
    assert added["ticker"].tolist() == ["CCC"]
    print("\n[sop fallback] 8-K 300 d before -> no row (AAA); 8-K 500 d before -> row (CCC). Validated.")


def test_fallback_never_reads_a_carried_value() -> None:
    """A proxy with no filed support gets no row, even where an earlier proxy filed one."""
    proxies = _proxies(
        [
            ("AAA", "2018-04-01", "PX-AAA-2018", 0.90),
            ("AAA", "2019-04-01", "PX-AAA-2019", np.nan),
            ("AAA", "2020-04-01", "PX-AAA-2020", 1.5),  # outside [0, 1]: refused
        ]
    )
    votes, stats = _clean(proxies, None)
    added = votes.loc[votes["source"] == "proxy"]
    assert added["accession_number"].tolist() == ["PX-AAA-2018"]
    assert added["votes_for"].tolist() == pytest.approx([0.90])
    assert added["votes_against"].tolist() == pytest.approx([0.10])
    assert any("REFUSED" in k for k in stats), stats
    print("\n[sop fallback] only the FILED 2018 value becomes a row; the 2019 gap and the 1.5 are not. Validated.")
