"""Cross-filing gender consensus over `def14a_directors`. Deterministic, no LLM.

The extraction resolves gender from in-document evidence and records the rule in `gender_basis` (`stated` >
`honorific` > `pronoun` > `name`). A director's gender does not change and directors recur across years and boards,
so the rows of one person are reconciled to their highest-provenance value: a director whose own bio carried no
evidence is filled, and a first-name inference is corrected wherever a sibling row has evidence. A value is never
downgraded to a weaker basis, so a second run is a no-op.
"""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables

# `person_key` lives in `src/utils/names.py` so the governance cube buckets the same people without importing this
# package; `votes/roles.py` and `votes/flatten.py` import it from here. Do not change it there: `sec_8k_votes`'
# stored role-category sums were bucketed with it.
from src.utils.names import person_key  # noqa: F401

logger = logging.getLogger(__name__)

#: Provenance ranking. `stated` is the proxy identifying the director itself (a diversity
#: matrix); `honorific` and `pronoun` are in-document evidence; `name` is a bare first-name
#: prior with no evidence behind it at all.
BASIS_RANK = {"stated": 3, "honorific": 2, "pronoun": 1, "name": 0}
#: Above this rank a value is DOCUMENT evidence rather than an inference -- the threshold
#: `pct_gender_stated` reports and the one a consumer can filter on.
EVIDENCE_RANK = BASIS_RANK["honorific"]
_DIRECTOR_COLUMNS = ["ticker", "accession_number", "name", "as_of", "gender", "gender_basis"]


def _norm(value: object) -> str | None:
    """Lower-cased stripped text, or None for anything that is not a non-empty string. A null read back from Postgres
    arrives as `float('nan')`, which is truthy, so `value or ""` is not enough."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def _basis_rank(value: object) -> int:
    """The `BASIS_RANK` of a `gender_basis` value; -1 for none or an unknown basis."""
    normalized = _norm(value)
    return BASIS_RANK.get(normalized, -1) if normalized is not None else -1


def _is_female(value: object) -> bool:
    """Whether a gender value reads female."""
    normalized = _norm(value)
    return normalized is not None and normalized.startswith("f")


def consensus(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One gender per person (`name`, `gender`, `gender_basis` needed), applied to every row of that person: the
    highest basis rank present wins outright, then the majority value within it, then the most recent row's value.
    Returns the copy with `gender` / `gender_basis` / `person_key` set and the stats."""
    # clean: the person key, basis rank and normalised gender of each row
    out = df.copy()
    for col in ("gender", "gender_basis"):
        if col not in out.columns:
            out[col] = None
    out["person_key"] = out["name"].map(person_key)
    out["_rank"] = out["gender_basis"].map(_basis_rank)
    out["_gender"] = out["gender"].map(_norm)

    # compute: each person's winning gender and basis
    resolved: dict[str, tuple[str, str]] = {}
    for key, grp in out[out["person_key"].notna()].groupby("person_key"):
        known = grp[grp["_gender"].notna()]  # notna() covers None AND pd.NA
        if known.empty:
            continue
        best_rank = known["_rank"].max()
        top = known[known["_rank"] == best_rank]
        counts = Counter(top["_gender"])
        winner, n_top = counts.most_common(1)[0]
        if len([g for g, c in counts.items() if c == n_top]) > 1:
            # a genuine tie at the same provenance -> the most recent row decides
            latest = top.sort_values("as_of").iloc[-1] if "as_of" in top.columns else top.iloc[-1]
            winner = latest["_gender"]
        basis = next((b for b, r in BASIS_RANK.items() if r == best_rank), "name")
        resolved[str(key)] = (str(winner), basis)

    # post-process: apply the winner to every row of its person, counting fills and overturns
    filled = overturned = unchanged = 0
    new_gender, new_basis = [], []
    for _, r in out.iterrows():
        key = r["person_key"]
        if not isinstance(key, str) or key not in resolved:
            new_gender.append(None if pd.isna(r["_gender"]) else r["_gender"])
            new_basis.append(_norm(r["gender_basis"]))
            continue
        winner, basis = resolved[key]
        # a pyarrow-backed string column turns `_norm`'s None into `pd.NA`, and `pd.NA is None` is False: compare
        # with `pd.isna`, never with `is None`
        old = r["_gender"]
        if pd.isna(old):
            filled += 1
        elif old != winner:
            overturned += 1
        else:
            unchanged += 1
        new_gender.append(winner)
        # never DOWNGRADE the recorded provenance: the row's own basis stands when it is already
        # at least as strong as the consensus's, which is what makes a second run a no-op
        own_rank = _basis_rank(r["gender_basis"])
        new_basis.append(_norm(r["gender_basis"]) if own_rank >= BASIS_RANK[basis] else basis)

    # post-process: the resolved columns and the stats
    out["gender"] = new_gender
    out["gender_basis"] = new_basis
    stats = {
        "rows": int(len(out)),
        "people": len(resolved),
        "filled": filled,
        "overturned": overturned,
        "unchanged": unchanged,
        "unmatched_rows": int(out["person_key"].isna().sum()),
    }
    return out.drop(columns=["_rank", "_gender"]), stats


def basis_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Count of rows per `gender_basis`, over the rows that carry a gender."""
    if df.empty or "gender_basis" not in df.columns:
        return {}
    g = df[df["gender"].notna()] if "gender" in df.columns else df
    return {str(k): int(v) for k, v in g["gender_basis"].value_counts(dropna=False).items()}


def recompute_parent_gender(directors: pd.DataFrame) -> pd.DataFrame:
    """Per-filing `pct_female_directors` / `pct_gender_stated` from consensus-corrected rows, keyed on
    `(ticker, accession_number)` so the caller writes only those columns back to `def14a_llm` (the filing's own
    `n_women_directors` keeps its precedence)."""
    if directors.empty:
        return pd.DataFrame(columns=["ticker", "accession_number", "pct_female_directors", "pct_gender_stated"])
    d = directors[directors["gender"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["ticker", "accession_number", "pct_female_directors", "pct_gender_stated"])
    d["_female"] = d["gender"].map(_is_female)
    d["_evidence"] = d["gender_basis"].map(lambda b: _basis_rank(b) >= EVIDENCE_RANK)
    out = (
        d.groupby(["ticker", "accession_number"])
        .agg(
            pct_female_directors=("_female", lambda s: round(float(s.mean()), 3)),
            pct_gender_stated=("_evidence", lambda s: round(float(s.mean()), 3)),
        )
        .reset_index()
    )
    return out


def log_consensus(stats: dict[str, int], before: dict[str, int], after: dict[str, int]) -> None:
    """Log the consensus stats and the basis distribution before and after it. `overturned == 0` and `filled == 0`
    over matched people means the name key matched nobody across filings: a mechanism failure, not a clean result."""
    logger.info(
        "gender consensus: %d rows -> %d distinct people | filled %d, overturned %d, unchanged %d, unkeyable %d",
        stats["rows"],
        stats["people"],
        stats["filled"],
        stats["overturned"],
        stats["unchanged"],
        stats["unmatched_rows"],
    )
    logger.info("gender_basis before: %s", before or "-")
    logger.info("gender_basis after:  %s", after or "-")
    if stats["people"] and not stats["overturned"] and not stats["filled"]:
        logger.warning(
            "gender consensus changed NOTHING across %d people -- if this persists the person key is matching nobody across filings", stats["people"]
        )


def finalise_gender(context: Context) -> None:
    """Cross-ticker gender consensus over `def14a_directors`, run once after the per-ticker loop: a director recurs
    across companies, so it needs every ticker's rows. Rewrites the directors' gender columns and the parent's gender
    ratios only when something changed."""
    # load: every ticker's directors
    directors = context.store.load(Tables.def14a_directors, columns=_DIRECTOR_COLUMNS, optional=True)
    if directors is None or directors.empty:
        return

    # compute: the consensus, logged against the basis distribution before it
    before = basis_distribution(directors)
    resolved, stats = consensus(directors)
    after = basis_distribution(resolved)
    log_consensus(stats, before, after)
    if not (stats["filled"] or stats["overturned"]):
        return

    # save: the changed gender columns
    context.store.save(Tables.def14a_directors, resolved[_DIRECTOR_COLUMNS], pk=["ticker", "accession_number", "name"])

    # save: the parent's fallback ratio (the filing's stated `n_women_directors` still wins)
    parent = recompute_parent_gender(resolved)
    if not parent.empty:
        context.store.save(Tables.def14a_llm, parent, pk=["ticker", "accession_number"])
        logger.info("gender consensus: refreshed pct_female_directors / pct_gender_stated on %d filings", len(parent))
