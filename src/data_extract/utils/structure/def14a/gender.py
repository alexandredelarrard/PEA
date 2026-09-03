"""
gender.py  (src/data_extract/utils/structure/def14a/gender.py)
---------------------------------------------------------------------
Cross-filing gender consensus over `def14a_directors`. Deterministic — no LLM.

Why this exists. `pct_female_directors` carries alpha, so the answer to "gender is 97.9% filled
while only 17.4% of proxies disclose it" is to make the field RIGHT, not to delete it. The
extraction side resolves gender from in-document evidence in a fixed order and records which
rule fired in `gender_basis` (`stated` > `honorific` > `pronoun` > `name`). This module is the
second half: a director's gender does not change, and directors recur across years and across
boards, so the many rows describing one PERSON can be reconciled to their
highest-provenance value.

Two things it buys that per-filing extraction cannot:
  * it FILLS a director whose own bio carried no honorific or pronoun (the pre-2001 ASCII case)
    from a filing where the evidence was present;
  * it CORRECTS an unstable first-name inference wherever any sibling row has real evidence.

The `overturned` count it logs is the direct measure of what the upgrade bought. An overturn
count of 0 on real data means the name key never matched anyone across filings -- a silent
failure of the mechanism, not a clean bill of health.

IDEMPOTENT: the consensus is recomputed from `gender_basis`, and a value is never DOWNGRADED to
a weaker basis, so a second run is a no-op. That matters because the pass runs at the end of
every extraction, and DEF 14A is a yearly filing -- an incremental day adds ~0 rows.
"""
from __future__ import annotations

import logging
import re
from collections import Counter

import pandas as pd

from src.data_extract.utils.structure.def14a.validate import clean_person_name

logger = logging.getLogger(__name__)

#: Provenance ranking. `stated` is the proxy identifying the director itself (a diversity
#: matrix); `honorific` and `pronoun` are in-document evidence; `name` is a bare first-name
#: prior with no evidence behind it at all.
BASIS_RANK = {"stated": 3, "honorific": 2, "pronoun": 1, "name": 0}
#: Above this rank a value is DOCUMENT evidence rather than an inference -- the threshold
#: `pct_gender_stated` reports and the one a consumer can filter on.
EVIDENCE_RANK = BASIS_RANK["honorific"]

#: Post-nominals must be matched AFTER the dots are removed, not before. The academic ones are
#: written `Ph.D.` / `M.D.` / `DVM`, and `\bphd\b` does not match `ph.d.` -- so with the dots
#: still in place the suffix survived, `_NON_ALPHA_RE` then split it into `ph d`, and `d` became
#: the SURNAME. Measured: `person_key("Albert Bourla, DVM, Ph.D.")` returned `d|a`, which is not
#: merely a failure to match `A. Bourla` (`bourla|a`) -- it collapses every credentialed
#: director with the same first initial onto ONE key, so the consensus pass would propagate one
#: person's gender onto unrelated people.
_DOT_RE = re.compile(r"\.")
_SUFFIX_RE = re.compile(
    r"\b(?:jr|sr|ii|iii|iv|v|phd|md|dvm|dds|dsc|edd|pharmd|mph|cpa|cfa|esq)\b", re.I)
_NON_ALPHA_RE = re.compile(r"[^a-z ]+")


def _norm(value: object) -> str | None:
    """Lower-cased text, or None for anything that is not a real string.

    `value or ""` is NOT enough: a null read back from Postgres arrives as `float('nan')`,
    which is TRUTHY, so that idiom returns the nan and the next `.strip()` raises
    `AttributeError: 'float' object has no attribute 'strip'`. Every value in this module comes
    out of a DataFrame, so a null-bearing column is the normal case, not an edge one.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def person_key(name: str | None) -> str | None:
    """A person key stable across filings and across companies: `lastname|firstinitial`.

    Keyed on the first INITIAL, not the first name, because a filer's own spelling drifts:
    "Katherine J. Smith" and "Kathy Smith" are one director and must reconcile. Generational
    suffixes and post-nominals are stripped first -- "John Smith Jr." and "John Smith" are the
    same person for this purpose, and treating them as two would split their evidence.

    This is deliberately the SAME key Phase 5's vote role map uses, so a nominee that matches
    there matches here.
    """
    cleaned = clean_person_name(name)
    if not cleaned:
        return None
    # dots first, so `Ph.D.` becomes `phd` and the suffix pattern can see it; DELETED rather
    # than replaced with a space, because a space would leave `ph d` and put `d` in surname
    # position. `H.` -> `h` is unaffected either way.
    flat = _NON_ALPHA_RE.sub(" ", _SUFFIX_RE.sub(" ", _DOT_RE.sub("", cleaned.lower())))
    parts = [p for p in flat.split() if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-1]}|{parts[0][0]}"


def consensus(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Resolve one gender per PERSON and apply it to every row of that person.

    `df` needs `name`, `gender`, `gender_basis`. Returns (a copy with `gender` /
    `gender_basis` / `person_key` set, stats).

    Resolution, per person:
      1. the highest basis rank present for that person wins outright -- a single `stated` row
         beats any number of `name`-inferred ones, because provenance is not a popularity
         contest and a first-name prior is not evidence;
      2. within that rank, the majority value;
      3. ties break toward the value that appears in the most recent row, which is the only
         tiebreak that is not arbitrary.
    """
    out = df.copy()
    for col in ("gender", "gender_basis"):
        if col not in out.columns:
            out[col] = None
    out["person_key"] = out["name"].map(person_key)
    out["_rank"] = out["gender_basis"].map(lambda b: BASIS_RANK.get(_norm(b), -1))
    out["_gender"] = out["gender"].map(_norm)

    resolved: dict[str, tuple[str, str]] = {}
    for key, grp in out[out["person_key"].notna()].groupby("person_key"):
        known = grp[grp["_gender"].notna()]   # notna() covers None AND pd.NA
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
        resolved[key] = (winner, basis)

    filled = overturned = unchanged = 0
    new_gender, new_basis = [], []
    for _, r in out.iterrows():
        key = r["person_key"]
        if key is None or key not in resolved:
            new_gender.append(None if pd.isna(r["_gender"]) else r["_gender"])
            new_basis.append(_norm(r["gender_basis"]))
            continue
        winner, basis = resolved[key]
        # `_norm` returns None, but a pyarrow-backed string column turns that into `pd.NA` on
        # the way through `.map`, and `pd.NA is None` is False -- which silently counted every
        # FILLED row as an OVERTURNED one, i.e. overstated exactly the number this pass is
        # judged on. Compare with `pd.isna`, never with `is None`.
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
        own_rank = BASIS_RANK.get(_norm(r["gender_basis"]), -1)
        new_basis.append(_norm(r["gender_basis"]) if own_rank >= BASIS_RANK[basis] else basis)

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
    """Per-filing `pct_female_directors` / `pct_gender_stated` from consensus-corrected rows.

    Returned as its own frame keyed on `(ticker, accession_number)` so the caller writes only
    those columns back to `def14a_llm` -- the fallback ratio therefore gets BETTER, not
    narrower, and the existing precedence (the filing's own `n_women_directors` first) is
    untouched.
    """
    if directors.empty:
        return pd.DataFrame(columns=["ticker", "accession_number",
                                     "pct_female_directors", "pct_gender_stated"])
    d = directors[directors["gender"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["ticker", "accession_number",
                                     "pct_female_directors", "pct_gender_stated"])
    d["_female"] = d["gender"].map(lambda g: bool(_norm(g) or "") and _norm(g).startswith("f"))
    d["_evidence"] = d["gender_basis"].map(
        lambda b: BASIS_RANK.get(_norm(b), -1) >= EVIDENCE_RANK)
    out = d.groupby(["ticker", "accession_number"]).agg(
        pct_female_directors=("_female", lambda s: round(float(s.mean()), 3)),
        pct_gender_stated=("_evidence", lambda s: round(float(s.mean()), 3)),
    ).reset_index()
    return out


def log_consensus(log, stats: dict, before: dict, after: dict) -> None:
    """Print the evidence that the upgrade worked. `overturned == 0` on real data means the name
    key matched nobody across filings, which is a mechanism failure, not a clean result."""
    log.info("gender consensus: %d rows -> %d distinct people | filled %d, overturned %d, "
             "unchanged %d, unkeyable %d",
             stats["rows"], stats["people"], stats["filled"], stats["overturned"],
             stats["unchanged"], stats["unmatched_rows"])
    log.info("gender_basis before: %s", before or "-")
    log.info("gender_basis after:  %s", after or "-")
    if stats["people"] and not stats["overturned"] and not stats["filled"]:
        log.warning("gender consensus changed NOTHING across %d people -- if this persists the "
                    "person key is matching nobody across filings", stats["people"])
