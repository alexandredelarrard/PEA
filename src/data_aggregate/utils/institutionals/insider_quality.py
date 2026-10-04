"""
insider_quality.py  (src/data_aggregate/utils/institutionals/insider_quality.py)
-------------------------------------------------------------------
Scope and repair the Form 3/4/5 transaction log before any feature is summed off it.

Scope: non-derivative, common-stock, open-market (P/S) rows with a usable price. Repair: a filed
`price_per_share` more than `PRICE_TOLERANCE` above the split-free consensus (median price of
same-ticker, same-class common-stock filings over a trailing window ending on the filing day) is
replaced by `shares x consensus`; too-low prices are only counted. The clock is `filing_date`, so a
later filing never enters an earlier reference. Corrupt `shares` cannot be separated by size, so
oversized rows are logged, never dropped.

Versions: a trade repeated in a later accession (same ticker, transaction day, code, shares, price
and post-trade holding) is kept once, from its earliest filing. An amendment replaces the original's
whole (transaction day, code, security type) cell from its own filing day; every row carries the day
the trade was first disclosed (`anchor`) and the interval `[visible_from, visible_until)` in which its
record is the one known. The stored table is never deduplicated.
"""

from __future__ import annotations

import logging
import re
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.data_utils import to_day

_log = logging.getLogger(__name__)

#: Open-market purchase/sale codes; every other code is non-discretionary and never netted.
OPEN_MARKET_CODES: tuple[str, ...] = ("P", "S")

#: Codes whose `price_per_share` is a market price and may back the consensus (`F` is withholding at the close).
MARKET_PRICED_CODES: tuple[str, ...] = ("P", "S", "F")

#: Common/ordinary-stock security titles, including common misspellings; preferred and 401(k) units fall through.
COMMON_STOCK_RE = re.compile(
    r"COMMON|COMON|COMMOM|COMM|ORDINARY|REGISTER|CLASS\s+[A-Z]\b|SHARES\s+OF\s+BENEFICIAL|DEPOSITARY|" r"^\s*(?:COM|STOCK|SHARES)\s*$", re.I
)

#: Max ratio of filed price to the trailing consensus before the price is treated as corrupt.
PRICE_TOLERANCE: float = 10.0

#: Share-class token in a security title, so the consensus is built within one class.
_CLASS_RE = re.compile(r"\bCLASS\s+([A-Z])\b|\bSERIES\s+([A-Z])\b", re.I)

#: Trailing publication-time window behind the consensus median.
CONSENSUS_WINDOW: str = "31D"

#: A single transaction above this share of the company is reported, never dropped.
FLAG_PCT_SHARES_OUTSTANDING: float = 0.25

#: Free-text `officer_title` -> role, first match wins (a "Chairman, President & CEO" is a CEO).
CHIEF = r"(?:CHIEF|CHF)"
OFFICER = r"OFF(?:ICER|CR|R)"
OPERATIONS = r"OPER(?:ATION(?:S)?|ATING)?"
ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "CEO",
        rf"\b{CHIEF}\s+EXEC(?:UTIVE)?\s+{OFFICER}\b" rf"|\bC\s*E\s*O\b",
    ),
    ("CFO", rf"\b{CHIEF}\s+FIN(?:ANCIAL)?\s+{OFFICER}\b" rf"|\bPRINCIPAL\s+FIN(?:ANCIAL)?\s+{OFFICER}\b" rf"|\bC\s*F\s*O\b" rf"|\bTREASURER\b"),
    ("COO", rf"\b{CHIEF}\s+{OPERATIONS}\s+{OFFICER}\b" rf"|\bC\s*O\s*O\b"),
    ("CTO_or_technology", rf"\b{CHIEF}\s+TECH(?:NOLOGY)?\s+{OFFICER}\b" rf"|\bC\s*T\s*O\b" rf"|\bTECH(?:NOLOGY)?\b"),
    ("CMO_or_marketing", rf"\b{CHIEF}\s+MARKETING\s+{OFFICER}\b"),
)
OTHER_OFFICER = "other_named_officer"

#: Expected share of titled officer rows falling through to `OTHER_OFFICER` (real named officers outside the map).
ROLE_FALLTHROUGH_MEASURED: float = 0.320

_ROLE_COMPILED = tuple((name, re.compile(pat, re.I)) for name, pat in ROLE_PATTERNS)

#: Decimals of the repeat-copy key: the SEC zip's precision, so its record and EDGAR's of one trade match.
COPY_KEY_DECIMALS: int = 2

#: Columns of the repeat-copy key, built by `_copy_key`.
COPY_KEY: tuple[str, ...] = ("ticker", "transaction_day", "code", "shares", "price", "owned_after")

#: Columns of an amendment cell: the unit a Form 4/A restates.
CELL_KEY: tuple[str, ...] = ("transaction_day", "code", "security_type")

#: Columns every cleaned row carries for point-in-time aggregation.
VISIBILITY_COLUMNS: tuple[str, ...] = ("anchor", "visible_from", "visible_until")


def common_stock_mask(security_title: pd.Series) -> pd.Series:
    """True where the security traded is common/ordinary stock; gates both the consensus and the features."""
    return security_title.fillna("").astype(str).str.contains(COMMON_STOCK_RE)


def officer_role(title: object) -> str:
    """Free-text `officer_title` -> the first matching `ROLE_PATTERNS` name, else `OTHER_OFFICER` (also for a blank)."""
    s = "" if title is None else str(title)
    for name, rx in _ROLE_COMPILED:
        if rx.search(s):
            return name
    return OTHER_OFFICER


def security_class(security_title: pd.Series) -> pd.Series:
    """Share-class key for the consensus: `A`, `B`, ... or `COMMON` for an unclassed title."""
    s = security_title.fillna("").astype(str)
    got = s.str.extract(_CLASS_RE)
    return got[0].fillna(got[1]).str.upper().fillna("COMMON")


def consensus_price(txns: pd.DataFrame, *, window: str = CONSENSUS_WINDOW) -> pd.Series:
    """Per-row reference price: trailing-`window` rolling median of per-day medians of common-stock,
    market-priced filed prices for the same ticker and share class, keyed on `filing_date` (the
    publication clock). Aligned to `txns.index`; NaN where the filing day has no reference.
    """
    need = {"ticker", "filing_date", "price_per_share", "transaction_code", "security_title"}
    if txns.empty or not need.issubset(txns.columns):
        return pd.Series(np.nan, index=txns.index, dtype="float64")

    code = txns["transaction_code"].astype(str).str.upper().str.strip()
    pps = pd.to_numeric(txns["price_per_share"], errors="coerce")
    publication_day = to_day(txns["filing_date"])
    usable = code.isin(MARKET_PRICED_CODES) & (pps > 0) & publication_day.notna() & common_stock_mask(txns["security_title"]) & txns["ticker"].notna()

    if not usable.any():
        return pd.Series(np.nan, index=txns.index, dtype="float64")

    klass = security_class(txns["security_title"])
    src = pd.DataFrame(
        {
            "ticker": txns.loc[usable, "ticker"].astype(str),
            "klass": klass[usable],
            "publication_day": publication_day[usable],
            "pps": pps[usable],
        }
    )
    day = src.groupby(["ticker", "klass", "publication_day"], sort=False)["pps"].median().rename("m").reset_index()

    out = []
    for (tkr, cls), g in day.groupby(["ticker", "klass"], sort=False):
        g = g.set_index("publication_day").sort_index()
        med = g["m"].rolling(window).median()
        out.append(pd.DataFrame({"ticker": tkr, "klass": cls, "publication_day": g.index, "consensus": med.to_numpy()}))
    ref = pd.concat(out, ignore_index=True)

    keyed = pd.DataFrame({"ticker": txns["ticker"].astype(str), "klass": klass, "publication_day": publication_day})
    merged = keyed.merge(ref, on=["ticker", "klass", "publication_day"], how="left")
    merged.index = txns.index
    return merged["consensus"]


def clean_transactions(insider: pd.DataFrame, *, price_tolerance: float = PRICE_TOLERANCE) -> tuple[pd.DataFrame, dict]:
    """Scope to priced common-stock open-market trades and repair the mispriced ones.

    Returns `(frame, diagnostics)`. Added columns: `code`, `day` (normalised `filing_date`, the
    point-in-time stamp, never `transaction_date`), `value` (repaired USD), `shares_n`,
    `price_repaired`, `role`, `in_exercise_package` (an `S` sharing accession and transaction
    date with an `M`) and `VISIBILITY_COLUMNS` (see `versioned_records`). Derivative, non-common
    and unpriced rows are dropped, never zero-filled; later repeat copies are dropped.
    """

    need = {"ticker", "filing_date", "transaction_code", "shares", "value_usd"}
    diag: dict = {"input_rows": 0 if insider is None else len(insider)}
    if insider is None or insider.empty or not need.issubset(insider.columns):
        return pd.DataFrame(), diag

    df_trades = insider.copy()
    df_trades["ticker"] = df_trades["ticker"].astype(str).str.upper().str.strip()
    df_trades["code"] = df_trades["transaction_code"].astype(str).str.upper().str.strip()
    df_trades["day"] = to_day(df_trades["filing_date"])
    df_trades["shares_n"] = pd.to_numeric(df_trades["shares"], errors="coerce")
    pps = pd.to_numeric(cast(pd.Series, df_trades.get("price_per_share")), errors="coerce")

    # Computed before the scope cut, which removes the derivative `M` leg of the package.
    df_trades["in_exercise_package"] = _exercise_packages(df_trades)

    if "security_type" in df_trades.columns:
        df_trades = df_trades[df_trades["security_type"].astype(str).str.lower().eq("nonderiv")]

    if "security_title" in df_trades.columns:
        df_trades = df_trades[common_stock_mask(df_trades["security_title"])]

    df_trades = df_trades[df_trades["code"].isin(OPEN_MARKET_CODES) & df_trades["day"].notna() & (df_trades["ticker"] != "")]
    df_trades = df_trades.dropna(subset=["shares_n"])
    pps = pps.reindex(df_trades.index)
    diag["scoped_rows"] = len(df_trades)
    if df_trades.empty:
        return pd.DataFrame(), diag

    df_trades, versions = versioned_records(df_trades, pps)
    diag.update(versions)
    pps = pps.reindex(df_trades.index)

    priced = pps > 0
    diag["dropped_unpriced"] = int((~priced).sum())
    diag["dropped_unpriced_shares"] = float(df_trades.loc[~priced, "shares_n"].sum()) / df_trades["shares_n"].sum()
    diag["unpriced_events"] = df_trades.loc[~priced, ["ticker", "day", "code", *VISIBILITY_COLUMNS]].copy()
    df_trades, pps = df_trades[priced], pps[priced]

    ref = consensus_price(insider).reindex(df_trades.index)
    ratio = pps / ref.where(ref > 0)
    # One-sided repair: a too-low ratio may be a genuine low-priced row under a reused ticker, so it is only counted.
    bad = ratio.notna() & (ratio > price_tolerance)
    low = ratio.notna() & (ratio < 1.0 / price_tolerance)
    raw_value = pd.to_numeric(df_trades["value_usd"], errors="coerce")

    # Repair rather than drop: the trade and its share count are real, only the price is wrong.
    df_trades["value"] = raw_value.where(~bad, df_trades["shares_n"] * ref)
    df_trades["price_repaired"] = bad

    # A row whose price survived but whose `value_usd` is missing is still a real trade.
    df_trades["value"] = df_trades["value"].fillna(df_trades["shares_n"] * pps)

    diag["repaired_rows"] = int(bad.sum())
    diag["underpriced_rows"] = int(low.sum())
    diag["value_before"] = float(raw_value.sum())
    diag["value_after"] = float(df_trades["value"].sum())
    diag["no_consensus_rows"] = int(ratio.isna().sum())

    df_trades["role"] = df_trades["officer_title"].map(officer_role) if "officer_title" in df_trades.columns else OTHER_OFFICER
    for flag in ("is_director", "is_officer", "is_ten_pct_owner"):
        df_trades[flag] = pd.to_numeric(cast(pd.Series, df_trades.get(flag)), errors="coerce")
    df_trades["is_10b5_1"] = pd.to_numeric(cast(pd.Series, df_trades.get("is_10b5_1")), errors="coerce")

    _log.info(
        "insider: %s rows -> %s scoped, %s unpriced dropped, %s overpriced repaired, %s underpriced left as filed ($%.3ftn -> $%.3fbn)",
        diag["input_rows"],
        diag["scoped_rows"],
        diag["dropped_unpriced"],
        diag["repaired_rows"],
        diag["underpriced_rows"],
        diag["value_before"] / 1e12,
        diag["value_after"] / 1e9,
    )
    _log.info(
        "insider versions: %s repeat-copy group(s), %s later cop(ies) dropped ($%.3fbn as filed); %s amendment(s) linked "
        "(%s ambiguous), cells %s superseded / %s partial / %s new / %s identical",
        diag["copy_groups"],
        diag["copy_rows_dropped"],
        diag["copy_value_removed"] / 1e9,
        diag["amendments_linked"],
        diag["amendments_ambiguous"],
        diag["amendment_cells_superseded"],
        diag["amendment_cells_partial"],
        diag["amendment_cells_new"],
        diag["amendment_cells_identical"],
    )
    return df_trades, diag


def _exercise_packages(t: pd.DataFrame) -> pd.Series:
    """True for an `S` row in the same accession and transaction date as an option exercise `M`
    (a mechanical exercise-and-sell, excluded from discretionary-sell features)."""
    if "accession_number" not in t.columns or "transaction_date" not in t.columns:
        return pd.Series(False, index=t.index)
    key = pd.MultiIndex.from_arrays([t["accession_number"].astype(str), pd.to_datetime(t["transaction_date"], errors="coerce")])
    has_m = pd.Series(t["code"].eq("M").to_numpy(), index=key).groupby(level=[0, 1]).any()
    return pd.Series(has_m.reindex(key).to_numpy(), index=t.index).fillna(False) & t["code"].eq("S")


def versioned_records(df_scoped: pd.DataFrame, pps: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Collapse repeat copies (REQ-010) and supersede amended cells (REQ-011) on scoped rows, never reordering them.

    Adds `anchor` (the day the trade was first disclosed), `visible_from` and `visible_until`
    (exclusive, NaT = open). A row in no copy group and no amended cell keeps anchor =
    visible_from = `day` and an open interval. Linked amendment rows never enter the copy
    collapse. Returns `(frame, counts)`.
    """
    df_rows = df_scoped.assign(
        anchor=df_scoped["day"], visible_from=df_scoped["day"], visible_until=pd.Series(pd.NaT, index=df_scoped.index, dtype=df_scoped["day"].dtype)
    )
    counts: dict = {
        "copy_groups": 0,
        "copy_rows_dropped": 0,
        "copy_value_removed": 0.0,
        "amendments_linked": 0,
        "amendments_ambiguous": 0,
        "amendment_cells_superseded": 0,
        "amendment_cells_partial": 0,
        "amendment_cells_new": 0,
        "amendment_cells_identical": 0,
        "amendment_rows_identical": 0,
    }
    if "accession_number" not in df_rows.columns:
        return df_rows, counts

    df_key = _copy_key(df_rows, pps)
    links, counts["amendments_ambiguous"] = _link_amendments(df_rows, df_key)
    counts["amendments_linked"] = len(links)
    df_plan, cell_counts = _supersede_plan(df_key, links)
    counts.update(cell_counts)
    linked = df_key["accession_number"].isin(links.index)
    drop_copy, df_owners, copy_counts = _collapse_copies(df_rows, df_key, eligible=~linked)
    counts.update(copy_counts)

    if not df_plan.empty:
        df_rows.loc[df_plan.index, "anchor"] = df_plan["anchor"].to_numpy()
        df_rows.loc[df_plan.index, "visible_until"] = df_plan["visible_until"].to_numpy()
    for column in ("owner_ciks", "n_reporting_owners"):
        if column in df_rows.columns and not df_owners.empty:
            df_rows[column] = df_rows[column].astype(object)
            df_rows.loc[df_owners.index, column] = df_owners[column]
    identical = df_rows.index.isin(df_plan.index[df_plan["identical"]]) if not df_plan.empty else np.zeros(len(df_rows), dtype=bool)
    return df_rows[~(drop_copy.to_numpy() | identical)], counts


def _copy_key(df: pd.DataFrame, pps: pd.Series) -> pd.DataFrame:
    """Per row: accession, filing day, the `COPY_KEY` fields (rounded to `COPY_KEY_DECIMALS`) and the `CELL_KEY` fields."""
    nan = pd.Series(np.nan, index=df.index)
    owned = pd.to_numeric(df["shares_owned_after"], errors="coerce") if "shares_owned_after" in df.columns else nan
    txn_day = to_day(df["transaction_date"]) if "transaction_date" in df.columns else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    security = df["security_type"].astype(str).str.lower() if "security_type" in df.columns else pd.Series("nonderiv", index=df.index)
    return pd.DataFrame(
        {
            "accession_number": df["accession_number"].astype(str),
            "day": df["day"],
            "ticker": df["ticker"],
            "transaction_day": txn_day.astype("datetime64[ns]"),
            "code": df["code"],
            "security_type": security,
            "shares": df["shares_n"].round(COPY_KEY_DECIMALS),
            "price": pd.to_numeric(pps.reindex(df.index), errors="coerce").round(COPY_KEY_DECIMALS),
            "owned_after": owned.round(COPY_KEY_DECIMALS),
        },
        index=df.index,
    )


def _owner_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (`accession_number`, `owner`) from `owner_ciks`, else the primary `owner_cik`; CIKs without leading zeros."""
    blank = pd.Series(pd.NA, index=df.index, dtype="string")
    listed = (df["owner_ciks"] if "owner_ciks" in df.columns else blank).astype("string").str.strip()
    primary = (df["owner_cik"] if "owner_cik" in df.columns else blank).astype("string").str.strip()
    owners = listed.where(listed.fillna("").ne(""), primary)
    df_pairs = pd.DataFrame({"accession_number": df["accession_number"].astype(str), "owner": owners.str.split(",")}).explode("owner")
    df_pairs["owner"] = df_pairs["owner"].astype("string").str.strip().str.lstrip("0")
    return df_pairs[df_pairs["owner"].fillna("").ne("")].drop_duplicates()


def _link_amendments(df: pd.DataFrame, df_key: pd.DataFrame) -> tuple[pd.Series, int]:
    """Amendment accession -> its original, and the number of amendments with several candidates.

    The original shares the ticker and an owner, is not an amendment, and was filed on the
    amendment's `original_submission_date`, on or before the amendment. Among several
    candidates the one sharing the most cells wins, then the lowest accession number.
    """
    none = pd.Series(dtype="string")
    if "document_type" not in df.columns or "original_submission_date" not in df.columns:
        return none, 0
    amend = df["document_type"].astype("string").str.strip().str.upper().str.endswith("/A").fillna(False).astype(bool)
    if not amend.any():
        return none, 0

    df_acc = df_key[["accession_number", "ticker"]].assign(
        day=df_key["day"].astype("datetime64[ns]"), submitted=to_day(df["original_submission_date"]).astype("datetime64[ns]")
    )
    df_amend = df_acc[amend].drop_duplicates("accession_number")
    df_amend = df_amend[df_amend["submitted"].notna() & (df_amend["submitted"] <= df_amend["day"])]
    df_orig = df_acc[~amend].drop_duplicates("accession_number")[["accession_number", "ticker", "day"]]
    df_owner = _owner_pairs(df)
    df_orig = df_orig.merge(df_owner, on="accession_number").rename(columns={"accession_number": "original", "day": "submitted"})
    df_pairs = df_amend.merge(df_owner, on="accession_number").merge(df_orig, on=["ticker", "submitted", "owner"])
    df_pairs = df_pairs[["accession_number", "original"]].drop_duplicates()
    if df_pairs.empty:
        return none, 0

    df_cells = df_key[["accession_number", *CELL_KEY]].drop_duplicates()
    shared = (
        df_pairs.merge(df_cells, on="accession_number")
        .merge(df_cells.rename(columns={"accession_number": "original"}), on=["original", *CELL_KEY])
        .groupby(["accession_number", "original"])
        .size()
        .rename("shared")
    )
    df_pairs = df_pairs.join(shared, on=["accession_number", "original"]).fillna({"shared": 0})
    df_pairs = df_pairs.sort_values(["accession_number", "shared", "original"], ascending=[True, False, True])
    n_candidates = df_pairs.groupby("accession_number").size()
    links = df_pairs.drop_duplicates("accession_number").set_index("accession_number")["original"]
    return links, int((n_candidates > 1).sum())


def _supersede_plan(df_key: pd.DataFrame, links: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Per row of a linked original or amendment: `anchor`, `visible_until` and `identical`.

    The records of one (original, cell) are ordered original first, then amendments by filing
    day and accession. A record equal (as a multiset of shares, price and post-trade holding)
    to the one before it is `identical` and adds nothing; every other record is visible from
    its filing day until the next one's, anchored on the first record's day.
    """
    columns = ["anchor", "visible_until", "identical"]
    counts = dict.fromkeys(
        ("amendment_cells_superseded", "amendment_cells_partial", "amendment_cells_new", "amendment_cells_identical", "amendment_rows_identical"), 0
    )
    if links.empty:
        return pd.DataFrame(columns=columns), counts

    original = df_key["accession_number"].map(links)
    member = original.notna() | df_key["accession_number"].isin(set(links.to_numpy()))
    df_rows = df_key[member].assign(original=original[member].fillna(df_key.loc[member, "accession_number"]), row=df_key.index[member])
    df_rows["is_original"] = df_rows["original"].eq(df_rows["accession_number"])
    shares, price, owned = (df_rows[c].to_numpy(dtype="float64").astype("U32") for c in ("shares", "price", "owned_after"))
    df_rows["record"] = np.char.add(np.char.add(np.char.add(np.char.add(shares, "|"), price), "|"), owned)
    cell = ["original", *CELL_KEY]
    reporter = [*cell, "accession_number"]

    df_rep = (
        df_rows.sort_values([*reporter, "record"])
        .groupby(reporter, sort=False, dropna=False)
        .agg(day=("day", "first"), is_original=("is_original", "first"), n_rows=("record", "size"), signature=("record", _joined))
        .reset_index()
        .sort_values([*cell, "is_original", "day", "accession_number"], ascending=[True] * len(cell) + [False, True, True])
    )
    df_rep["identical"] = df_rep["signature"].eq(df_rep.groupby(cell, sort=False, dropna=False)["signature"].shift())
    df_eff = df_rep[~df_rep["identical"]].copy()
    by_cell = df_eff.groupby(cell, sort=False, dropna=False)
    df_eff["anchor"] = by_cell["day"].transform("first")
    df_eff["visible_until"] = by_cell["day"].shift(-1)
    replaces = by_cell.cumcount() > 0

    counts["amendment_cells_superseded"] = int(replaces.sum())
    counts["amendment_cells_partial"] = int((replaces & (df_eff["n_rows"] < by_cell["n_rows"].shift())).sum())
    counts["amendment_cells_new"] = int((~replaces & ~df_eff["is_original"]).sum())
    counts["amendment_cells_identical"] = int(df_rep["identical"].sum())
    counts["amendment_rows_identical"] = int(df_rep.loc[df_rep["identical"], "n_rows"].sum())

    df_plan = (
        df_rows[[*reporter, "row"]]
        .merge(df_rep[[*reporter, "identical"]], on=reporter)
        .merge(df_eff[[*reporter, "anchor", "visible_until"]], on=reporter, how="left")
    )
    return df_plan.set_index("row")[columns].rename_axis(None), counts


def _joined(values: pd.Series) -> str:
    """The values of one group joined by commas, in order."""
    return ",".join(map(str, values))


def _collapse_copies(df: pd.DataFrame, df_key: pd.DataFrame, *, eligible: pd.Series) -> tuple[pd.Series, pd.DataFrame, dict]:
    """REQ-010 on the `eligible` rows: rows of different accessions with an equal `COPY_KEY` are one trade.

    The earliest accession (filing day, then accession number) keeps its rows; the others are
    dropped. A NaN key field never matches. Returns the drop mask, the kept rows' union of
    owners (`owner_ciks`, `n_reporting_owners`) and the counts.
    """
    drop = pd.Series(False, index=df.index)
    counts = {"copy_groups": 0, "copy_rows_dropped": 0, "copy_value_removed": 0.0}
    df_cand = df_key[eligible].dropna(subset=list(COPY_KEY))
    n_accessions = df_cand.groupby(list(COPY_KEY), sort=False)["accession_number"].transform("nunique")
    df_group = df_cand[n_accessions > 1]
    if df_group.empty:
        return drop, pd.DataFrame(columns=["owner_ciks", "n_reporting_owners"]), counts

    df_group = df_group.assign(group=df_group.groupby(list(COPY_KEY), sort=False).ngroup())
    keeper = df_group.sort_values(["group", "day", "accession_number"]).drop_duplicates("group").set_index("group")["accession_number"]
    later = df_group["accession_number"].ne(df_group["group"].map(keeper))
    drop.loc[later.index[later.to_numpy()]] = True
    counts["copy_groups"] = int(df_group["group"].nunique())
    counts["copy_rows_dropped"] = int(drop.sum())
    counts["copy_value_removed"] = float(pd.to_numeric(df.loc[drop, "value_usd"], errors="coerce").sum())

    df_union = (
        df_group[["group", "accession_number"]]
        .drop_duplicates()
        .merge(_owner_pairs(df.loc[df_group.index]), on="accession_number")
        .assign(owner=lambda d: d["owner"].str.zfill(10))
        .drop_duplicates(["group", "owner"])
        .sort_values(["group", "owner"])
        .groupby("group")["owner"]
        .agg(owner_ciks=_joined, n_reporting_owners="size")
    )
    df_kept = df_group.loc[~later.to_numpy(), ["group"]]
    df_owners = df_kept.join(df_union, on="group")[["owner_ciks", "n_reporting_owners"]]
    return drop, df_owners, counts


def asof_values(frame: pd.DataFrame | None, tickers: pd.Series, days: pd.Series) -> pd.Series:
    """Value of a wide (date x ticker) `frame` as of each `(ticker, day)`, forward-filled: the last
    row at or before the day, NaN before the first observation (never a look-ahead).
    """
    idx = pd.RangeIndex(len(tickers)) if tickers.index.has_duplicates else tickers.index
    if frame is None or frame.empty:
        return pd.Series(np.nan, index=idx, dtype="float64")
    f = frame.ffill()
    pos = f.index.searchsorted(pd.to_datetime(days).to_numpy(), side="right") - 1
    known = (pos >= 0) & tickers.isin(f.columns).to_numpy()
    out = np.full(len(tickers), np.nan)
    if known.any():
        out[known] = f.to_numpy()[pos[known], f.columns.get_indexer(pd.Index(tickers[known]))]
    return pd.Series(out, index=idx, dtype="float64")


def report_oversized(t: pd.DataFrame, shares_outstanding: pd.DataFrame | None, *, threshold: float = FLAG_PCT_SHARES_OUTSTANDING) -> pd.DataFrame:
    """Transactions above `threshold` of shares outstanding, as a frame to log (never a filter);
    empty when no share count is available.
    """
    if t.empty or shares_outstanding is None or shares_outstanding.empty:
        return pd.DataFrame()
    so = asof_values(shares_outstanding, t["ticker"], t["day"]).to_numpy()
    pct = t["shares_n"].to_numpy() / np.where(so > 0, so, np.nan)
    hit = pct > threshold
    if not hit.any():
        return pd.DataFrame()
    cols = [c for c in ("ticker", "day", "code", "owner_name", "shares_n", "value") if c in t]
    flagged = t.loc[hit, cols].copy()
    flagged["pct_shares_outstanding"] = pct[hit]
    return flagged.sort_values("pct_shares_outstanding", ascending=False)
