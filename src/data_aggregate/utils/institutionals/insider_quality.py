"""Canonical insider identities, common-stock scope, public filing versions and quote quality.

Corrections retain original disclosure anchors and half-open visibility intervals.
Unsupported quantities and execution prices remain separate evidence for downstream masks.
"""

from __future__ import annotations

import logging
import re
from typing import Any, cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.level_basis import _period_close
from src.utils.names import clean_person_name
from src.utils.price_adjustments import ShareBasis
from src.utils.string import normalise_ticker, pad_cik

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
    ("CFO", rf"\b{CHIEF}\s+FIN(?:ANCIAL)?\s+{OFFICER}\b" rf"|\bPRINCIPAL\s+FIN(?:ANCIAL)?\s+{OFFICER}\b" rf"|\bC\s*F\s*O\b"),
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

#: Ulps below a half that still round up: a filed decimal half (105.085) is stored just below it in binary.
_HALF_ULPS: float = 8.0

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
    s = clean_officer_title(title)
    if re.search(r"\b(?:FORMER|RETIRED|SUBSIDIAR(?:Y|IES)|DIVISION(?:AL)?|REGIONAL|SEGMENT|BUSINESS\s+UNIT)\b", s, re.I):
        return OTHER_OFFICER
    for name, rx in _ROLE_COMPILED:
        match = rx.search(s)
        if match is None:
            continue
        # A named entity after the role is a unit/affiliate, not another issuer title.
        tail = s[match.end() :].strip(" ,;&/:-().")
        companion = r"^(?:AND\s+)?(?:CHIEF\b|C\s*[BEFOT]\s*O\b|PRES(?:IDENT)?\b|(?:SR|SENIOR)\s+(?:VP|VICE|EXEC)\b|VP\b|VICE\b|[ES]VP\b|MD\b|MANAGING\s+DIRECTOR\b|INTERIM\b|ACTING\b|EXEC(?:UTIVE)?\b|CHAIR\w*\b|DIR(?:ECTOR)?\b|TREASURER\b|CONTROLLER\b|SECRETARY\b|PRINCIPAL\b|(?:CO[- ]?)?FOUNDER\b)"
        if not tail or re.match(companion, tail, re.I):
            return name
        return OTHER_OFFICER
    return OTHER_OFFICER


def clean_officer_title(value: object) -> str:
    """Normalize title punctuation and explicit abbreviations, preserving role modifiers."""
    if not isinstance(value, str):
        return ""
    title = value.replace(".", " ").strip()
    title = re.sub(r",\s*", ", ", title)
    title = re.sub(r"\bCHIEF\s+EXEC\b(?=\s+OFF(?:ICER|CR|R)\b)", "Chief Executive", title, flags=re.I)
    title = re.sub(r"\bCHIEF\s+EXEC\b(?!\s*UTIVE)", "Chief Executive Officer", title, flags=re.I)
    return re.sub(r"\s+", " ", title)


def owner_name_key(value: object) -> str:
    """Strict full-name comparison tokens; actual initials and surname particles survive."""
    name = clean_person_name(value)
    if not name:
        return ""
    tokens = re.findall(r"[^\W\d_]+", name.casefold())
    return " ".join(sorted(token for token in tokens if token not in {"mr", "mrs", "ms", "dr", "jr", "sr", "ii", "iii", "iv"}))


def _canonical_inputs(df_rows: pd.DataFrame, role_evidence: pd.DataFrame | None) -> pd.DataFrame:
    """Clean shared downstream fields and enrich blank titles from prior public identity evidence."""
    df_rows = df_rows.copy().reset_index(drop=True)
    df_rows["ticker"] = df_rows["ticker"].fillna("").map(normalise_ticker)
    df_rows["source_row"] = np.arange(len(df_rows))
    for column in ("officer_title", "owner_name", "owner_cik", "issuer_cik", "direct_indirect", "security_title"):
        if column not in df_rows:
            df_rows[column] = ""
    for column in ("owner_cik", "issuer_cik"):
        df_rows[column + "_raw"] = df_rows[column]
        values = df_rows[column]
        df_rows[column] = values.map({value: pad_cik(value) for value in values.dropna().unique()}).fillna("")
    values = df_rows["owner_name"]
    df_rows["owner_name_clean"] = values.map({value: owner_name_key(value) for value in values.dropna().unique()}).fillna("")
    values = df_rows["officer_title"]
    df_rows["officer_title_clean"] = values.map({value: clean_officer_title(value) for value in values.dropna().unique()}).fillna("")
    df_rows["role_source"] = "filed"
    df_rows["role_as_of"] = to_day(df_rows["filing_date"])
    df_rows["transaction_day"] = to_day(df_rows.get("transaction_date", pd.Series(pd.NaT, index=df_rows.index)))
    df_rows["direct_indirect"] = df_rows["direct_indirect"].fillna("").astype(str).str.upper().str.strip()
    for column in ("shares", "price_per_share", "value_usd", "shares_owned_after"):
        values = pd.to_numeric(df_rows.get(column, pd.Series(np.nan, index=df_rows.index)), errors="coerce")
        df_rows[column + "_raw"] = df_rows.get(column, values)
        df_rows[column] = values.where(np.isfinite(values))
    for column in ("is_director", "is_officer", "is_ten_pct_owner", "is_10b5_1"):
        values = cast(pd.Series, df_rows.get(column, pd.Series(np.nan, index=df_rows.index))).replace({"Y": 1, "N": 0, "true": 1, "false": 0})
        df_rows[column] = pd.to_numeric(values, errors="coerce").where(lambda x: x.isin([0, 1]))
    titles = df_rows["officer_title_clean"]
    df_rows["role"] = titles.map({title: officer_role(title) for title in titles.unique()})
    df_rows = _fill_prior_roles(df_rows, role_evidence)
    df_rows["is_management"] = df_rows["role"].isin(["CEO", "CFO", "COO"])
    df_rows["is_management"] &= ~df_rows["is_officer"].eq(0)
    return df_rows


def _fill_prior_roles(df_rows: pd.DataFrame, role_evidence: pd.DataFrame | None) -> pd.DataFrame:
    """Fill only blank current officer titles; later/conflicting evidence never overwrites a row."""
    keys = ["ticker", "issuer_cik", "owner_cik"]
    df_prior = df_rows.loc[df_rows["owner_cik"].ne(""), [*keys, "role_as_of", "officer_title_clean", "role"]].copy()
    df_prior["role_source"] = "prior_form4"
    df_prior = df_prior[df_prior["officer_title_clean"].ne("") & df_prior["role_as_of"].notna()]
    # A conflicting or non-management title is a history barrier, not a row to skip.
    grouped = df_prior.groupby([*keys, "role_as_of"], dropna=False)
    conflict = grouped["officer_title_clean"].transform("nunique").gt(1)
    df_prior.loc[conflict, ["role", "officer_title_clean"]] = [OTHER_OFFICER, "conflicting titles"]
    df_prior = df_prior.drop_duplicates([*keys, "role_as_of"])
    if role_evidence is not None and not role_evidence.empty:
        df_proxy = role_evidence.copy()
        for column in ("issuer_cik", "owner_cik"):
            df_proxy[column] = df_proxy[column].map(pad_cik).fillna("")
        df_prior = pd.concat([df_prior, df_proxy[df_prior.columns]], ignore_index=True)
    eligible = df_rows["officer_title_clean"].eq("") & df_rows["is_officer"].eq(1) & df_rows["owner_cik"].ne("") & df_rows["role_as_of"].notna()
    if not eligible.any() or df_prior.empty:
        return df_rows
    df_prior = df_prior.dropna(subset=["role_as_of"])
    conflict = df_prior.groupby([*keys, "role_as_of"])["role"].transform("nunique").gt(1)
    df_prior.loc[conflict, ["role", "officer_title_clean"]] = [OTHER_OFFICER, "conflicting roles"]
    df_prior = df_prior.sort_values("role_as_of").drop_duplicates([*keys, "role_as_of"], keep="first")
    df_match = pd.merge_asof(
        df_rows.loc[eligible, [*keys, "role_as_of"]].reset_index().sort_values("role_as_of"),
        df_prior.rename(columns={"role_as_of": "evidence_day"}).sort_values("evidence_day"),
        left_on="role_as_of",
        right_on="evidence_day",
        by=keys,
        allow_exact_matches=False,
        tolerance=pd.Timedelta(days=365),
    )
    supported = df_match["role"].isin(["CEO", "CFO", "COO"])
    df_match = df_match[supported].set_index("index")
    for column in ("role", "officer_title_clean", "role_source"):
        df_rows.loc[df_match.index, column] = df_match[column]
    df_rows.loc[df_match.index, "role_as_of"] = df_match["evidence_day"]
    return df_rows


def _source_assessments() -> tuple[dict[str, object], ...]:
    """Accession-exact source evidence, applied only from its public disclosure day."""
    return (
        {
            "accession_number": "0001127602-15-015564",
            "public_from": "2015-04-29",
            "state": "uncertain_price",
            "match": {"transaction_day": pd.Timestamp("2015-04-29"), "code": "S", "shares_n": 2475, "security_title": "Class A Common Stock"},
            "evidence": "https://www.sec.gov/Archives/edgar/data/1288776/000112760215015564/xslF345X03/form4.xml",
        },
        {
            "accession_number": "0000947871-17-000614",
            "public_from": "2017-08-04",
            "state": "corrected_price",
            "price_per_share": 13.14,
            "evidence": "https://ir.amd.com/financial-information/sec-filings/content/0000947871-17-000613/ss54101_sc13da.htm",
        },
        {
            "accession_number": "0001584209-16-000008",
            "public_from": "2016-10-19",
            "state": "uncertain",
            "evidence": "https://www.investing.com/news/stock-market-news/apparent-fake-filing-says-artist-bought-%2488-billion-of-bank-of-america-stock-433448",
        },
        {
            "accession_number": "0001584209-16-000008",
            "public_from": "2017-03-22",
            "state": "invalidated",
            "evidence": "https://www.treasuryandrisk.com/2017/03/22/just-disregard-that-sec-filing/",
        },
    )


def _apply_assessments(df_rows: pd.DataFrame, assessments: tuple[dict[str, object], ...]) -> pd.DataFrame:
    """Split affected visibility intervals at dated evidence, never rewriting earlier information."""
    df_rows["quality_uncertain"] = False
    df_rows["quality_evidence"] = ""
    if "accession_number" not in df_rows:
        return df_rows
    for evidence in sorted(assessments, key=lambda x: str(x["public_from"])):
        date = pd.Timestamp(cast(str, evidence["public_from"]))
        affected = df_rows["accession_number"].eq(cast(str, evidence["accession_number"])) & (
            df_rows["visible_until"].isna() | df_rows["visible_until"].gt(date)
        )
        for column, value in cast(dict[str, object], evidence.get("match", {})).items():
            affected &= df_rows[column].eq(cast(str | float | pd.Timestamp, value))
        if not affected.any():
            continue
        df_new = df_rows.loc[affected].copy()
        # Pandas accepts datetime clip bounds; its numeric-only stub does not.
        starts = df_new["visible_from"].clip(lower=cast(Any, date))
        before = affected & df_rows["visible_from"].lt(date)
        df_rows.loc[before, "visible_until"] = date
        df_rows = df_rows.loc[~(affected & ~before)].copy()
        if evidence["state"] == "invalidated":
            continue
        df_new["visible_from"] = starts
        df_new["quality_evidence"] = str(evidence["evidence"])
        if evidence["state"] == "uncertain":
            df_new["quality_uncertain"] = True
        elif evidence["state"] == "uncertain_price":
            df_new[["price_clean", "value"]] = np.nan
        elif evidence["state"] == "corrected_price":
            df_new["price_clean"] = float(cast(str | float, evidence["price_per_share"]))
            df_new["value"] = df_new["shares_n"] * df_new["price_clean"]
            df_new["price_repaired"] = True
            df_new["price_estimated"] = False
        df_rows = pd.concat([df_rows, df_new], ignore_index=True)
    return df_rows.reset_index(drop=True)


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


def clean_transactions(
    insider: pd.DataFrame,
    *,
    as_traded_quotes: pd.DataFrame | None = None,
    role_evidence: pd.DataFrame | None = None,
    share_basis: ShareBasis | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return priced common-stock P/S versions and diagnostics for unsupported evidence."""
    need = {"ticker", "filing_date", "transaction_code", "shares"}
    diag: dict = {"input_rows": 0 if insider is None else len(insider)}
    if insider is None or insider.empty or not need.issubset(insider.columns):
        return pd.DataFrame(), diag

    # Canonicalize identities, titles and amounts before selecting common-stock P/S.
    df_trades = _canonical_inputs(insider, role_evidence)
    df_trades, scope_counts = _scope_transactions(df_trades)
    diag.update(scope_counts)

    # Resolve public versions before separating unsupported quantities/classes.
    df_trades, versions = versioned_records(df_trades, df_trades["price_per_share"])
    diag.update(versions)
    unsupported = ~df_trades["shares_n"].gt(0) | df_trades["scope_unknown"]
    diag["unsupported_events"] = df_trades.loc[unsupported].assign(quality_uncertain=True).copy()
    df_trades = df_trades[~unsupported]

    # Apply quote checks and dated source evidence, retaining unpriced share claims.
    df_trades, price_counts = _clean_trade_prices(df_trades, as_traded_quotes, share_basis, PRICE_TOLERANCE, 7, None)
    diag.update(price_counts)

    # Report the same economic and filing-version counts after cleaning.
    _log_transaction_diagnostics(diag)
    return df_trades, diag


def _scope_transactions(df_trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select common-stock P/S while retaining unknown scope and non-P/S flow evidence."""
    diag: dict = {}
    df_trades["code"] = df_trades["transaction_code"].astype(str).str.upper().str.strip()
    df_trades["day"] = to_day(df_trades["filing_date"])
    df_trades["shares_n"] = df_trades["shares"]

    # Capture source coverage and other account flows before the P/S scope cut.
    diag["source_first"] = df_trades.loc[df_trades["ticker"].ne("") & df_trades["day"].notna()].groupby("ticker")["day"].min()
    diag["other_flows"] = (
        df_trades.loc[~df_trades["code"].isin(OPEN_MARKET_CODES), ["ticker", "owner_cik", "transaction_day", "day", "code", "security_type"]].copy()
        if "security_type" in df_trades
        else pd.DataFrame()
    )
    # Computed before the scope cut, which removes the derivative `M` leg of the package.
    df_trades["in_exercise_package"] = _exercise_packages(df_trades)

    df_trades["scope_unknown"] = False
    if "security_type" in df_trades.columns:
        security_type = df_trades["security_type"].fillna("").astype(str).str.lower().str.strip()
        unknown_type = ~security_type.isin(["nonderiv", "deriv"])
        df_trades["scope_unknown"] |= unknown_type
        df_trades = df_trades[security_type.eq("nonderiv") | unknown_type]

    if "security_title" in df_trades.columns:
        unknown_title = df_trades["security_title"].fillna("").astype(str).str.strip().eq("")
        df_trades["scope_unknown"] |= unknown_title
        df_trades = df_trades[common_stock_mask(df_trades["security_title"]) | unknown_title]

    df_trades = df_trades[df_trades["code"].isin(OPEN_MARKET_CODES) & df_trades["day"].notna() & (df_trades["ticker"] != "")]
    diag["scoped_rows"] = int((df_trades["shares_n"].gt(0) & ~df_trades["scope_unknown"]).sum())
    return df_trades, diag


def _clean_trade_prices(
    df_trades: pd.DataFrame,
    as_traded_quotes: pd.DataFrame | None,
    share_basis: ShareBasis | None,
    price_tolerance: float,
    max_price_age_days: int,
    assessments: tuple[dict[str, object], ...] | None,
) -> tuple[pd.DataFrame, dict]:
    """Check execution prices against quotes and separate priced, unpriced and uncertain records."""
    diag: dict = {}
    filed_price = df_trades["price_per_share"]

    # Match only supported transaction-date/class quotes in the declared share units.
    ref = pd.Series(np.nan, index=df_trades.index, dtype=float)
    if as_traded_quotes is not None and not as_traded_quotes.empty:
        quotes = _period_close(as_traded_quotes, df_trades["transaction_day"], max_price_age_days)
        lookup = pd.MultiIndex.from_arrays([df_trades["transaction_day"], df_trades["ticker"]], names=["period", "ticker"])
        ref[:] = quotes.reindex(lookup).to_numpy()
        # An explicit non-primary class needs its own quote; the canonical ticker cannot certify it.
        ref = ref.where(security_class(df_trades["security_title"]).eq("COMMON"))
        if share_basis is not None:
            units = share_basis.factor(df_trades["ticker"], df_trades["transaction_day"], "units")
            declared = share_basis.declared_units(df_trades["ticker"], df_trades["transaction_day"], df_trades["security_title"])
            ref *= declared / units

    # Repair gross corruption and apply dated assessments in their original order.
    ratio = filed_price / ref.where(ref.gt(0))
    bad = ratio.gt(price_tolerance)
    low = ratio.lt(1.0 / price_tolerance)
    df_trades["price_check_ratio"] = ratio
    df_trades["price_outside_band"] = ratio.notna() & ~ratio.between(0.90, 1.10)
    df_trades["price_clean"] = filed_price.where(~bad, ref)
    df_trades["price_repaired"] = bad
    df_trades["price_estimated"] = bad
    raw_value = pd.to_numeric(df_trades.loc[df_trades["price_clean"].gt(0), "value_usd"], errors="coerce")
    df_trades["value"] = df_trades["shares_n"] * df_trades["price_clean"]
    df_trades = _apply_assessments(df_trades, _source_assessments() if assessments is None else assessments)

    # Preserve unpriced share evidence and the existing diagnostic denominators.
    priced = df_trades["price_clean"].gt(0)
    diag["dropped_unpriced"] = int((~priced).sum())
    total_shares = df_trades["shares_n"].sum()
    diag["dropped_unpriced_shares"] = float(df_trades.loc[~priced, "shares_n"].sum()) / total_shares if total_shares > 0 else 0.0
    diag["unpriced_events"] = df_trades.loc[~priced].copy()
    df_trades = df_trades[priced].copy()
    diag["uncertain_events"] = pd.concat([df_trades, diag["unpriced_events"]], ignore_index=True).loc[lambda rows: rows["quality_uncertain"]].copy()

    diag["repaired_rows"] = int(df_trades["price_repaired"].sum())
    diag["underpriced_rows"] = int(low.sum())
    diag["value_before"] = float(raw_value.sum())
    diag["value_after"] = float(df_trades["value"].sum())
    diag["no_consensus_rows"] = int(ratio.isna().sum())

    diag["roles_imputed"] = int(df_trades["role_source"].ne("filed").sum())
    diag["outside_price_band"] = int(df_trades["price_outside_band"].sum())
    diag["uncertain_rows"] = len(diag["uncertain_events"])
    return df_trades, diag


def _log_transaction_diagnostics(diag: dict) -> None:
    """Report cleaned value totals, excluded prices and public filing-version counts."""
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
    if df_rows.empty or "accession_number" not in df_rows.columns:
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
    identical = df_rows.index.isin(df_plan.index[df_plan["identical"]]) if not df_plan.empty else np.zeros(len(df_rows), dtype=bool)
    df_rows = df_rows[~(drop_copy.to_numpy() | identical)].copy()
    if not df_owners.empty:
        # Co-reporters learned from later copies change metadata prospectively, not transaction size.
        df_copies = df_rows.loc[df_owners.index].copy().reset_index(drop=True)
        df_copies["visible_from"] = df_owners["visible_from"].to_numpy()
        stop = df_owners["visible_until"].reset_index(drop=True)
        df_copies["visible_until"] = pd.concat([df_copies["visible_until"], stop], axis=1).min(axis=1)
        for column in ("owner_ciks", "n_reporting_owners"):
            df_copies[column] = df_owners[column].to_numpy()
        df_rows = pd.concat([df_rows.drop(index=df_owners.index.unique()), df_copies], ignore_index=True)
    return df_rows, counts


def _copy_key(df: pd.DataFrame, pps: pd.Series) -> pd.DataFrame:
    """Per row: accession, filing day, the `COPY_KEY` fields (`round_half_up`) and the `CELL_KEY` fields."""
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
            "shares": round_half_up(df["shares_n"]),
            "price": round_half_up(pps.reindex(df.index)),
            "owned_after": round_half_up(owned),
        },
        index=df.index,
    )


def round_half_up(values: pd.Series, decimals: int = COPY_KEY_DECIMALS) -> pd.Series:
    """`values` rounded half away from zero at `decimals`, as the SEC zip rounds the filed decimal
    text; a value within `_HALF_ULPS` below a half counts as that half. NaN stays NaN."""
    number = pd.to_numeric(values, errors="coerce").astype("float64")
    scale = 10.0**decimals
    scaled = number.abs() * scale
    whole = scaled // 1.0
    up = (scaled - whole) >= 0.5 - _HALF_ULPS * np.spacing(scaled.fillna(0.0))
    magnitude = (whole + up) / scale
    return magnitude.where(number.ge(0.0), -magnitude)


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
    dropped. A NaN key field never matches. Returns the drop mask, dated metadata versions
    of already-public owners (`owner_ciks`, `n_reporting_owners`) and the counts.
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

    df_members = (
        df_group[["group", "accession_number"]]
        .drop_duplicates()
        .merge(_owner_pairs(df.loc[df_group.index]), on="accession_number")
        .assign(owner=lambda d: d["owner"].str.zfill(10))
        .merge(df_group[["accession_number", "day"]].drop_duplicates(), on="accession_number")
    )
    records = []
    for group_id, members in df_members.groupby("group", sort=False):
        owners: set[str] = set()
        versions = []
        for day, public in members.groupby("day", sort=True):
            new_owners = set(public["owner"]) - owners
            if new_owners:
                owners.update(new_owners)
                versions.append((day, ",".join(sorted(owners)), len(owners)))
        kept_rows = df_group.index[df_group["group"].eq(group_id) & ~later]
        for i, (day, owner_ciks, number) in enumerate(versions):
            stop = versions[i + 1][0] if i + 1 < len(versions) else pd.NaT
            records.extend(
                {"row": row, "visible_from": day, "visible_until": stop, "owner_ciks": owner_ciks, "n_reporting_owners": number} for row in kept_rows
            )
    df_owners = pd.DataFrame(records).set_index("row")
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
