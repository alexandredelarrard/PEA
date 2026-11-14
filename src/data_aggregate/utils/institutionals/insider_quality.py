"""Canonical insider identities, common-stock scope, public filing versions and quote quality.

Corrections retain original disclosure anchors and half-open visibility intervals.
Unsupported quantities and execution prices remain separate evidence for downstream masks.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.quotes import lookup_asof
from src.utils.names import clean_person_name
from src.utils.string import pad_cik
from src.utils.title_vocabulary import CHAIR, COUNSEL, GENERAL, SENIOR, VICE
from src.utils.title_vocabulary import CHIEF as CHIEF_WORD

_log = logging.getLogger(__name__)

#: Open-market purchase/sale codes; every other code is non-discretionary and never netted.
OPEN_MARKET_CODES: tuple[str, ...] = ("P", "S")

#: Direction on the account's holdings of a non-open-market row, from its transaction code (+1 acquired, -1 disposed).
#: Codes that run either way (G gift, J, K, L, V, W, Z, I, U, O, E, H) are absent: such a row has no declared change.
FLOW_SIGN: dict[str, int] = {"A": 1, "M": 1, "C": 1, "X": 1, "F": -1, "D": -1}

#: Codes whose `price_per_share` is a market price and may back the consensus (`F` is withholding at the close).
MARKET_PRICED_CODES: tuple[str, ...] = ("P", "S", "F")

#: Common/ordinary-stock security titles, including common misspellings; preferred and 401(k) units fall through.
COMMON_STOCK_RE = re.compile(
    r"COMMON|COMON|COMMOM|COMM|ORDINARY|REGISTER|CLASS\s+[A-Z]\b|SHARES\s+OF\s+BENEFICIAL|DEPOSITARY|" r"^\s*(?:COM|STOCK|SHARES)\s*$", re.I
)

#: Filed price / as-traded close beyond which the filed price is replaced by the close: above `PRICE_BAND_HIGH` or below
#: `PRICE_BAND_LOW` (a ratio exactly on a bound is kept).
PRICE_BAND_HIGH: float = 2.0
PRICE_BAND_LOW: float = 0.5

#: Calendar days a quote may be older than the trade's reference day.
MAX_QUOTE_AGE_DAYS: int = 7

#: Share-class token in a security title, so the consensus is built within one class.
_CLASS_RE = re.compile(r"\bCLASS\s+([A-Z])\b|\bSERIES\s+([A-Z])\b", re.I)

#: Trailing publication-time window behind the consensus median.
CONSENSUS_WINDOW: str = "31D"

#: A transaction above this share of the shares outstanding on its day is removed, and its direct account is left out of the pooled ratio.
MAX_PCT_SHARES_OUTSTANDING: float = 0.25

#: Free-text `officer_title` -> role, first match wins (a "Chairman, President & CEO" is a CEO).
CHIEF = rf"(?:{CHIEF_WORD}|CHF)"
OFFICER = r"OFF(?:ICER|CR|R)"
OPERATIONS = r"OP(?:ER(?:ATION(?:S)?|ATING)?)?"
ROLE_CEO, ROLE_CFO, ROLE_COO = "CEO", "CFO", "COO"
ROLE_OTHER_C_SUITE, ROLE_PRESIDENT, ROLE_CHAIR, ROLE_VP_LEVEL = "OTHER_C_SUITE", "PRESIDENT", "CHAIR", "VP_LEVEL"
#: Order matters: the first role whose pattern matches wins. A chief title beats PRESIDENT/CHAIR, and PRESIDENT beats
#: CHAIR, so "Chairman, President & CEO" is CEO and "Vice Chairman and President" is PRESIDENT. "Vice President",
#: "Group President" and "President, <unit>" never reach PRESIDENT.
ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        ROLE_CEO,
        rf"\b{CHIEF}\s+EXEC(?:UTIVE)?\s+{OFFICER}\b" rf"|\bPRINCIPAL\s+EXEC(?:UTIVE)?\s+{OFFICER}\b" rf"|\bC\s*E\s*O\b",
    ),
    (ROLE_CFO, rf"\b{CHIEF}\s+FIN(?:ANCIAL)?\s+{OFFICER}\b" rf"|\bPRINCIPAL\s+FIN(?:ANCIAL)?\s+{OFFICER}\b" rf"|\bC\s*F\s*O\b"),
    (ROLE_COO, rf"\b{CHIEF}\s+{OPERATIONS}\s+{OFFICER}\b" rf"|\bC\s*O\s*O\b"),
    # Any other "Chief ... Officer" (technology, marketing, accounting, legal, product, information, people, strategy ...).
    (
        ROLE_OTHER_C_SUITE,
        rf"\b{CHIEF}\s+[A-Z][A-Z&/ \-]*?\s*{OFFICER}\b" rf"|\bC\s*[A-Z]\s*O\b(?!\w)" rf"|\bC\s*H\s*R\s*O\b" rf"|\bPRINCIPAL\s+[A-Z]+\s+{OFFICER}\b",
    ),
    (ROLE_PRESIDENT, r"(?<!VICE )(?<!VICE-)(?<!GROUP )(?<!V )(?<!SR )(?<!SENIOR )\bPRES(?:IDENT)?\b"),
    (ROLE_CHAIR, r"\bCHAIR(?:MAN|WOMAN|PERSON)?\b"),
    (ROLE_VP_LEVEL, r"\b(?:SE|S|E)?\s*V\s*P\b|\bTREASURER\b|\bCONTROLLER\b|\bVICE[\s-]+PRES(?:IDENT)?\b"),
)
OTHER_OFFICER = "other_named_officer"

#: Insider populations, disjoint and ordered by precedence (a lower code wins a row). `POP_OTHER` (including an officer
#: whose title says nothing) appears only in all-insider features.
POP_C_SUITE, POP_MANAGEMENT, POP_TEN_PCT, POP_OTHER = 1, 2, 3, 4

#: Cleaned `officer_title` -> population, first match wins. A director flag or a blank-title 10%-owner is resolved from the
#: flags in `_assign_population`, not here. Former/retired titles never match.
TITLE_POPULATIONS: tuple[tuple[int, str], ...] = (
    (
        POP_C_SUITE,
        rf"\b{CHIEF_WORD}\b|\bCHF\b|\bC\s*[A-Z]\s*O\b|\bPRINCIPAL\s+(?:EXEC|FIN|ACC|OPER)|\bFOUNDER\b"
        rf"|\b(?:E|EX|SE)\s*V\s*P\b|\bEXEC(?:UTIVE)?\b|\b{CHAIR}|\bCHARIMAN\b|\bCOB\b"
        r"|(?<!VICE )(?<!VICE-)(?<!V )(?<!SR )(?<!SENIOR )(?<!ASSISTANT )(?<!ASST )\bPRES(?:IDENT)?\b",
    ),
    (
        POP_MANAGEMENT,
        rf"\b[A-Z]{{0,3}}\s*V\s*P\b|\b{VICE}\b|\b{SENIOR}\b|\bSR\b|\bPRES(?:IDENT)?\b|\b{GENERAL}\s+(?:{COUNSEL}|MANAGER)\b|\bGEN\s+{COUNSEL}\b|\b{COUNSEL}\b"
        r"|\bGC\b|\bTREASURER\b|\b(?:CONTROLLER|COMPTROLLER)\b|\bSECRETARY\b|\bSTRATEG|\bVENTURES\b|\bMANAGING\b|\bDIRECTOR\b|\bOFFICER\b|\bHEAD\b|\bPARTNER\b",
    ),
)
_TITLE_POPULATIONS_COMPILED = tuple((code, re.compile(pattern, re.I)) for code, pattern in TITLE_POPULATIONS)
_FORMER_TITLE = re.compile(r"\b(?:FORMER|FORMERLY|RETIRED|EMERITUS)\b", re.I)

#: Roles whose title may be followed by a named functional area (a VP of Sales is still a VP of the issuer).
_NO_TAIL_CHECK = frozenset({ROLE_VP_LEVEL})

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
    if re.search(r"\b(?:FORMER|RETIRED|INCOMING|DESIGNATE|ELECT|SUBSIDIAR(?:Y|IES)|DIVISION(?:AL)?|REGIONAL|SEGMENT|BUSINESS\s+UNIT)\b", s, re.I):
        return OTHER_OFFICER
    for name, rx in _ROLE_COMPILED:
        match = rx.search(s)
        if match is None:
            continue
        # A named entity after the role is a unit/affiliate, not another issuer title.
        tail = s[match.end() :].strip(" ,;&/:-().")
        companion = r"^(?:AND\s+)?(?:CHIEF\b|C\s*[A-Z]\s*O\b|GC\b|GENERAL\s+COUNSEL\b|PRES(?:IDENT)?\b|(?:SR|SENIOR)\s+(?:VP|VICE|EXEC)\b|VP\b|VICE\b|[ES]VP\b|MD\b|MANAGING\s+DIRECTOR\b|INTERIM\b|ACTING\b|EXEC(?:UTIVE)?\b|CHAIR\w*\b|OF\s+(?:THE\s+)?(?:BOARD|DIRECTORS|EXEC)\w*\b|DIR(?:ECTOR)?\b|TREASURER\b|CONTROLLER\b|SECRETARY\b|PRINCIPAL\b|(?:CO[- ]?)?FOUNDER\b)"
        if name in _NO_TAIL_CHECK or not tail or re.match(companion, tail, re.I):
            return name
        return OTHER_OFFICER
    return OTHER_OFFICER


def title_population(title: object) -> int:
    """Population code a cleaned `officer_title` alone resolves to, else `POP_OTHER`."""
    clean = clean_officer_title(title)
    if not clean or _FORMER_TITLE.search(clean):
        return POP_OTHER
    return next((code for code, rx in _TITLE_POPULATIONS_COMPILED if rx.search(clean)), POP_OTHER)


def _assign_population(df_rows: pd.DataFrame) -> pd.Series:
    """Disjoint population per row: C-suite title, then director or management title, then 10% owner; an officer without a usable title is OTHER."""
    from_title = _clean_map_dict(df_rows["officer_title_clean"], clean_func=title_population).astype(int)
    unresolved = df_rows["is_officer"].eq(1) & (df_rows["role_unknown"] | df_rows["officer_title_clean"].map(role_needs_evidence).astype(bool))
    population = pd.Series(POP_OTHER, index=df_rows.index)
    population = population.mask(df_rows["is_ten_pct_owner"].eq(1), POP_TEN_PCT)
    population = population.mask(df_rows["is_director"].eq(1) | from_title.eq(POP_MANAGEMENT), POP_MANAGEMENT)
    population = population.mask(unresolved & from_title.eq(POP_OTHER), POP_OTHER)
    return population.mask(from_title.eq(POP_C_SUITE), POP_C_SUITE)


def clean_officer_title(value: object) -> str:
    """Normalize title punctuation and explicit abbreviations, preserving role modifiers."""
    if not isinstance(value, str):
        return ""
    title = value.replace(".", " ").strip()
    title = re.sub(r",\s*", ", ", title)
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"\bCHIEF\s+EXEC\b(?=\s+OFF(?:ICER|CR|R)\b)", "Chief Executive", title, flags=re.I)
    title = re.sub(r"\bCHIEF\s+EXEC\b(?!\s*UTIVE)", "Chief Executive Officer", title, flags=re.I)
    return re.sub(r"\s+", " ", title)


def role_needs_evidence(value: object) -> bool:
    """A blank or generic reference contains no job claim to contradict dated evidence."""
    return re.fullmatch(r"(?:SEE\s+(?:REMARKS?|FOOTNOTES?))?", clean_officer_title(value), re.I) is not None


def unresolved_role(value: object) -> bool:
    """Separate missing/ambiguous jobs from explicit jobs outside issuer CEO/CFO/COO."""
    title = clean_officer_title(value)
    if not title or re.search(
        r"\b(?:REMARKS?|FOOTNOTES?|UNKNOWN|UNSPECIFIED|CONFLICT\w*|AMBIGUOUS|INCOMING|DESIGNATE|ELECT|EFFECTIVE)\b", title, re.I
    ):
        return True
    if officer_role(title) != OTHER_OFFICER:
        return False
    if re.search(r"\b(?:FORMER|RETIRED|SUBSIDIAR(?:Y|IES)|DIVISION(?:AL)?|REGIONAL|SEGMENT|BUSINESS\s+UNIT)\b", title, re.I):
        return False
    # A recognized chief title rejected by the role parser has a unit/affiliate tail.
    if any(pattern.search(title) for _, pattern in _ROLE_COMPILED):
        return False
    return re.search(r"\b(?:PRESIDENT|PRES|CHAIR\w*|DIRECTOR|TREASURER|SECRETARY|CONTROLLER|COUNSEL|VICE|VP|SVP|EVP)\b", title, re.I) is None


def owner_name_key(value: object) -> str:
    """Strict full-name comparison tokens; actual initials and surname particles survive."""
    name = clean_person_name(value)
    if not name:
        return ""
    tokens = re.findall(r"[^\W\d_]+", name.casefold())
    return " ".join(sorted(token for token in tokens if token not in {"mr", "mrs", "ms", "dr", "jr", "sr", "ii", "iii", "iv"}))


def insider_days(values: pd.Series) -> pd.Series:
    """Reject implausible/unsafe source dates before converting pandas' retained microseconds to nanoseconds."""
    days = pd.to_datetime(values, errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    return days.where(days.between(pd.Timestamp("1900-01-01"), pd.Timestamp.max)).dt.normalize().dt.as_unit("ns")


def publication_day(days: pd.Series, lag_days: int, sessions: pd.DatetimeIndex) -> pd.Series:
    """`days` + `lag_days` rolled forward to the first session on or after it; outside the session span the shifted day itself, NaT stays NaT."""
    shifted = days + pd.Timedelta(days=lag_days)
    position = sessions.searchsorted(pd.DatetimeIndex(shifted), side="left")
    rolled = pd.Series(sessions.take(np.minimum(position, len(sessions) - 1)).astype(shifted.dtype), index=days.index)
    return rolled.where(shifted.notna() & (position < len(sessions)) & (shifted >= sessions[0]), shifted)


def normalize_insider_dates(df_rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize an owned source frame; execution evidence stays separate from filing-backed quote references."""
    for column in ("filing_date", "transaction_date", "original_submission_date"):
        if column not in df_rows:
            df_rows[column] = pd.Series(pd.NaT, index=df_rows.index, dtype="datetime64[ns]")
        if column + "_raw" not in df_rows:
            df_rows[column + "_raw"] = df_rows[column]
        df_rows[column] = insider_days(df_rows[column])
        if column != "filing_date":
            df_rows[column] = df_rows[column].where(df_rows[column].le(df_rows["filing_date"]))
    df_rows["transaction_day"] = df_rows["transaction_date"]
    df_rows["reference_day"] = df_rows["transaction_day"].fillna(df_rows["filing_date"])
    df_rows["reference_day_estimated"] = df_rows["transaction_day"].isna() & df_rows["reference_day"].notna()

    return df_rows


def _clean_map_dict(df_serie: pd.Series, clean_func: Callable) -> pd.Series:
    dict_values = {v: clean_func(v) for v in df_serie.dropna().unique()}
    return df_serie.map(dict_values).fillna("")


def _canonical_inputs(df_insider: pd.DataFrame, df_roles: pd.DataFrame | None) -> pd.DataFrame:
    """Clean shared downstream fields and enrich blank titles from prior public identity evidence."""

    df_rows = df_insider.copy().reset_index(drop=True)
    df_rows["source_row"] = np.arange(len(df_rows))
    for column in ("owner_cik", "issuer_cik"):
        df_rows[column + "_raw"] = df_rows[column]
        df_rows[column] = _clean_map_dict(df_rows[column], clean_func=pad_cik)

    df_rows["role_source"] = "filed"
    df_rows["owner_name"] = _clean_map_dict(df_rows["owner_name"], clean_func=pad_cik)
    df_rows["role_as_of"] = to_day(df_rows["filing_date"])
    df_rows["direct_indirect"] = df_rows["direct_indirect"].fillna("").astype(str).str.upper().str.strip()
    for column in ("shares", "price_per_share", "value_usd", "shares_owned_after"):
        values = pd.to_numeric(df_rows.get(column, pd.Series(np.nan, index=df_rows.index)), errors="coerce")
        df_rows[column + "_raw"] = df_rows.get(column, values)
        df_rows[column] = values.where(np.isfinite(values))

    for column in ("is_director", "is_officer", "is_ten_pct_owner", "is_10b5_1"):
        values = cast(pd.Series, df_rows.get(column, pd.Series(np.nan, index=df_rows.index))).replace({"Y": 1, "N": 0, "true": 1, "false": 0})
        df_rows[column] = pd.to_numeric(values, errors="coerce").where(lambda x: x.isin([0, 1]))

    df_rows["officer_title_clean"] = _clean_map_dict(df_rows["officer_title"], clean_func=clean_officer_title)
    df_rows["role"] = _clean_map_dict(df_rows["officer_title_clean"], clean_func=officer_role)
    df_rows["role_unknown"] = _clean_map_dict(df_rows["officer_title_clean"], clean_func=unresolved_role) & ~df_rows["is_officer"].eq(0)
    df_rows = _fill_prior_roles(df_rows, df_roles)
    df_rows["population"] = _assign_population(df_rows)
    return df_rows


def _fill_prior_roles(df_rows: pd.DataFrame, df_roles: pd.DataFrame | None) -> pd.DataFrame:
    """Resolve missing job claims from prior issuer/person proof; retain exclusion/uncertainty barriers."""

    keys = ["ticker", "issuer_cik", "owner_cik"]
    known_identity = df_rows["issuer_cik"].ne("") & df_rows["owner_cik"].ne("")

    # 1. Build base prior data from rows
    valid_titles = ~df_rows["officer_title_clean"].map(role_needs_evidence)
    df_prior = (
        df_rows.loc[known_identity & valid_titles & df_rows["role_as_of"].notna(), [*keys, "role_as_of", "officer_title_clean", "role"]]
        .assign(role_source="prior_form4")
        .copy()
    )

    # Resolve initial title conflicts
    conflict = df_prior.groupby([*keys, "role_as_of"], dropna=False)["officer_title_clean"].transform("nunique").gt(1)
    df_prior.loc[conflict, ["role", "officer_title_clean"]] = [OTHER_OFFICER, "conflicting titles"]
    df_prior = df_prior.drop_duplicates([*keys, "role_as_of"])

    # 2. Append proxy roles from df_roles if available - already cleaned from load_insider_role_evidence
    if df_roles is not None and not df_roles.empty:
        df_proxy = df_roles.copy()
        df_proxy = df_proxy[
            df_proxy["issuer_cik"].ne("") & df_proxy["owner_cik"].ne("") & df_proxy["officer_title_clean"].map(role_needs_evidence).eq(False)
        ]
        df_prior = pd.concat([df_prior, df_proxy[df_prior.columns]], ignore_index=True)

    # 3. Check eligibility
    eligible = df_rows["officer_title_clean"].map(role_needs_evidence) & df_rows["is_officer"].eq(1) & known_identity & df_rows["role_as_of"].notna()
    if not eligible.any() or df_prior.empty:
        return df_rows

    # 4. Finalize prior dataset processing
    df_prior = df_prior.dropna(subset=["role_as_of"])
    conflict = df_prior.groupby([*keys, "role_as_of"])["role"].transform("nunique").gt(1)
    df_prior.loc[conflict, ["role", "officer_title_clean"]] = [OTHER_OFFICER, "conflicting roles"]
    mapped = df_prior["officer_title_clean"].map(officer_role)
    df_prior["role_unknown"] = df_prior["officer_title_clean"].map(unresolved_role).astype(bool) | (
        df_prior["role"].eq(OTHER_OFFICER) & mapped.ne(OTHER_OFFICER)
    )
    df_prior = df_prior.sort_values("role_as_of").drop_duplicates([*keys, "role_as_of"], keep="first")

    # 5. Merge asof and update rows
    df_match = pd.merge_asof(
        df_rows.loc[eligible, [*keys, "role_as_of"]].reset_index().sort_values("role_as_of"),
        df_prior.rename(columns={"role_as_of": "evidence_day"}).sort_values("evidence_day"),
        left_on="role_as_of",
        right_on="evidence_day",
        by=keys,
        allow_exact_matches=False,
        tolerance=pd.Timedelta(days=365),
    )

    # Update unknown statuses
    classified = df_match[df_match["role_unknown"].eq(False)].set_index("index")
    df_rows.loc[classified.index, "role_unknown"] = False

    # Update supported top roles
    supported_match = df_match[df_match["role"].notna() & df_match["role"].ne(OTHER_OFFICER)].set_index("index")
    for col in ("role", "officer_title_clean", "role_source"):
        df_rows.loc[supported_match.index, col] = supported_match[col]
    df_rows.loc[supported_match.index, "role_as_of"] = supported_match["evidence_day"]

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


def clean_transactions(
    df_insider: pd.DataFrame,
    *,
    df_valuation_close: pd.DataFrame | None = None,
    df_roles: pd.DataFrame | None = None,
    df_shares_outstanding: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return priced common-stock P/S versions and diagnostics for unsupported evidence.

    `df_valuation_close` is the saved point-in-time close and `df_shares_outstanding` the saved point-in-time share count.
    """

    need = {"ticker", "filing_date", "transaction_code", "shares"}
    diag: dict = {"input_rows": 0 if df_insider is None else len(df_insider)}
    if df_insider is None or df_insider.empty or not need.issubset(df_insider.columns):
        return pd.DataFrame(), diag

    # Canonicalize identities, titles and amounts before selecting common-stock P/S.
    df_trades = _canonical_inputs(df_insider, df_roles)
    diag["unavailable_filing_dates"] = int(df_trades["filing_date"].isna().sum())
    for column in ("transaction_date", "original_submission_date"):
        diag["invalid_" + column + "s"] = int((df_trades[column + "_raw"].notna() & df_trades[column].isna()).sum())
    diag["date_fallback_rows"] = int(df_trades["reference_day_estimated"].sum())

    # clean transactions
    df_trades, scope_counts = _scope_transactions(df_trades)
    diag.update(scope_counts)

    # Resolve public versions before separating unsupported quantities/classes.
    df_trades, versions = versioned_records(df_trades, df_trades["price_per_share"])
    diag.update(versions)
    unsupported = ~df_trades["shares_n"].gt(0) | df_trades["scope_unknown"]
    diag["unsupported_events"] = df_trades.loc[unsupported].assign(quality_uncertain=True).copy()
    df_trades = df_trades[~unsupported]

    # Remove trades above the share-count cap; their owner accounts stay out of the pooled ratio.
    df_trades, diag["oversized_events"] = _split_oversized(df_trades, df_shares_outstanding)

    # Apply quote checks and dated source evidence, retaining unpriced share claims.
    df_trades, price_counts = _clean_trade_prices(df_trades, df_valuation_close, MAX_QUOTE_AGE_DAYS, None)
    diag.update(price_counts)

    # Report the same economic and filing-version counts after cleaning.
    _log_transaction_diagnostics(diag)
    return df_trades, diag


def _scope_transactions(df_trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select common-stock P/S while retaining unknown scope and non-P/S flow evidence."""
    diag: dict = {}
    df_trades["code"] = df_trades["transaction_code"].str.upper().str.strip()
    df_trades["day"] = to_day(df_trades["filing_date"])
    df_trades["shares_n"] = df_trades["shares"]

    # Capture source coverage and other account flows before the P/S scope cut.
    diag["source_first"] = df_trades.loc[df_trades["day"].notna()].groupby("ticker")["day"].min()
    diag["other_flows"] = _other_flow_versions(df_trades)

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


def _other_flow_versions(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Retain non-P/S account metadata and its matching amendment-cell visibility."""

    df_flows = df_trades.loc[~df_trades["code"].isin(OPEN_MARKET_CODES)].copy()
    df_flows["visible_from"] = df_flows["day"]
    df_flows["visible_until"] = pd.NaT
    df_flows["version_known"] = False
    if df_flows.empty or "accession_number" not in df_trades:
        return df_flows

    df_flows["version_known"] = df_flows["document_type"].isin(["3", "4", "5"]) & df_flows["accession_number"].notnull()
    df_key = _copy_key(df_trades, df_trades["price_per_share"])
    links, _ = _link_amendments(df_trades, df_key, require_unique=True)
    if links.empty:
        return df_flows

    df_flows["version_known"] |= df_flows["accession_number"].isin(links.index)
    df_cells = df_key.loc[df_flows.index, ["accession_number", "day", *CELL_KEY]].copy()
    df_cells["transaction_day"] = _declared_action_cells(df_key, links).reindex(df_flows.index)
    df_cells[["owner_cik", "direct_indirect"]] = df_flows[["owner_cik", "direct_indirect"]]
    df_cells["klass"] = security_class(df_flows["security_title"])
    members = (
        df_flows.get("owner_ciks", pd.Series("", index=df_flows.index))
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.split(",")
        .explode()
        .str.lstrip("0")
    )
    df_cells["owner_members"] = members.sort_values().groupby(level=0).agg(_joined).reindex(df_flows.index)
    df_cells["n_reporting_owners"] = pd.to_numeric(df_flows.get("n_reporting_owners", pd.Series(np.nan, index=df_flows.index)), errors="coerce")
    df_cells["original"] = df_cells["accession_number"].map(links).fillna(df_cells["accession_number"])
    df_cells = df_cells.dropna(subset=["transaction_day"])
    if df_cells.empty:
        return df_flows
    keys = ["original", *CELL_KEY, "owner_cik", "direct_indirect", "klass", "owner_members", "n_reporting_owners"]
    df_boundaries = (
        df_cells.groupby([*keys, "accession_number"], dropna=False)
        .agg(day=("day", "min"), n_rows=("day", "size"))
        .reset_index()
        .sort_values([*keys, "day", "accession_number"])
    )
    versions = df_boundaries.groupby(keys, dropna=False)
    df_boundaries["visible_until"] = versions["day"].shift(-1).where(df_boundaries["n_rows"].eq(versions["n_rows"].shift(-1)))
    df_stops = df_cells.reset_index(names="source_index").merge(
        df_boundaries[[*keys, "accession_number", "visible_until"]], on=[*keys, "accession_number"], how="left"
    )
    df_flows["visible_until"] = df_stops.set_index("source_index")["visible_until"].reindex(df_flows.index)
    return df_flows


def _reference_quotes(df_trades: pd.DataFrame, quotes: pd.DataFrame | None, max_age_days: int) -> pd.Series:
    """Per trade, the as-traded close of its reference day, else the last one within `max_age_days`; NaN for a class-titled row.

    An explicit non-primary class needs its own quote; the canonical ticker cannot certify it.
    """
    reference = pd.Series(np.nan, index=df_trades.index, dtype=float)
    if quotes is None or quotes.empty:
        return reference
    df_long = (
        quotes.where(np.isfinite(quotes) & quotes.gt(0)).rename_axis(index="reference_day", columns="ticker").stack().rename("quote").reset_index()
    )
    df_long["reference_day"] = df_long["reference_day"].astype("datetime64[ns]")
    df_wanted = df_trades.loc[df_trades["reference_day"].notna(), ["reference_day", "ticker"]].astype({"reference_day": "datetime64[ns]"})
    df_found = pd.merge_asof(
        df_wanted.rename_axis("row").reset_index().sort_values("reference_day"),
        df_long.sort_values("reference_day"),
        on="reference_day",
        by="ticker",
        tolerance=pd.Timedelta(days=max_age_days),
    )
    reference.loc[df_found["row"].to_numpy()] = df_found["quote"].to_numpy()
    return reference.where(security_class(df_trades["security_title"]).eq("COMMON"))


def _clean_trade_prices(
    df_trades: pd.DataFrame,
    as_traded_quotes: pd.DataFrame | None,
    max_price_age_days: int,
    assessments: tuple[dict[str, object], ...] | None,
) -> tuple[pd.DataFrame, dict]:
    """Replace a filed price far from the as-traded close, price a missing one from it, and separate priced from unpriced records."""
    diag: dict = {}
    filed_price = df_trades["price_per_share"].where(df_trades["price_per_share"].gt(0))

    # A filing-backed reference can estimate a price, never execution-time quantities or capital.
    ref = _reference_quotes(df_trades, as_traded_quotes, max_price_age_days)
    ratio = filed_price / ref
    high, low = ratio.gt(PRICE_BAND_HIGH), ratio.lt(PRICE_BAND_LOW)
    replaced = high | low
    filled = filed_price.isna() & ref.notna()
    df_trades["price_clean"] = filed_price.where(~(replaced | filled), ref)
    df_trades["price_repaired"] = replaced
    df_trades["price_estimated"] = replaced | filled
    value_filed = float((df_trades["shares_n"] * filed_price).sum())
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
    diag["repaired_high"] = int(high.sum())
    diag["repaired_low"] = int(low.sum())
    diag["priced_from_quote"] = int(filled.sum())
    diag["value_before"] = value_filed
    diag["value_after"] = float(df_trades["value"].sum())
    diag["no_consensus_rows"] = int(ref.isna().sum())

    diag["roles_imputed"] = int(df_trades["role_source"].ne("filed").sum())
    diag["uncertain_rows"] = len(diag["uncertain_events"])
    return df_trades, diag


def _split_oversized(df_trades: pd.DataFrame, df_shares_outstanding: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trades above `MAX_PCT_SHARES_OUTSTANDING` of the shares outstanding on their reference day, split off from the rest.

    The split-off frame carries `pct_shares_outstanding`; with no share count nothing is removed.
    """
    if df_trades.empty or df_shares_outstanding is None or df_shares_outstanding.empty:
        return df_trades, df_trades.iloc[:0].copy()
    outstanding = lookup_asof(df_shares_outstanding.ffill(), df_trades["ticker"], df_trades["reference_day"])
    pct = df_trades["shares_n"] / outstanding.where(outstanding.gt(0))
    oversized = pct.gt(MAX_PCT_SHARES_OUTSTANDING)
    df_oversized = df_trades.loc[oversized].assign(pct_shares_outstanding=pct[oversized])
    if not df_oversized.empty:
        _log.warning(
            "insider: %s trade(s) above %.0f%% of shares outstanding removed with their direct accounts: %s",
            len(df_oversized),
            100 * MAX_PCT_SHARES_OUTSTANDING,
            ", ".join(f"{r.ticker} {r.day:%Y-%m-%d} {r.code} {r.pct_shares_outstanding:.1%}" for r in df_oversized.itertuples()),
        )
    return df_trades.loc[~oversized], df_oversized


def _log_transaction_diagnostics(diag: dict) -> None:
    """Report cleaned value totals, excluded prices and public filing-version counts."""
    _log.info(
        "insider dates: %s unavailable filing, %s invalid execution, %s invalid original submission; %s filing-backed references",
        diag["unavailable_filing_dates"],
        diag["invalid_transaction_dates"],
        diag["invalid_original_submission_dates"],
        diag["date_fallback_rows"],
    )
    _log.info(
        "insider: %s rows -> %s scoped, %s unpriced dropped, %s repaired (%s high, %s low), %s priced from the quote ($%.3fbn -> $%.3fbn)",
        diag["input_rows"],
        diag["scoped_rows"],
        diag["dropped_unpriced"],
        diag["repaired_rows"],
        diag["repaired_high"],
        diag["repaired_low"],
        diag["priced_from_quote"],
        diag["value_before"] / 1e9,
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
    if "accession_number" not in t.columns or "transaction_day" not in t.columns:
        return pd.Series(False, index=t.index)
    key = pd.MultiIndex.from_arrays([t["accession_number"].astype(str), t["transaction_day"]])
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
    unique_links, _ = _link_amendments(df_rows, df_key, require_unique=True)
    df_plan, cell_counts = _supersede_plan(df_key, links, unique_links)
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
    blank = pd.Series("", index=df.index)
    owned = pd.to_numeric(df["shares_owned_after"], errors="coerce") if "shares_owned_after" in df.columns else nan
    txn_day = df.get("transaction_day")
    if txn_day is None:
        txn_day = insider_days(df.get("transaction_date", pd.Series(pd.NaT, index=df.index)))
        txn_day = txn_day.where(txn_day.le(insider_days(df["day"])))
    security = df["security_type"].astype(str).str.lower() if "security_type" in df.columns else pd.Series("nonderiv", index=df.index)
    return pd.DataFrame(
        {
            "accession_number": df["accession_number"].astype(str),
            "day": df["day"],
            "ticker": df["ticker"],
            "transaction_day": txn_day,
            "declared_day": insider_days(df.get("transaction_date_raw", df.get("transaction_date", pd.Series(pd.NaT, index=df.index)))),
            "code": df["code"],
            "security_type": security,
            "shares": round_half_up(df["shares_n"]),
            "price": round_half_up(pps.reindex(df.index)),
            "owned_after": round_half_up(owned),
            "owner": df.get("owner_cik", blank),
            "account": df.get("direct_indirect", blank),
            "security_title": df.get("security_title", blank).fillna("").astype(str).str.upper().str.strip(),
            "row_sequence": pd.to_numeric(df.get("row_sequence", nan), errors="coerce"),
            "n_reporting_owners": pd.to_numeric(df.get("n_reporting_owners", nan), errors="coerce"),
            "source_shares": df["shares_n"],
            "source_price": pps.reindex(df.index),
            "source_owned": owned,
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


def _link_amendments(df: pd.DataFrame, df_key: pd.DataFrame, *, require_unique: bool = False) -> tuple[pd.Series, int]:
    """Amendment accession -> its original, and the number of amendments with several candidates.

    The original shares the ticker and an owner, is not an amendment, and was filed on the
    amendment's `original_submission_date`, on or before the amendment. By default several
    candidates resolve by shared cells then accession; `require_unique` refuses such ties.
    """
    none = pd.Series(dtype="string")
    if "document_type" not in df.columns or "original_submission_date" not in df.columns:
        return none, 0
    amend = df["document_type"].astype("string").str.strip().str.upper().str.endswith("/A").fillna(False).astype(bool)
    if not amend.any():
        return none, 0

    df_acc = df_key[["accession_number", "ticker"]].assign(day=df_key["day"], submitted=df["original_submission_date"])
    df_amend = df_acc[amend].drop_duplicates("accession_number")
    df_amend = df_amend[df_amend["submitted"].notna() & (df_amend["submitted"] <= df_amend["day"])]
    df_orig = df_acc[~amend].drop_duplicates("accession_number")[["accession_number", "ticker", "day"]]
    df_owner = _owner_pairs(df)
    df_orig = df_orig.merge(df_owner, on="accession_number").rename(columns={"accession_number": "original", "day": "submitted"})
    df_pairs = df_amend.merge(df_owner, on="accession_number").merge(df_orig, on=["ticker", "submitted", "owner"])
    df_pairs = df_pairs[["accession_number", "original"]].drop_duplicates()
    if df_pairs.empty:
        return none, 0

    df_cells = df_key[["accession_number", *CELL_KEY]].dropna(subset=["transaction_day"]).drop_duplicates()
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
    if require_unique:
        links = links[n_candidates.reindex(links.index).eq(1)]
    return links, int((n_candidates > 1).sum())


def _corrected_action_cells(df_rows: pd.DataFrame, unique_links: pd.Series) -> pd.Series:
    """Match date-only corrections only with unique filing, explicit row identity and complete unchanged correspondence."""
    cell_days = df_rows["transaction_day"].copy()
    future_original = df_rows["is_original"] & cell_days.isna() & df_rows["declared_day"].gt(df_rows["day"])
    keys = ["owner", "account", "security_type", "security_title", "row_sequence", "code", "source_shares", "source_price", "source_owned"]
    known = (
        df_rows["owner"].ne("")
        & df_rows["account"].eq("D")
        & df_rows["security_type"].eq("nonderiv")
        & df_rows["security_title"].ne("")
        & df_rows["row_sequence"].gt(0)
        & df_rows["row_sequence"].mod(1).eq(0)
        & df_rows["n_reporting_owners"].eq(1)
        & (df_rows["transaction_day"].notna() | future_original)
    )
    df_known = df_rows.loc[known].dropna(subset=keys)
    df_amended = df_known.loc[df_known["accession_number"].isin(unique_links.index)]
    df_original = df_known.loc[df_known["is_original"], ["original", *keys, "transaction_day", "row"]].copy()
    df_original["transaction_day"] = df_original["transaction_day"].fillna(df_known["declared_day"])
    df_matches = df_amended.merge(df_original, on=["original", *keys], suffixes=("", "_original"))
    if df_matches.empty:
        return cell_days
    paired = df_matches.groupby(["accession_number", "row"])["row_original"].transform("size").eq(1)
    paired &= df_matches.groupby(["accession_number", "row_original"])["row"].transform("size").eq(1)
    df_matches = df_matches.loc[paired]
    counts = df_rows.groupby("accession_number").size()
    matched = df_matches.groupby("accession_number").size()
    complete = df_matches["accession_number"].map(matched).eq(df_matches["accession_number"].map(counts)) & df_matches["accession_number"].map(
        matched
    ).eq(df_matches["original"].map(counts))
    corrections = df_matches.loc[complete & df_matches["transaction_day"].ne(df_matches["transaction_day_original"])]
    cell_days.loc[corrections["row"].to_numpy()] = corrections["transaction_day_original"].to_numpy()
    confirmed = df_matches.loc[complete & df_matches["row_original"].isin(df_rows.loc[future_original, "row"])]
    cell_days.loc[confirmed["row_original"].to_numpy()] = confirmed["transaction_day_original"].to_numpy()
    return cell_days


def _declared_action_cells(df_key: pd.DataFrame, unique_links: pd.Series) -> pd.Series:
    """Let a later verified amendment confirm an original's declared future day for supersession only."""
    cell_days = df_key["transaction_day"].copy()
    if unique_links.empty:
        return cell_days
    df_original = df_key.loc[
        df_key["accession_number"].isin(unique_links.to_numpy()) & cell_days.isna() & df_key["declared_day"].gt(df_key["day"]),
        ["accession_number", "declared_day", "code", "security_type"],
    ].reset_index(names="row")
    if df_original.empty:
        return cell_days
    df_amended = df_key.loc[df_key["accession_number"].isin(unique_links.index) & cell_days.notna()].assign(
        original=lambda rows: rows["accession_number"].map(unique_links)
    )
    df_matches = df_original.merge(
        df_amended[["original", *CELL_KEY]].drop_duplicates(),
        left_on=["accession_number", "declared_day", "code", "security_type"],
        right_on=["original", *CELL_KEY],
    )
    cell_days.loc[df_matches["row"].to_numpy()] = df_matches["declared_day"].to_numpy()
    return cell_days


def _supersede_plan(df_key: pd.DataFrame, links: pd.Series, unique_links: pd.Series) -> tuple[pd.DataFrame, dict]:
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
    df_rows["cell_day"] = _corrected_action_cells(df_rows, unique_links).fillna(_declared_action_cells(df_key, unique_links))
    df_rows = df_rows.dropna(subset=["cell_day"])
    if df_rows.empty:
        return pd.DataFrame(columns=columns), counts
    cell = ["original", "cell_day", *(name for name in CELL_KEY if name != "transaction_day")]
    unknown_original = df_rows["is_original"] & df_rows["transaction_day"].isna()
    if unknown_original.any():
        confirmed_cells = pd.MultiIndex.from_frame(df_rows.loc[unknown_original, cell])
        independent = df_rows["is_original"] | df_rows["accession_number"].isin(unique_links.index)
        df_rows = df_rows[independent | ~pd.MultiIndex.from_frame(df_rows[cell]).isin(confirmed_cells)]
    shares, price, owned = (df_rows[c].to_numpy(dtype="float64").astype("U32") for c in ("shares", "price", "owned_after"))
    df_rows["record"] = np.char.add(np.char.add(np.char.add(np.char.add(shares, "|"), price), "|"), owned)
    df_rows["record"] = df_rows["record"] + "|" + df_rows["transaction_day"].astype(str)
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
