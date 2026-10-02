"""Shared insider Forms 3/4/5 contract for the quarterly bulk ZIP path and the live EDGAR XML path.

Both paths extract the same canonical string frame (`INSIDER_FIELDS`), type it once with
`build_insider_frame`, and screen it with `screen_insider_rows`. The bulk-vs-live differences
that reach stored values are explicit arguments: `value_rule` (which `value_usd` source wins)
and `numeric_rule` (pandas `to_numeric` vs Python `float` after stripping `,` and `$`).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from src.data_extract.utils.common.identity import Identity

#: The `insider_transactions` column contract.
INSIDER_COLUMNS = [
    "accession_number",
    "security_type",
    "transaction_sk",
    "ticker",
    "issuer_cik",
    "issuer_name",
    "owner_cik",
    "owner_name",
    "is_director",
    "is_officer",
    "is_ten_pct_owner",
    "is_other",
    "officer_title",
    "document_type",
    "transaction_date",
    "filing_date",
    "period_of_report",
    "security_title",
    "transaction_code",
    "acquired_disposed",
    "shares",
    "price_per_share",
    "value_usd",
    "shares_owned_after",
    "direct_indirect",
    "quarter",
    "is_10b5_1",
    "transaction_form_type",
    "equity_swap_involved",
    "deemed_execution_date",
    "nature_of_ownership",
    "transaction_timeliness",
    "exercise_price",
    "exercise_date",
    "expiration_date",
    "underlying_security_title",
    "underlying_shares",
    "underlying_value",
]
FOOTNOTE_COLUMNS = ["accession_number", "footnote_id", "footnote_text"]
#: Quarantine rows carry every `insider_transactions` column plus the verdict.
VERDICT_COLUMNS = ["reject_reason", "resolved_entity_id", "universe_entity_id", "screened_on"]
QUARANTINE_COLUMNS = INSIDER_COLUMNS + VERDICT_COLUMNS

#: Yes/no source text (`AFF10B5ONE`, `aff10b5One`, relationship checkboxes); anything else is unknown.
FLAG_TRUE = frozenset({"1", "true", "y", "yes"})
FLAG_FALSE = frozenset({"0", "false", "n", "no"})
#: Date formats tried in order on the insider date fields of both paths.
INSIDER_DATE_FORMATS = ("mixed",)
#: Role flag -> regex matched in the lower-cased comma-joined relationship text.
ROLE_PATTERNS = {"is_director": "director", "is_officer": "officer", "is_ten_pct_owner": "ten|10", "is_other": "other"}


class InsiderField(NamedTuple):
    """One canonical field: its kind, bulk TSV column, and XML paths (first non-empty wins) under `scope`."""

    name: str
    kind: Literal["text", "symbol", "number", "date", "flag", "role"]
    scope: Literal["filing", "owner", "transaction"]
    bulk: str | None
    xml: tuple[str, ...]


#: The canonical string frame. XML paths are relative to the root (`filing`), the first
#: `reportingOwner` (`owner`), or one transaction node (`transaction`). `relationship` is the
#: comma-joined role names (`Director,Officer,TenPercentOwner,Other`); `total_value` feeds `value_usd`.
INSIDER_FIELDS = (
    InsiderField("issuer_cik", "text", "filing", "ISSUERCIK", ("issuer/issuerCik",)),
    InsiderField("issuer_name", "text", "filing", "ISSUERNAME", ("issuer/issuerName",)),
    InsiderField("ticker", "symbol", "filing", "ISSUERTRADINGSYMBOL", ("issuer/issuerTradingSymbol",)),
    InsiderField("document_type", "text", "filing", "DOCUMENT_TYPE", ("documentType",)),
    InsiderField("filing_date", "date", "filing", "FILING_DATE", ()),
    InsiderField("period_of_report", "date", "filing", "PERIOD_OF_REPORT", ("periodOfReport",)),
    InsiderField("is_10b5_1", "flag", "filing", "AFF10B5ONE", ("aff10b5One",)),
    InsiderField("owner_cik", "text", "owner", "RPTOWNERCIK", ("reportingOwnerId/rptOwnerCik",)),
    InsiderField("owner_name", "text", "owner", "RPTOWNERNAME", ("reportingOwnerId/rptOwnerName",)),
    InsiderField("officer_title", "text", "owner", "RPTOWNER_TITLE", ("reportingOwnerRelationship/officerTitle",)),
    InsiderField("relationship", "role", "owner", "RPTOWNER_RELATIONSHIP", ("reportingOwnerRelationship",)),
    InsiderField("security_title", "text", "transaction", "SECURITY_TITLE", ("securityTitle",)),
    InsiderField("transaction_date", "date", "transaction", "TRANS_DATE", ("transactionDate",)),
    InsiderField("transaction_code", "text", "transaction", "TRANS_CODE", ("transactionCoding/transactionCode",)),
    InsiderField("acquired_disposed", "text", "transaction", "TRANS_ACQUIRED_DISP_CD", ("transactionAmounts/transactionAcquiredDisposedCode",)),
    InsiderField("shares", "number", "transaction", "TRANS_SHARES", ("transactionAmounts/transactionShares",)),
    InsiderField("price_per_share", "number", "transaction", "TRANS_PRICEPERSHARE", ("transactionAmounts/transactionPricePerShare",)),
    InsiderField("total_value", "number", "transaction", "TRANS_TOTAL_VALUE", ("transactionAmounts/transactionTotalValue",)),
    InsiderField(
        "shares_owned_after", "number", "transaction", "SHRS_OWND_FOLWNG_TRANS", ("postTransactionAmounts/sharesOwnedFollowingTransaction",)
    ),
    InsiderField("direct_indirect", "text", "transaction", "DIRECT_INDIRECT_OWNERSHIP", ("ownershipNature/directOrIndirectOwnership",)),
    InsiderField("transaction_form_type", "text", "transaction", "TRANS_FORM_TYPE", ("transactionCoding/transactionFormType",)),
    InsiderField("equity_swap_involved", "text", "transaction", "EQUITY_SWAP_INVOLVED", ("transactionCoding/equitySwapInvolved",)),
    InsiderField("deemed_execution_date", "date", "transaction", "DEEMED_EXECUTION_DATE", ("deemedExecutionDate",)),
    InsiderField("nature_of_ownership", "text", "transaction", "NATURE_OF_OWNERSHIP", ("ownershipNature/natureOfOwnership",)),
    InsiderField(
        "transaction_timeliness", "text", "transaction", "TRANS_TIMELINESS", ("transactionTimeliness", "transactionCoding/transactionTimeliness")
    ),
    InsiderField("exercise_price", "number", "transaction", "CONV_EXERCISE_PRICE", ("conversionOrExercisePrice",)),
    # SEC misspells this column `EXCERCISE_DATE` in every quarterly data set.
    InsiderField("exercise_date", "date", "transaction", "EXCERCISE_DATE", ("exerciseDate",)),
    InsiderField("expiration_date", "date", "transaction", "EXPIRATION_DATE", ("expirationDate",)),
    InsiderField("underlying_security_title", "text", "transaction", "UNDLYNG_SEC_TITLE", ("underlyingSecurity/underlyingSecurityTitle",)),
    InsiderField("underlying_shares", "number", "transaction", "UNDLYNG_SEC_SHARES", ("underlyingSecurity/underlyingSecurityShares",)),
    InsiderField("underlying_value", "number", "transaction", "UNDLYNG_SEC_VALUE", ("underlyingSecurity/underlyingSecurityValue",)),
)

ValueRule = Literal["shares_x_price_first", "stated_total_first"]
NumericRule = Literal["to_numeric", "strip_currency_float"]


# --------------------------------------------------------------------------- #
# Coercers                                                                      #
# --------------------------------------------------------------------------- #
def _python_float(value: object) -> float:
    """`float(value)`, NaN when missing or not a number."""
    if not isinstance(value, str):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def coerce_numeric(s: pd.Series, *, numeric_rule: NumericRule) -> pd.Series:
    """Numbers from source text: pandas `to_numeric` (bulk TSV) or, after stripping `,` and `$`,
    Python `float` per value (live XML; `to_numeric` rounds some long decimals differently)."""
    if numeric_rule == "to_numeric":
        return pd.to_numeric(s, errors="coerce")
    text = s.astype("string").str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    return pd.Series([_python_float(value) for value in text], index=s.index, dtype="float64")


def parse_sec_date(s: pd.Series, *, formats: Sequence[str]) -> pd.Series:
    """Dates from SEC text: each format in turn fills the values the previous ones left NaT."""
    out = pd.to_datetime(s, format=formats[0], errors="coerce")
    for date_format in formats[1:]:
        todo = out.isna() & s.notna()
        out = out.mask(todo, pd.to_datetime(s.where(todo), format=date_format, errors="coerce"))
    return out


def normalize_flag(s: pd.Series) -> pd.Series:
    """Yes/no text -> 1.0 / 0.0 / NaN; float so an absent answer never reads as False."""
    text = s.astype("string").str.strip().str.lower()
    out = pd.Series(float("nan"), index=s.index, dtype="float64")
    return out.mask(text.isin(FLAG_TRUE), 1.0).mask(text.isin(FLAG_FALSE), 0.0)


def _role_flags(relationship: pd.Series) -> dict[str, pd.Series]:
    """Role flags matched in the relationship text; NaN where the relationship is unknown."""
    text = relationship.astype("string").str.lower()
    return {name: text.str.contains(pattern, na=False).astype(float).where(text.notna()) for name, pattern in ROLE_PATTERNS.items()}


def _value_usd(df: pd.DataFrame, value_rule: ValueRule) -> pd.Series:
    """`value_usd`: shares x price filled by the stated total, or the stated total filled by a finite shares x price."""
    product = df["shares"] * df["price_per_share"]
    if value_rule == "shares_x_price_first":
        return product.fillna(df["total_value"])
    finite_product = product.where(np.isfinite(df["shares"]) & np.isfinite(df["price_per_share"]))
    return df["total_value"].where(np.isfinite(df["total_value"]), finite_product)


# --------------------------------------------------------------------------- #
# Canonical frame                                                               #
# --------------------------------------------------------------------------- #
def _built_columns(columns: Iterable[str]) -> list[str]:
    """Column names `build_insider_frame` returns for a string frame with `columns`."""
    names = list(columns)
    kept = [name for name in names if name not in ("relationship", "total_value")]
    value = ["value_usd"] if "total_value" in names else []
    roles = list(ROLE_PATTERNS) if "relationship" in names else []
    return kept + value + roles


def build_insider_frame(df_str: pd.DataFrame, *, value_rule: ValueRule, numeric_rule: NumericRule) -> pd.DataFrame:
    """Type a canonical string frame: numbers, dates, yes/no flags, upper-cased ticker, the four
    role flags from `relationship`, and `value_usd`; the two helper columns are dropped."""
    if df_str.empty:
        return pd.DataFrame(columns=_built_columns(df_str.columns))
    fields = [field for field in INSIDER_FIELDS if field.name in df_str.columns]
    typed = {field.name: _typed(df_str[field.name], field.kind, numeric_rule) for field in fields if field.kind not in ("text", "role")}
    df_built = df_str.assign(**typed)
    df_built = df_built.assign(value_usd=_value_usd(df_built, value_rule), **_role_flags(df_built["relationship"]))
    return df_built.drop(columns=["relationship", "total_value"])


def _typed(s: pd.Series, kind: str, numeric_rule: NumericRule) -> pd.Series:
    """One string column typed by its field kind."""
    if kind == "number":
        return coerce_numeric(s, numeric_rule=numeric_rule)
    if kind == "date":
        return parse_sec_date(s, formats=INSIDER_DATE_FORMATS)
    if kind == "flag":
        return normalize_flag(s)
    return s.str.strip().str.upper()


# --------------------------------------------------------------------------- #
# Universe screen                                                               #
# --------------------------------------------------------------------------- #
def repair_transaction_dates(df: pd.DataFrame) -> pd.DataFrame:
    """A transaction cannot postdate the filing that discloses it. A year below 1900 is moved into
    the filing's century when that lands on or before the filing date; any `transaction_date`
    still after `filing_date` is nulled."""
    if df.empty or not {"transaction_date", "filing_date"}.issubset(df.columns):
        return df
    td, fd = df["transaction_date"], df["filing_date"]
    lost = td.notna() & fd.notna() & (td.dt.year < 1900)
    if lost.any():
        shifted = td.where(~lost).copy()
        for i in df.index[lost]:
            try:
                candidate = td[i].replace(year=(fd[i].year // 100) * 100 + td[i].year % 100)
            except ValueError:  # 29 Feb in a non-leap target year
                continue
            if candidate <= fd[i]:
                shifted[i] = candidate
        df = df.assign(transaction_date=shifted)
        td = df["transaction_date"]
    impossible = td.notna() & fd.notna() & (td > fd)
    if impossible.any():
        df = df.assign(transaction_date=td.mask(impossible))
    return df


def insider_verdicts(df: pd.DataFrame, universe: Collection[str], identity: Identity) -> pd.DataFrame:
    """Resolve each row CIK-first and attach `claimed_ticker`, the resolved `ticker`, both entity
    ids, `screened_on` (= `filing_date`) and `reject_reason` (NA on kept rows)."""
    universe = set(universe)
    raw = df["issuer_cik"]
    unique_ciks = [value for value in pd.unique(raw) if value is not None and not pd.isna(value)]
    to_ticker = {value: identity.entity_ticker(value) for value in unique_ciks}
    to_entity = {value: identity.entity_of(value) for value in unique_ciks}

    claimed = df["ticker"].astype("string")
    resolved = raw.map(to_ticker).astype("string")
    # `universe_entity` raises on a ticker absent from the roster, so only roster tickers are asked.
    known = {ticker: identity.universe_entity(ticker) for ticker in set(claimed.dropna()) & set(identity.roster_cik)}

    screened_on = df["filing_date"] if "filing_date" in df.columns else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    out = df.assign(
        claimed_ticker=claimed,
        ticker=resolved,
        resolved_entity_id=raw.map(to_entity).astype("string"),
        universe_entity_id=claimed.map(known).astype("string"),
        screened_on=screened_on,
    )
    keep = resolved.isin(universe)
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason[~keep] = "entity_not_in_universe"
    reason[~keep & claimed.isin(universe)] = "entity_mismatch"
    reason[~keep & (raw.isna() | (raw.astype("string").str.strip() == ""))] = "no_issuer_cik"
    return out.assign(reject_reason=reason)


def quarantine_frame(rejected: pd.DataFrame) -> pd.DataFrame:
    """Rejected rows in `insider_transactions_quarantine` shape; `ticker` is the CLAIMED symbol."""
    if rejected is None or rejected.empty:
        return pd.DataFrame(columns=QUARANTINE_COLUMNS)
    out = rejected.assign(ticker=rejected["claimed_ticker"])
    return out[[column for column in QUARANTINE_COLUMNS if column in out.columns]]


def screen_insider_rows(df: pd.DataFrame, universe: Sequence[str], identity: Identity) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair transaction dates, resolve CIK-first, and split into (kept, quarantine).

    No date filter: Forms 3/4/5 are union events. The quarantine is scoped to rows that claimed a
    universe ticker, resolve to a roster company, or carry no CIK; other filers are dropped.
    """
    if df.empty:
        return df, pd.DataFrame(columns=QUARANTINE_COLUMNS)
    scored = insider_verdicts(repair_transaction_dates(df), universe, identity)
    keep = scored["reject_reason"].isna()
    in_scope = scored["claimed_ticker"].isin(set(universe)) | scored["ticker"].notna() | (scored["reject_reason"] == "no_issuer_cik")
    return scored[keep], quarantine_frame(scored[~keep & in_scope])


# --------------------------------------------------------------------------- #
# Footnotes                                                                     #
# --------------------------------------------------------------------------- #
def empty_footnotes() -> pd.DataFrame:
    """An empty `insider_footnotes` frame."""
    return pd.DataFrame(columns=FOOTNOTE_COLUMNS)


def filter_footnotes(notes: pd.DataFrame, keep_accessions: Collection[str]) -> pd.DataFrame:
    """Footnotes of the kept accessions only, keyed on (accession_number, footnote_id)."""
    if notes.empty:
        return empty_footnotes()
    out = notes[notes["accession_number"].isin(keep_accessions)].dropna(subset=["accession_number", "footnote_id"])
    return out[FOOTNOTE_COLUMNS] if not out.empty else empty_footnotes()
