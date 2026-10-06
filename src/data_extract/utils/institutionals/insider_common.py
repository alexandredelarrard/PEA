"""Shared insider Forms 3/4/5 contract for the quarterly bulk ZIP path and the EDGAR XML path.

Both paths extract the same two canonical string frames from `INSIDER_FIELDS`: one row per
transaction line (keyed by `row_sequence`) and one row per reporting owner. `build_insider_frame`
types them with one numeric parser, one value rule and one owner rule; only the date formats
differ by source. `screen_insider_rows` then resolves each row CIK-first and stamps the kept rows'
lineage (`stamp_lineage`: the filing's own symbol, the economic date and the role of the issuer CIK on
that date); rejected rows, co-registrant rows included, are never stored, only summarised by `log_exclusions`.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator, Sequence
from typing import Literal, NamedTuple

import pandas as pd

from src.constants.constants import CANONICAL_CURRENT, CANONICAL_PREDECESSOR
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.common.security_master import ACQUIRED_CONSTITUENT, MergerBoundary
from src.data_store.schema import Tables
from src.utils.string import pad_cik, pad_cik_series

#: The `insider_transactions` row key.
INSIDER_KEY = list(Tables.insider_transactions.pk)
#: Accessions per `IN` list when stored rows are read or deleted by accession.
ACCESSION_BATCH = 2_000
#: The `insider_transactions` column contract.
INSIDER_COLUMNS = [
    "accession_number",
    "security_type",
    "row_sequence",
    "source",
    "ticker",
    "issuer_cik",
    "issuer_name",
    "owner_cik",
    "owner_name",
    "owner_ciks",
    "n_reporting_owners",
    "is_director",
    "is_officer",
    "is_ten_pct_owner",
    "is_other",
    "officer_title",
    "document_type",
    "original_submission_date",
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
    "footnote_ids",
    "acceptance_datetime",
    "fetched_at",
    "source_symbol",
    "economic_date",
    "lineage_role",
]
#: The lineage stamp `stamp_lineage` adds and a re-stamp rewrites.
LINEAGE_COLUMNS = ["source_symbol", "economic_date", "lineage_role"]
FOOTNOTE_COLUMNS = ["accession_number", "footnote_id", "footnote_text"]
#: The columns of a rejected in-scope row that the identity-exclusion warning reads.
EXCLUSION_COLUMNS = ["accession_number", "transaction_code", "claimed_ticker", "reject_reason"]
#: Reject reasons in warning order, and the transaction codes counted as open-market trades.
REJECT_REASONS = ("entity_mismatch", "entity_not_in_universe", "no_issuer_cik", "co_registrant")
OPEN_MARKET_CODES = frozenset({"P", "S"})
#: Boundary-day codes under the legal acquirer: received in the merger, and surrendered (disposed / other).
RECEIVED_CODES = frozenset({"A"})
SURRENDERED_CODES = frozenset({"D", "J"})
#: Row identity of an amendment's original: same issuer, owner, table, security title and row.
_ORIGINAL_KEY = ["_cik", "owner_cik", "security_type", "security_title", "row_sequence"]

#: Yes/no source text (`AFF10B5ONE`, `aff10b5One`, relationship checkboxes); anything else is unknown.
FLAG_TRUE = frozenset({"1", "true", "y", "yes"})
FLAG_FALSE = frozenset({"0", "false", "n", "no"})
#: Date formats tried in order: SEC bulk TSVs ship `03-FEB-2026` (ISO as fallback); EDGAR XML ships ISO dates.
BULK_DATE_FORMATS = ("%d-%b-%Y", "ISO8601")
XML_DATE_FORMATS = ("mixed",)
#: Role flag -> regex matched in the lower-cased comma-joined relationship text.
ROLE_PATTERNS = {"is_director": "director", "is_officer": "officer", "is_ten_pct_owner": "ten|10", "is_other": "other"}
#: Primary-owner precedence: an owner ranks by its best role in this order; no role ranks last.
ROLE_RANK = ("is_officer", "is_director", "is_ten_pct_owner", "is_other")
#: Owner fields that describe the primary owner of an accession.
PRIMARY_OWNER_COLUMNS = ["owner_cik", "owner_name", "officer_title"]
#: Per-accession owner columns `build_insider_frame` attaches to every transaction row.
OWNER_SUMMARY_COLUMNS = [*PRIMARY_OWNER_COLUMNS, "owner_ciks", "n_reporting_owners", *ROLE_PATTERNS]


class InsiderField(NamedTuple):
    """One canonical field: its kind, bulk TSV column, and XML paths (first non-empty wins) under `scope`."""

    name: str
    kind: Literal["text", "symbol", "number", "date", "flag", "role"]
    scope: Literal["filing", "owner", "transaction"]
    bulk: str | None
    xml: tuple[str, ...]


#: The canonical string fields. XML paths are relative to the root (`filing`), each
#: `reportingOwner` (`owner`), or one transaction node (`transaction`). `relationship` is the
#: comma-joined role names (`Director,Officer,TenPercentOwner,Other`); `total_value` feeds `value_usd`.
INSIDER_FIELDS = (
    InsiderField("issuer_cik", "text", "filing", "ISSUERCIK", ("issuer/issuerCik",)),
    InsiderField("issuer_name", "text", "filing", "ISSUERNAME", ("issuer/issuerName",)),
    InsiderField("ticker", "symbol", "filing", "ISSUERTRADINGSYMBOL", ("issuer/issuerTradingSymbol",)),
    InsiderField("document_type", "text", "filing", "DOCUMENT_TYPE", ("documentType",)),
    InsiderField("filing_date", "date", "filing", "FILING_DATE", ()),
    InsiderField("period_of_report", "date", "filing", "PERIOD_OF_REPORT", ("periodOfReport",)),
    InsiderField("original_submission_date", "date", "filing", "DATE_OF_ORIG_SUB", ("dateOfOriginalSubmission",)),
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

#: The owner string frame: one row per reporting owner of an accession.
OWNER_STRING_COLUMNS = ["accession_number", *(field.name for field in INSIDER_FIELDS if field.scope == "owner")]


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


def parse_number(s: pd.Series) -> pd.Series:
    """Numbers from SEC text: `,` and `$` stripped, then Python `float` per value; NaN when missing
    or not a number."""
    text = s.astype("string").str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    return pd.Series([_python_float(value) for value in text], index=s.index, dtype="float64")


def parse_sec_date(s: pd.Series, *, formats: Sequence[str]) -> pd.Series:
    """Dates from SEC text: each format in turn fills the values the previous ones left NaT; the
    first format sets the datetime resolution."""
    out = pd.to_datetime(s, format=formats[0], errors="coerce")
    for date_format in formats[1:]:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        out = out.mask(todo, pd.to_datetime(s.where(todo), format=date_format, errors="coerce"))
    return out


def normalize_flag(s: pd.Series) -> pd.Series:
    """Yes/no text -> 1.0 / 0.0 / NaN; float so an absent answer never reads as False."""
    text = s.astype("string").str.strip().str.lower()
    out = pd.Series(float("nan"), index=s.index, dtype="float64")
    return out.mask(text.isin(FLAG_TRUE), 1.0).mask(text.isin(FLAG_FALSE), 0.0)


def _role_flags(relationship: pd.Series) -> dict[str, pd.Series]:
    """Role flags 1.0/0.0 matched in the relationship text; an absent or blank relationship has no role."""
    text = relationship.astype("string").str.lower()
    return {name: text.str.contains(pattern, na=False).astype(float) for name, pattern in ROLE_PATTERNS.items()}


def _number_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Typed number column `name`, all NaN when the frame has none."""
    return df[name] if name in df.columns else pd.Series(float("nan"), index=df.index, dtype="float64")


def _value_usd(df: pd.DataFrame) -> pd.Series:
    """`value_usd`: shares x price, filled by the stated total where either is missing."""
    product = _number_column(df, "shares") * _number_column(df, "price_per_share")
    return product.fillna(_number_column(df, "total_value"))


# --------------------------------------------------------------------------- #
# Owners                                                                        #
# --------------------------------------------------------------------------- #
def _owner_rank(df_roles: pd.DataFrame) -> pd.Series:
    """Each owner's best role position in `ROLE_RANK`; `len(ROLE_RANK)` when it has none."""
    rank = pd.Series(len(ROLE_RANK), index=df_roles.index, dtype="int64")
    for position, name in reversed(list(enumerate(ROLE_RANK))):
        rank = rank.mask(df_roles[name].eq(1.0), position)
    return rank


def owner_summary(df_owners: pd.DataFrame) -> pd.DataFrame:
    """One row per accession from the owner string frame: the primary owner's CIK, name and title
    (best role rank, then lowest numeric CIK, missing CIK last), `owner_ciks` (sorted unique 10-digit
    CIKs joined by `,`), `n_reporting_owners` (its length) and each role flag OR'ed across owners."""
    if df_owners.empty:
        return pd.DataFrame(columns=["accession_number", *OWNER_SUMMARY_COLUMNS])
    ciks = pad_cik_series(df_owners["owner_cik"])
    df_owned = df_owners.assign(owner_cik=ciks.where(ciks != "", None), **_role_flags(df_owners["relationship"]))
    df_ranked = df_owned.assign(_rank=_owner_rank(df_owned), _cik_number=pd.to_numeric(df_owned["owner_cik"], errors="coerce"))
    df_primary = df_ranked.sort_values(["accession_number", "_rank", "_cik_number"], na_position="last", kind="mergesort").drop_duplicates(
        "accession_number", keep="first"
    )
    df_ciks = df_owned.dropna(subset=["owner_cik"]).drop_duplicates(["accession_number", "owner_cik"]).sort_values(["accession_number", "owner_cik"])
    joined = df_ciks.groupby("accession_number")["owner_cik"].agg(",".join).rename("owner_ciks")
    counts = df_ciks.groupby("accession_number").size().rename("n_reporting_owners")
    flags = df_owned.groupby("accession_number")[list(ROLE_PATTERNS)].max()
    df_summary = df_primary.set_index("accession_number")[PRIMARY_OWNER_COLUMNS].join(joined).join(counts).join(flags)
    df_summary["n_reporting_owners"] = df_summary["n_reporting_owners"].fillna(0).astype("int64")
    return df_summary.reset_index()[["accession_number", *OWNER_SUMMARY_COLUMNS]]


# --------------------------------------------------------------------------- #
# Canonical frame                                                               #
# --------------------------------------------------------------------------- #
def build_insider_frame(df_str: pd.DataFrame, df_owners: pd.DataFrame, *, date_formats: Sequence[str]) -> pd.DataFrame:
    """Type the transaction string frame (numbers, dates, yes/no flags, ticker, `value_usd`) and
    attach `owner_summary(df_owners)` to every row of its accession. An accession without an owner
    row gets role flags 0, `n_reporting_owners` 0 and NULL owner fields; `total_value` is dropped."""
    if df_str.empty:
        return pd.DataFrame(columns=[*(name for name in df_str.columns if name != "total_value"), "value_usd", *OWNER_SUMMARY_COLUMNS])
    fields = [field for field in INSIDER_FIELDS if field.name in df_str.columns and field.kind != "text"]
    df_typed = df_str.assign(**{field.name: _typed(df_str[field.name], field.kind, date_formats) for field in fields})
    df_typed = df_typed.assign(value_usd=_value_usd(df_typed)).drop(columns=["total_value"], errors="ignore")
    df_built = df_typed.merge(owner_summary(df_owners), on="accession_number", how="left")
    no_owner = {name: df_built[name].fillna(0.0).astype("float64") for name in ROLE_PATTERNS}
    return df_built.assign(n_reporting_owners=df_built["n_reporting_owners"].fillna(0).astype("int64"), **no_owner)


def _symbol_text(s: pd.Series) -> pd.Series:
    """A claimed trading symbol: stripped and upper-cased; blank is missing."""
    text = s.astype("string").str.strip().str.upper()
    return text.mask(text.eq("").fillna(False))


def _typed(s: pd.Series, kind: str, date_formats: Sequence[str]) -> pd.Series:
    """One string column typed by its field kind."""
    if kind == "number":
        return parse_number(s)
    if kind == "date":
        return parse_sec_date(s, formats=date_formats)
    if kind == "flag":
        return normalize_flag(s)
    return _symbol_text(s)


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
    """Resolve each row by its issuer CIK (event policy: any CIK of the entity) and attach
    `claimed_ticker`, the resolved `ticker` and `reject_reason` (NA on kept rows). A co-registrant
    CIK's rows resolve to their company but are rejected (`co_registrant`)."""
    universe = set(universe)
    raw = df["issuer_cik"]
    unique_ciks = [value for value in pd.unique(raw) if value is not None and not pd.isna(value)]
    to_ticker = {value: identity.ticker_for_cik(value, None, "event") for value in unique_ciks}

    claimed = df["ticker"].astype("string")
    resolved = raw.map(to_ticker).astype("string")
    keep = resolved.isin(universe)
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason[~keep] = "entity_not_in_universe"
    reason[~keep & claimed.isin(universe)] = "entity_mismatch"
    reason[~keep & (raw.isna() | (raw.astype("string").str.strip() == ""))] = "no_issuer_cik"
    co_registrant = raw.map({value: pad_cik(value) in identity.co_registrant_ciks for value in unique_ciks}).fillna(False).astype(bool)
    reason[keep & co_registrant] = "co_registrant"
    return df.assign(claimed_ticker=claimed, ticker=resolved, reject_reason=reason)


def screen_insider_rows(df: pd.DataFrame, universe: Sequence[str], identity: Identity) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair transaction dates, resolve CIK-first, and split into (kept, rejected in scope).

    No date filter: Forms 3/4/5 are union events. Rejected rows are in scope when they claimed a
    universe ticker, resolve to a roster company, or carry no CIK; other filers are dropped silently.
    """
    if df.empty:
        return df, pd.DataFrame(columns=EXCLUSION_COLUMNS)
    scored = insider_verdicts(repair_transaction_dates(df), universe, identity)
    keep, in_scope = _screen_masks(scored, universe)
    return stamp_lineage(scored[keep], identity), scored[~keep & in_scope]


def _screen_masks(scored: pd.DataFrame, universe: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    """(kept, in-scope) masks over `insider_verdicts` output; both read only the issuer CIK and the
    claimed ticker."""
    keep = scored["reject_reason"].isna()
    in_scope = scored["claimed_ticker"].isin(set(universe)) | scored["ticker"].notna() | (scored["reject_reason"] == "no_issuer_cik")
    return keep, in_scope


def screened_accessions(df_filing: pd.DataFrame, universe: Sequence[str], identity: Identity) -> set[str]:
    """Accessions of a canonical filing-level frame (`accession_number`, `issuer_cik`, raw `ticker`)
    that `screen_insider_rows` keeps or rejects in scope; it drops every row of any other accession."""
    df_scored = insider_verdicts(df_filing.assign(ticker=_symbol_text(df_filing["ticker"])), universe, identity)
    keep, in_scope = _screen_masks(df_scored, universe)
    return set(df_scored.loc[keep | in_scope, "accession_number"])


# --------------------------------------------------------------------------- #
# Lineage stamp                                                                 #
# --------------------------------------------------------------------------- #
def _date_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Column `name` as datetimes, all NaT when the frame has none."""
    if name not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df[name], errors="coerce")


def _text_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Column `name` as stripped upper-case text, NA when the frame has none."""
    if name not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[name].astype("string").str.strip().str.upper()


def _is_amendment(df: pd.DataFrame) -> pd.Series:
    return _text_column(df, "document_type").str.endswith("/A").fillna(False).astype(bool)


def _own_dates(df: pd.DataFrame) -> pd.Series:
    """Each row's own economic date: Form 3 its period of report, every other form its transaction date."""
    form3 = _text_column(df, "document_type").str.startswith("3").fillna(False).astype(bool)
    return _date_column(df, "transaction_date").mask(form3, _date_column(df, "period_of_report"))


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    """The `_ORIGINAL_KEY` columns as text, the CIK padded and a missing column null."""
    ciks = pad_cik_series(df["issuer_cik"]) if "issuer_cik" in df.columns else pd.Series("", index=df.index)
    out = pd.DataFrame({"_cik": ciks.astype("string")}, index=df.index)
    for column in _ORIGINAL_KEY[1:]:
        out[column] = df[column].astype("string") if column in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    return out


def _inherited_dates(df: pd.DataFrame, missing: pd.Series, pool: pd.DataFrame) -> pd.Series:
    """For the amendment rows in `missing`, the own date of their original row in `pool`: not an amendment, filed
    on the amendment's `original_submission_date`, same `_ORIGINAL_KEY`. NaT where none is found."""
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if not missing.any() or pool.empty:
        return out
    originals = _keyed(pool).assign(_filed=_date_column(pool, "filing_date"), _date=_own_dates(pool))
    originals = originals[~_is_amendment(pool) & originals["_date"].notna()].drop_duplicates([*_ORIGINAL_KEY, "_filed"])
    wanted = _keyed(df[missing]).assign(_filed=_date_column(df[missing], "original_submission_date"), _row=df.index[missing])
    joined = wanted.merge(originals, on=[*_ORIGINAL_KEY, "_filed"], how="inner").drop_duplicates("_row")
    return out.fillna(joined.set_index("_row")["_date"].reindex(out.index))


def economic_dates(df: pd.DataFrame, originals: pd.DataFrame | None = None) -> pd.Series:
    """The date each row's economic event happened: Form 3 its period of report; Form 4/5 each row's
    transaction date (a late Form 5 keeps its old date); an amendment without one inherits its original
    row's (found in `df` or `originals`); otherwise the period of report, then the filing date."""
    dates = _own_dates(df)
    pool = df if originals is None or originals.empty else pd.concat([df, originals], ignore_index=True)
    dates = dates.fillna(_inherited_dates(df, dates.isna() & _is_amendment(df), pool))
    return dates.fillna(_date_column(df, "period_of_report")).fillna(_date_column(df, "transaction_date")).fillna(_date_column(df, "filing_date"))


def _boundary_role(boundary: MergerBoundary, cik: str, symbol: object, code: object) -> str | None:
    """A boundary-day row's role from the merger metadata, or None when no rule applies (the window decides).

    Open-market trades count as before the completion unless the merger closed before the open.
    """
    if isinstance(symbol, str) and symbol.strip().upper() == boundary.acquired_symbol:
        return ACQUIRED_CONSTITUENT
    if cik == boundary.legal_acquirer_cik:
        letter = code.strip().upper() if isinstance(code, str) else ""
        if letter in RECEIVED_CODES:
            return CANONICAL_CURRENT
        if letter in SURRENDERED_CODES or (letter in OPEN_MARKET_CODES and boundary.after_close is not False):
            return ACQUIRED_CONSTITUENT
    if cik == boundary.accounting_predecessor_cik:
        return CANONICAL_PREDECESSOR
    return None


def lineage_roles(df: pd.DataFrame, identity: Identity) -> pd.Series:
    """Each row's `lineage_role` from its issuer CIK and `economic_date` (`Identity.lineage_role`), with the
    boundary-day rules of the resolved ticker's mergers applied on the seam date."""
    ciks = pad_cik_series(df["issuer_cik"])
    days = pd.to_datetime(df["economic_date"], errors="coerce")
    keys = pd.DataFrame({"cik": ciks.to_numpy(), "day": days.to_numpy()})
    pairs = keys.drop_duplicates(ignore_index=True)
    pairs["role"] = [identity.lineage_role(cik, day) for cik, day in zip(pairs["cik"], pairs["day"], strict=True)]
    roles = pd.Series(keys.merge(pairs, on=["cik", "day"], how="left")["role"].to_numpy(dtype=object), index=df.index, dtype=object)
    tickers = df["ticker"].astype("string") if "ticker" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    codes = df["transaction_code"] if "transaction_code" in df.columns else pd.Series(None, index=df.index, dtype=object)
    for ticker, boundaries in identity.merger_boundaries.items():
        for boundary in boundaries:
            on_day = (tickers.eq(ticker) & days.eq(boundary.seam_date)).fillna(False).astype(bool)
            for index in df.index[on_day]:
                role = _boundary_role(boundary, ciks[index], df.at[index, "source_symbol"], codes[index])
                if role is not None:
                    roles[index] = role
    return roles


def stamp_lineage(df: pd.DataFrame, identity: Identity, originals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `source_symbol` (the filing's own trading symbol: `claimed_ticker` once the verdict has run),
    `economic_date` and `lineage_role` to resolved rows."""
    if df.empty:
        return df.assign(**{column: pd.Series(dtype=object) for column in LINEAGE_COLUMNS})
    if "claimed_ticker" in df.columns:
        symbol = _symbol_text(df["claimed_ticker"])
    else:
        symbol = df["source_symbol"] if "source_symbol" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    df_stamped = df.assign(source_symbol=symbol, economic_date=economic_dates(df, originals))
    return df_stamped.assign(lineage_role=lineage_roles(df_stamped, identity))


def accession_batches(accessions: Sequence[str]) -> Iterator[list[str]]:
    """`accessions` in consecutive lists of `ACCESSION_BATCH`, one store `IN` filter each."""
    for start in range(0, len(accessions), ACCESSION_BATCH):
        yield list(accessions[start : start + ACCESSION_BATCH])


def exclusion_rows(df_rejected: pd.DataFrame) -> pd.DataFrame:
    """The `EXCLUSION_COLUMNS` of rejected rows, the only part the exclusion warning keeps."""
    return df_rejected.reindex(columns=EXCLUSION_COLUMNS)


def top_counts(values: pd.Series, n: int) -> str:
    """`"T1 n1, T2 n2, ..."`: the `n` most frequent values, ties in value order; a missing value reads `<none>`."""
    df_counts = values.astype("string").fillna("<none>").value_counts().rename_axis("value").reset_index(name="n")
    df_top = df_counts.sort_values(["n", "value"], ascending=[False, True]).head(n)
    return ", ".join(f"{value} {count}" for value, count in zip(df_top["value"], df_top["n"], strict=True))


def exclusion_message(label: str, df_excluded: pd.DataFrame) -> str:
    """The identity-exclusion summary: filings, rows and P/S rows, filings per reject reason, and
    the 10 claimed tickers with the most excluded filings."""
    df_filings = df_excluded.drop_duplicates("accession_number")
    n_open_market = int(df_excluded["transaction_code"].isin(OPEN_MARKET_CODES).sum())
    by_reason = df_filings["reject_reason"].value_counts()
    reasons = ", ".join(f"{reason} {int(by_reason[reason])}" for reason in REJECT_REASONS if reason in by_reason.index)
    return (
        f"insider {label}: excluded {len(df_filings)} filing(s), {len(df_excluded)} row(s), {n_open_market} P/S row(s) whose issuer CIK "
        f"is not a universe entity; by reason: {reasons}; top claimed: {top_counts(df_filings['claimed_ticker'], 10)}"
    )


def log_exclusions(log: logging.Logger, label: str, frames: Sequence[pd.DataFrame]) -> None:
    """One WARNING summarising the excluded rows in `frames`, or one INFO line when there are none."""
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        log.info("insider %s: no filing excluded by the identity screen", label)
        return
    log.warning(exclusion_message(label, pd.concat(non_empty, ignore_index=True)))


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
