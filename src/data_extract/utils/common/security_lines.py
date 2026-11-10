"""The shared contract of `universe_cusip` lines: their source and role vocabulary, the FTD trade-date rule, the FTD
lines prepared for stamping and their change stamps, the tape intervals a line is voted by, the co-registrant CIKs it leaves out and the curated `configs/sec/security_master_manual.json`.

Read by the identity step (which builds the lines), the FTD and short-interest tapes, the insider lineage stamp, the
`Identity` reader and the fundamentals merges (merger metadata, reverse acquisitions, vendor exchanges).
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.common.identity_contract import DEI_SOURCE, ROLE_SYMBOL
from src.utils.cutover_continuity import ShareExchange
from src.utils.identity_flags import cik_activity, cik_items, flag_rows
from src.utils.predecessor_series import SECURITY_MANUAL_FILE
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series, squash, yahoo_symbol

#: The `source` of a line read off the SEC fails-to-deliver files.
SOURCE_FTD = "ftd"
ACQUIRED_CONSTITUENT = "acquired_constituent"
EXCLUDED = "excluded"
ROLES = (*CANONICAL_ROLES, SECONDARY_CLASS, ACQUIRED_CONSTITUENT, EXCLUDED)

#: First trade date of each settlement cycle (T+2 from 2017-09-05, T+1 from 2024-05-28).
T2_FROM = pd.Timestamp("2017-09-05")
T1_FROM = pd.Timestamp("2024-05-28")

#: Two tape intervals of one symbol overlapping by more than this are concurrent, not a former listing.
_LISTING_OVERLAP = pd.Timedelta(days=31)
_FAR = pd.Timestamp("2262-01-01")
#: The `universe_cusip` columns read for stamping FTD lines.
FTD_MASTER_COLUMNS = [
    "security_id",
    "canonical_company",
    "cusip",
    "source_symbol",
    "market_symbol",
    "security_class",
    "conversion_ratio",
    "lineage_role",
    "valid_from",
    "valid_to",
    "scope_changed_at",
]

_RATIO_COLUMNS = ("ticker", "cusip", "ratio", "valid_from", "valid_to", "source")
_BOUNDARY_COLUMNS = ("ticker", "cusip", "issuer_cik", "role", "valid_from", "valid_to", "reason", "source")
_CLASS_COLUMNS = ("cusip", "security_class", "source")
_EXCHANGE_COLUMNS = ("ticker", "predecessor_cik", "seam_date", "ratio", "source")
_REVERSE_COLUMNS = ("ticker", "seam_date", "accounting_acquirer_cik", "legal_acquirer_cik", "source")
_OVERRIDE_COLUMNS = ("cusip", "ticker", "source")


class SecurityManualError(ValueError):
    """A `security_master_manual.json` entry without a URL or accession, or with an unknown role."""


def trade_dates(settlement: pd.Series) -> pd.Series:
    """Trade dates of FTD settlement dates by the cycle in force (T+3, T+2 from 2017-09-05, T+1 from 2024-05-28)."""
    days = pd.to_datetime(settlement)
    bday = pd.offsets.BDay
    t1, t2, t3 = days - bday(1), days - bday(2), days - bday(3)
    out = t3.where(t2 < T2_FROM, t2).where(t1 < T1_FROM, t1)
    return out.where(days.notna())


def prepare_master(master: pd.DataFrame) -> pd.DataFrame:
    """The FTD rows of `universe_cusip`, keyed for stamping: upper-case CUSIP and symbol, open ends at a far date."""
    rows = master[master["source"].eq(SOURCE_FTD)] if "source" in master.columns else master
    out = rows.reindex(columns=FTD_MASTER_COLUMNS).copy()
    out["market_symbol"] = out["market_symbol"].fillna(out["source_symbol"]).map(yahoo_symbol)
    out["cusip"] = out["cusip"].astype(str).str.strip().str.upper()
    out["source_symbol"] = out["source_symbol"].fillna("").map(normalise_ticker)
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"]).fillna(_FAR)
    out["conversion_ratio"] = pd.to_numeric(out["conversion_ratio"], errors="coerce").astype("float64")
    company = out["canonical_company"].astype(object)
    out["canonical_company"] = company.where(company.notna() & company.astype(str).ne(""), None)
    out["scope_changed_at"] = pd.to_datetime(out["scope_changed_at"], errors="coerce")
    return out.reset_index(drop=True)


def master_stamps(master: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """`{canonical company: latest scope_changed_at}` over its master rows."""
    stamps = master.dropna(subset=["canonical_company"]).groupby("canonical_company")["scope_changed_at"].max()
    return {str(company): pd.Timestamp(stamp) for company, stamp in stamps.items() if pd.notna(stamp)}


@dataclass(frozen=True)
class ReverseAcquisition:
    """A merger whose legal survivor (the traded security) was not the accounting acquirer: from `seam_date` the
    survivor's filings restate pre-seam periods with the accounting acquirer's numbers."""

    ticker: str
    seam_date: pd.Timestamp
    accounting_acquirer_cik: str
    legal_acquirer_cik: str
    source: str


@dataclass(frozen=True)
class MergerBoundary:
    """One merger's boundary day for the insider lineage: who acquired whom on `seam_date`, and whether it closed after the market close."""

    ticker: str
    seam_date: pd.Timestamp
    after_close: bool | None
    legal_acquirer_cik: str
    accounting_predecessor_cik: str
    acquired_symbol: str


@dataclass(frozen=True)
class SecurityManual:
    """Parsed `security_master_manual.json`: dated conversion ratios, CUSIP market boundaries, class overrides, merger
    metadata, the declared co-registrant CIKs, the merger exchange ratios, the reverse acquisitions and the curated 13F
    CUSIP -> ticker overrides of `cusip_ticker_map`. The vendor series overrides are read by `predecessor_series.load_vendor_series`."""

    ratios: pd.DataFrame
    boundaries: pd.DataFrame
    classes: pd.DataFrame
    mergers: tuple[dict[str, Any], ...] = ()
    co_registrants: tuple[str, ...] = ()
    exchanges: tuple[ShareExchange, ...] = ()
    reverse_acquisitions: tuple[ReverseAcquisition, ...] = ()
    cusip_ticker_overrides: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> SecurityManual:
        """The manual of an absent config file."""
        return parse_security_manual({})


def _entries(blob: Mapping[str, Any], key: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """One manual list as a frame (dates parsed, CUSIPs upper-cased); an entry without a source is refused."""
    rows = list(blob.get(key) or [])
    for row in rows:
        if not str(row.get("source") or "").strip():
            raise SecurityManualError(f"security_master_manual.json: {key} entry {row} has no source (URL or accession)")
    frame = pd.DataFrame([{column: row.get(column) for column in columns} for row in rows], columns=list(columns))
    for column in ("valid_from", "valid_to"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    if "cusip" in frame.columns:
        frame["cusip"] = frame["cusip"].astype("string").str.upper().astype(object)
    return frame


def parse_security_manual(blob: Mapping[str, Any]) -> SecurityManual:
    """Validate and parse the manual config; every entry must cite a URL or an accession."""
    boundaries = _entries(blob, "market_boundaries", _BOUNDARY_COLUMNS)
    unknown = set(boundaries["role"].dropna()) - set(ROLES)
    if unknown:
        raise SecurityManualError(f"security_master_manual.json: unknown role(s) {sorted(unknown)}")
    boundaries["issuer_cik"] = pad_cik_series(boundaries["issuer_cik"]) if not boundaries.empty else boundaries["issuer_cik"]
    ratios = _entries(blob, "conversion_ratios", _RATIO_COLUMNS)
    ratios["ratio"] = pd.to_numeric(ratios["ratio"]).astype("float64")
    mergers = tuple(blob.get("merger_metadata") or [])
    for entry in mergers:
        if not str(entry.get("source") or "").strip():
            raise SecurityManualError(f"security_master_manual.json: merger entry {entry} has no source (URL or accession)")
    co_registrants = _entries(blob, "co_registrants", ("cik", "ticker", "source"))
    return SecurityManual(
        ratios=ratios,
        boundaries=boundaries,
        classes=_entries(blob, "class_overrides", _CLASS_COLUMNS),
        mergers=mergers,
        co_registrants=tuple(sorted(set(pad_cik_series(co_registrants["cik"])))) if not co_registrants.empty else (),
        exchanges=_exchanges(_entries(blob, "exchange_ratios", _EXCHANGE_COLUMNS)),
        reverse_acquisitions=_reverse_acquisitions(_entries(blob, "reverse_acquisitions", _REVERSE_COLUMNS)),
        cusip_ticker_overrides=_cusip_overrides(_entries(blob, "cusip_ticker_overrides", _OVERRIDE_COLUMNS)),
    )


def _cusip_overrides(rows: pd.DataFrame) -> dict[str, str]:
    """`{CUSIP-9: ticker}` of the `cusip_ticker_overrides` entries; a CUSIP listed twice is refused."""
    if rows["cusip"].duplicated().any():
        raise SecurityManualError(
            f"security_master_manual.json: cusip_ticker_overrides lists {sorted(set(rows.loc[rows['cusip'].duplicated(), 'cusip']))} twice"
        )
    return dict(zip(rows["cusip"].astype(str), rows["ticker"].astype(str), strict=True))


def _exchanges(rows: pd.DataFrame) -> tuple[ShareExchange, ...]:
    """The `exchange_ratios` entries; a ratio that is not a positive number is refused."""
    out = []
    for row in rows.itertuples(index=False):
        ratio = pd.to_numeric(row.ratio, errors="coerce")
        if pd.isna(ratio) or float(ratio) <= 0:
            raise SecurityManualError(f"security_master_manual.json: exchange ratio {row} is not a positive number")
        out.append(ShareExchange(normalise_ticker(row.ticker), pad_cik(row.predecessor_cik), pd.Timestamp(str(row.seam_date)), float(ratio)))
    return tuple(out)


def _reverse_acquisitions(rows: pd.DataFrame) -> tuple[ReverseAcquisition, ...]:
    """The `reverse_acquisitions` entries, CIKs padded and seams as Timestamps."""
    return tuple(
        ReverseAcquisition(
            ticker=normalise_ticker(row.ticker),
            seam_date=pd.Timestamp(str(row.seam_date)),
            accounting_acquirer_cik=pad_cik(row.accounting_acquirer_cik),
            legal_acquirer_cik=pad_cik(row.legal_acquirer_cik),
            source=str(row.source),
        )
        for row in rows.itertuples(index=False)
    )


def merger_boundaries(manual: SecurityManual) -> tuple[MergerBoundary, ...]:
    """The manual `merger_metadata` entries as `MergerBoundary` rows; a null `closing_time` leaves `after_close` unknown."""
    out = []
    for entry in manual.mergers:
        closing = str(entry.get("closing_time") or "").strip().lower()
        out.append(
            MergerBoundary(
                ticker=normalise_ticker(entry["ticker"]),
                seam_date=pd.Timestamp(entry["seam_date"]),
                after_close=("after" in closing) if closing else None,
                legal_acquirer_cik=pad_cik(entry["legal_acquirer_cik"]),
                accounting_predecessor_cik=pad_cik(entry["accounting_predecessor_cik"]),
                acquired_symbol=normalise_ticker(entry["acquired_symbol"]),
            )
        )
    return tuple(out)


def load_security_manual(config_dir: str | None = None) -> SecurityManual:
    """`configs/sec/security_master_manual.json`, parsed; an absent file is an empty config."""
    path = Path(resolve_config_dir(config_dir)) / SECURITY_MANUAL_FILE
    if not path.exists():
        return SecurityManual.empty()
    return parse_security_manual(json.loads(path.read_text(encoding="utf-8")))


def tape_intervals(lineage: pd.DataFrame) -> pd.DataFrame:
    """`[key, cik, valid_from, valid_to]` of the symbol rows a tape may read: not noise or conflict, not `dei` alone."""
    symbols = lineage[lineage["role"].eq(ROLE_SYMBOL)].copy()
    symbols["key"] = symbols["symbol"].map(squash)
    symbols["cik"] = pad_cik_series(symbols["cik"])
    sources = symbols["sources"].fillna("").astype(str).str.lower()
    tape_ok = ~symbols["status"].isin(("noise", "conflict")) & sources.ne(DEI_SOURCE) & symbols["key"].ne("")
    tape = symbols.loc[tape_ok, ["key", "cik", "valid_from", "valid_to"]].copy()
    tape["valid_from"] = pd.to_datetime(tape["valid_from"])
    tape["valid_to"] = pd.to_datetime(tape["valid_to"]).fillna(_FAR)
    return tape.sort_values(["key", "cik", "valid_from"], ignore_index=True)


def co_registrant_ciks(lineage: pd.DataFrame, evidence: pd.DataFrame | None, declared: Collection[str] = ()) -> frozenset[str]:
    """The extra CIKs the identity flags classify as co-registrants, except a former listing of the company, plus the
    `declared` ones (curated subsidiaries the flags never judge). A former listing held the company's ticker on its own,
    not alongside the home CIK (old GM as `GM` until 2009): a predecessor or an acquired constituent, not a subsidiary."""
    listed = frozenset(pad_cik(cik) for cik in declared)
    if evidence is None or evidence.empty or not {"role", "sources"} <= set(lineage.columns):  # membership rows only: nothing to judge
        return listed
    items = cik_items(flag_rows(lineage), cik_activity(evidence))
    tape = tape_intervals(lineage)
    out = set()
    for value in (item["ciks"] for item in items if item["kind"] == "co_registrant"):
        other, home = (str(value).split(",") + [""])[:2]
        if not _former_listing(tape, other, home):
            out.add(other)
    return frozenset(out) | listed


def _former_listing(tape: pd.DataFrame, other: str, home: str) -> bool:
    """`other` held a symbol `home` also holds, and their intervals overlap by at most the seam margin."""
    mine, theirs = tape[tape["cik"].eq(other)], tape[tape["cik"].eq(home)]
    shared = set(mine["key"]) & set(theirs["key"])
    if not shared:
        return False
    for key in shared:
        a, b = mine[mine["key"].eq(key)], theirs[theirs["key"].eq(key)]
        overlap = max(
            (
                min(x_to, y_to) - max(x_from, y_from)
                for x_from, x_to in zip(a["valid_from"], a["valid_to"], strict=True)
                for y_from, y_to in zip(b["valid_from"], b["valid_to"], strict=True)
            ),
            default=pd.Timedelta(0),
        )
        if overlap > _LISTING_OVERLAP:
            return False
    return True
