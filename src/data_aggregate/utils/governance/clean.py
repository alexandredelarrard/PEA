"""The single governance cleaning stage: every fill, carry, repair and reconciliation of the
raw proxy inputs runs here, in a fixed order, before any feature is built."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.typing import frame_column
from src.data_aggregate.utils.governance.def14a_impute import (
    CEO_COMP,
    FILL_BACKFILL,
    FILL_CARRY,
    FILL_CROSS_SOURCE,
    FILL_ESTIMATE,
    FILL_FILED,
    FILL_IDENTITY,
    FILL_INTERP,
    bridged_total,
    ensure_provenance,
    evidence_date_column,
    fill_kind_column,
    impute_def14a,
    impute_exec_comp,
    write_with_provenance,
)
from src.data_aggregate.utils.governance.director_comp import impute_director_comp
from src.data_aggregate.utils.governance.directors import (
    DERIVED_AGGREGATES,
    SOURCE_DERIVED,
    board_aggregates,
    fill_director_attributes,
    finalize_board_source,
    merge_board_aggregates,
)
from src.data_aggregate.utils.governance.names import ceo_identity_series, ceo_stint_ids, given_names
from src.data_aggregate.utils.governance.pay_features import SCT_GROWTH
from src.data_aggregate.utils.governance.staleness import source_date_column
from src.utils.names import person_key

_SHARES_OUTSTANDING = "sharesOutstandingPit"
_OWNERSHIP = "insider_ownership_pct"
_DIRECTOR_OFFICER = "director_officer"
#: a holder table naming fewer directors/officers than this is too partial to sum
_MIN_HOLDERS = 5
#: a `director_officer` holder row naming an entity (an institution, a vehicle, the "as a group"
#: total) is a 5% holder or a total mis-tagged by the extraction, never one person's holding
_ENTITY_HOLDER = re.compile(
    r"\b(?:inc|incorporated|corp|corporation|company|llc|l\.l\.c|lp|l\.p|llp|ltd|limited|plc|trust|funds?|foundation|bancorp|"
    r"partners|partnership|advisors|advisers|investors|associates|"
    r"bank of|vanguard|blackrock|state street|fmr|fidelity|t\.? ?rowe|mufg|mitsubishi ufj|primecap|"
    r"dodge & cox|invesco|geode|barclays|northern trust|jp ?morgan|"
    r"goldman,? sachs|morgan stanley|alliancebernstein|norges|berkshire hathaway)\b",
    re.IGNORECASE,
)
#: entity words that also appear in an executive's title ("..., Data Center Group"), so they are
#: tested on the holder's name before its first comma only
_ENTITY_NAME_WORD = re.compile(r"\b(?:group|holdings?|capital|management|investments?|securities)\b", re.IGNORECASE)
#: a block the board holds as a body, not any director in person (MetLife's Policyholder Trust)
_BOARD_HOLDER = re.compile(r"^(?:the\s+)?(?:company\s+)?board of directors\b|\bnot in (?:\w+\W+){0,4}individual capacit", re.IGNORECASE)
#: an estimate above this multiple of the ticker's largest filed value is refused
_MAX_FILED_MULTIPLE = 3.0
_SOP_SUPPORT = "say_on_pay_support_pct"
_SAY_ON_PAY = "say_on_pay"
#: an 8-K say-on-pay tally filed this many days before a proxy already covers that proxy's vote
_SOP_WINDOW_DAYS = 450
VOTE_SOURCE_8K = "8k"
VOTE_SOURCE_PROXY = "proxy"
_TOTAL = "ceo_total_comp"
#: stats stage of the historical pay backfill, and the warning its counts are logged under
PAY_BACKFILL_STAGE = "pay_backfill"
PAY_BACKFILL_WARNING = (
    "historical CEO pay cells filled from LATER filings (accepted lookahead): %s. A value missing today in live data is not filled the same way."
)
#: parent CEO pay column -> the same component in `def14a_executive_comp`
_SCT_COLUMN: dict[str, str] = {
    "ceo_salary": "salary",
    "ceo_bonus": "bonus",
    "ceo_stock_awards": "stock_awards",
    "ceo_option_awards": "option_awards",
    "ceo_non_equity_incentive": "non_equity_incentive",
    "ceo_all_other_comp": "other_compensation",
    _TOTAL: "total",
}
#: the most consecutive missing proxies of one CEO stint that interpolation may bridge
_INTERP_MAX_GAP = 2


@dataclass(frozen=True)
class GovernanceInputs:
    """The governance source frames; `cleaned` is True once the stage has run on them."""

    def14a: pd.DataFrame | None
    exec_comp: pd.DataFrame | None = None
    directors: pd.DataFrame | None = None
    director_comp: pd.DataFrame | None = None
    votes: pd.DataFrame | None = None
    ownership: pd.DataFrame | None = None
    cleaned: bool = False


def _present(df: pd.DataFrame | None) -> bool:
    return df is not None and not df.empty


def _record(stats: dict[str, int], stage: str, part: dict[str, int]) -> None:
    stats.update({f"{stage}|{k}": int(v) for k, v in part.items()})


def clean_governance_inputs(raw: GovernanceInputs, fundamentals: pd.DataFrame | None) -> tuple[GovernanceInputs, dict[str, int]]:
    """Clean the raw governance inputs. Returns `(cleaned bundle, stats keyed "<stage>|<rule>")`.

    Order: child fills, board merge, `impute_def14a`, board provenance, ownership basis repair,
    economic ownership, ownership estimate, same-filing SCT pay growth, historical CEO pay
    backfill, say-on-pay votes fallback. The `PAY_BACKFILL_STAGE` stats are lookahead fills that
    the caller reports as one warning (`PAY_BACKFILL_WARNING`). A bundle already cleaned
    is returned unchanged with empty stats.
    """
    if raw.cleaned:
        return raw, {}
    stats: dict[str, int] = {}

    exec_comp, directors, director_comp = raw.exec_comp, raw.directors, raw.director_comp
    if _present(exec_comp):
        exec_comp, part = impute_exec_comp(cast(pd.DataFrame, exec_comp))
        _record(stats, "exec_comp", part)
    if _present(directors):
        directors, part = fill_director_attributes(cast(pd.DataFrame, directors))
        _record(stats, "directors", part)
    if _present(director_comp):
        director_comp, part = impute_director_comp(cast(pd.DataFrame, director_comp))
        _record(stats, "director_comp", part)

    df = raw.def14a
    if _present(df):
        frame = cast(pd.DataFrame, df)
        if _present(directors):
            frame, part = merge_board_aggregates(frame, board_aggregates(cast(pd.DataFrame, directors)))
            _record(stats, "board_merge", part)
            _board_fill_kinds(frame)
        frame, part = impute_def14a(frame)
        _record(stats, "impute_def14a", part)
        frame, part = finalize_board_source(frame)
        _record(stats, "board_source", part)
        if "as_of" in frame.columns:
            # Blank per-class ownership before computed economic ownership fills the gap.
            tally: dict[str, int] = {}
            shares_out = pit_shares_outstanding(frame, fundamentals, tally)
            frame = repair_ownership_basis(frame, tally)
            frame = economic_ownership(frame, shares_out, tally)
            _record(stats, "ownership", tally)
            if _present(raw.ownership):
                tally = {}
                frame = estimate_insider_ownership(frame, cast(pd.DataFrame, raw.ownership), shares_out, tally)
                _record(stats, "ownership_estimate", tally)
        if _present(exec_comp):
            sct = sct_frames(frame, cast(pd.DataFrame, exec_comp))
            tally = {}
            frame = sct_pay_growth(frame, sct, tally)
            _record(stats, "sct_growth", tally)
            tally = {}
            frame = backfill_ceo_pay(frame, sct, tally)
            _record(stats, PAY_BACKFILL_STAGE, tally)
        df = frame

    votes = raw.votes
    if _present(raw.def14a):
        vote_tally: dict[str, int] = {}
        votes = say_on_pay_fallback(cast(pd.DataFrame, raw.def14a), votes, vote_tally)
        _record(stats, "votes_fallback", vote_tally)

    cleaned = replace(raw, def14a=df, exec_comp=exec_comp, directors=directors, director_comp=director_comp, votes=votes, cleaned=True)
    return cleaned, stats


def _board_fill_kinds(frame: pd.DataFrame) -> None:
    """Board averages derived from the director rows are a cross-source fill."""
    for field in DERIVED_AGGREGATES:
        if field in frame.columns and f"{field}_source" in frame.columns:
            ensure_provenance(frame, field)
            frame.loc[frame[f"{field}_source"] == SOURCE_DERIVED, fill_kind_column(field)] = FILL_CROSS_SOURCE


def _pit_shares_outstanding(def14a_hist: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series | None:
    """Point-in-time shares outstanding at each proxy's `as_of`, aligned on the proxy index."""
    # merge_asof requires identical datetime resolution across the two sources.
    left = pd.DataFrame({"ticker": def14a_hist["ticker"], "_as_of": pd.to_datetime(def14a_hist["as_of"], errors="coerce").astype("datetime64[ns]")})
    right = cast(
        pd.DataFrame,
        fundamentals[["ticker", "as_of", _SHARES_OUTSTANDING]]
        .copy()
        .assign(_as_of=lambda d: pd.to_datetime(d["as_of"], errors="coerce").astype("datetime64[ns]"))
        .drop(columns="as_of")
        .dropna(subset=["_as_of"]),
    )
    right = cast(pd.DataFrame, right.loc[frame_column(right, _SHARES_OUTSTANDING) > 0])
    if right.empty:
        return None

    # Preserve row identity because merge_asof resets the index.
    left["_key"] = range(len(left))
    dated_left = cast(pd.DataFrame, left.dropna(subset=["_as_of"])).sort_values("_as_of")
    merged = pd.merge_asof(dated_left, right.sort_values("_as_of"), on="_as_of", by="ticker", direction="backward")
    shares_out = cast(pd.Series, pd.to_numeric(frame_column(merged, _SHARES_OUTSTANDING), errors="coerce"))
    by_key = pd.Series(shares_out.to_numpy(), index=frame_column(merged, "_key").to_numpy())
    aligned = by_key.reindex(frame_column(left, "_key").to_numpy())
    aligned.index = def14a_hist.index
    return aligned


def pit_shares_outstanding(def14a_hist: pd.DataFrame, fundamentals: pd.DataFrame | None, tally: dict[str, int] | None = None) -> pd.Series | None:
    """Point-in-time shares outstanding at each proxy's `as_of` (the ownership denominator), or
    None when the fundamentals cannot supply it."""
    if fundamentals is None or fundamentals.empty or not {"ticker", "as_of"}.issubset(def14a_hist.columns):
        return None
    if not {"ticker", "as_of", _SHARES_OUTSTANDING}.issubset(fundamentals.columns):
        if tally is not None and "insider_shares" in def14a_hist.columns:
            tally[f"⚠ economic ownership not computed: {_SHARES_OUTSTANDING} absent"] = 1
        return None
    return _pit_shares_outstanding(def14a_hist, fundamentals)


def _proxy_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["as_of"], errors="coerce")


def _carried_total(df: pd.DataFrame) -> pd.Series:
    """True where `ceo_total_comp` is an identity built with a leg older than its own proxy (a
    carried salary): an exact SCT total, or the identity re-derived from exact legs, replaces it."""
    kind_col, source_col = fill_kind_column(_TOTAL), source_date_column(_TOTAL)
    if kind_col not in df.columns or source_col not in df.columns:
        return pd.Series(False, index=df.index)
    return (df[kind_col] == FILL_IDENTITY) & (pd.to_datetime(df[source_col], errors="coerce") < _proxy_dates(df))


def _clear_comp_imputed(df: pd.DataFrame, mask: pd.Series) -> None:
    """A total replaced by exact values no longer uses a carried component."""
    if "comp_imputed" in df.columns:
        df.loc[mask, "comp_imputed"] = 0.0


# The denominator must be point-in-time at the proxy date.
def economic_ownership(def14a_hist: pd.DataFrame, shares_out: pd.Series | None, tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Replace ownership with same-filing shares / `pit_shares_outstanding` where both are available."""
    if shares_out is None or not {"ticker", "as_of", "insider_shares"}.issubset(def14a_hist.columns):
        return def14a_hist

    out = def14a_hist.copy()
    insider = cast(pd.Series, pd.to_numeric(frame_column(out, "insider_shares"), errors="coerce"))
    computed = insider / shares_out
    # a share of a whole: anything outside (0, 1] means the two counts are on different bases
    # (one class's shares over the total, or a count in thousands) and is not usable
    filled = computed.where((shares_out > 0) & (computed > 0) & (computed <= 1.0))

    # Economic ownership cannot exceed the disclosed voting share on a comparable basis.
    if "insider_voting_pct" in out.columns:
        vote = cast(pd.Series, pd.to_numeric(frame_column(out, "insider_voting_pct"), errors="coerce"))
        bad = filled.notna() & vote.notna() & (filled > vote + 1e-9)
        if tally is not None and int(bad.sum()):
            tally["insider_ownership_pct: computed value REJECTED (exceeds voting power)"] = int(bad.sum())
        filled = filled.where(~bad)

    disclosed = cast(pd.Series, pd.to_numeric(frame_column(out, _OWNERSHIP), errors="coerce"))

    # Filed evidence wins; the computed percentage only fills a missing disclosure.
    fills = filled.notna() & disclosed.isna()
    n = int(fills.sum())
    if n:
        out[_OWNERSHIP] = disclosed
        # dated at the proxy that disclosed the share count
        write_with_provenance(out, fills, _OWNERSHIP, filled, FILL_CROSS_SOURCE, _proxy_dates(out))
        if tally is not None:
            tally["insider_ownership_pct: COMPUTED (no disclosed combined percentage)"] = n
    if tally is not None:
        kept = int((filled.notna() & disclosed.notna()).sum())
        if kept:
            tally["insider_ownership_pct: disclosed value PREFERRED over the computed one"] = kept
    return out


def estimate_insider_ownership(
    def14a_hist: pd.DataFrame, ownership: pd.DataFrame, shares_out: pd.Series | None, tally: dict[str, int] | None = None
) -> pd.DataFrame:
    """Fill a still-missing ownership share with the filing's summed director/officer holdings
    over `pit_shares_outstanding`, on tickers never flagged dual-class.

    Holder rows that name an entity are not summed, and an estimate above
    `_MAX_FILED_MULTIPLE` times the ticker's largest filed value is refused. Filed, computed and
    carried values are never overwritten. "Ever dual" and the filed ceiling read later proxies
    too; that can only refuse a fill.
    """
    if shares_out is None or not {"ticker", "as_of", "accession_number", _OWNERSHIP}.issubset(def14a_hist.columns):
        return def14a_hist

    named = ownership.loc[(ownership["holder_type"] == _DIRECTOR_OFFICER) & ownership["shares"].notna()]
    entity = _entity_holder(named["holder_name"]) if "holder_name" in named.columns else pd.Series(False, index=named.index)
    named = named.loc[~entity]
    holders = named.groupby(["ticker", "accession_number"])["shares"].agg(["sum", "count"])
    keys = pd.MultiIndex.from_frame(def14a_hist[["ticker", "accession_number"]])
    summed = pd.Series(holders["sum"].reindex(keys).to_numpy(), index=def14a_hist.index)
    n_holders = pd.Series(holders["count"].reindex(keys).fillna(0).to_numpy(), index=def14a_hist.index)
    estimate = summed / shares_out

    dual_raw = def14a_hist["dual_class_shares"] if "dual_class_shares" in def14a_hist.columns else pd.Series(np.nan, index=def14a_hist.index)
    dual = cast(pd.Series, pd.to_numeric(dual_raw, errors="coerce")) > 0
    ever_dual = def14a_hist["ticker"].map(dual.groupby(def14a_hist["ticker"]).max()).fillna(False).astype(bool)
    in_bound = (estimate > 0) & (estimate <= 1.0)
    # Like "ever dual", the ticker's filed history can only refuse an estimate.
    pct = cast(pd.Series, pd.to_numeric(def14a_hist[_OWNERSHIP], errors="coerce"))
    kind_col = fill_kind_column(_OWNERSHIP)
    filed = pct.where(def14a_hist[kind_col] == FILL_FILED) if kind_col in def14a_hist.columns else pct
    ceiling = def14a_hist["ticker"].map(filed.groupby(def14a_hist["ticker"]).max()) * _MAX_FILED_MULTIPLE
    implausible = estimate > ceiling
    usable = ~ever_dual & (n_holders >= _MIN_HOLDERS) & in_bound & ~implausible

    out = def14a_hist.copy()
    candidate = pct.isna() & summed.notna()
    fills = candidate & usable
    if int(fills.sum()):
        write_with_provenance(out, fills, _OWNERSHIP, estimate, FILL_ESTIMATE, _proxy_dates(out))
    if tally is not None:
        _tally_estimate(tally, def14a_hist, candidate, fills, ever_dual, n_holders, estimate, usable, implausible)
        if int(entity.sum()):
            tally["insider_ownership_pct: director/officer holder rows EXCLUDED (entity name)"] = int(entity.sum())
    return out


def _entity_holder(names: pd.Series) -> pd.Series:
    """True where a holder cell names an entity, a total or a board-held block rather than a person."""
    text = names.fillna("").astype(str)
    return text.str.contains(_ENTITY_HOLDER) | text.str.split(",").str[0].str.contains(_ENTITY_NAME_WORD) | text.str.contains(_BOARD_HOLDER)


def _tally_estimate(
    tally: dict[str, int],
    before: pd.DataFrame,
    candidate: pd.Series,
    fills: pd.Series,
    ever_dual: pd.Series,
    n_holders: pd.Series,
    estimate: pd.Series,
    usable: pd.Series,
    implausible: pd.Series,
) -> None:
    """Count each estimate outcome, in rule order, plus the carries a same-filing estimate could replace."""
    eligible = candidate & ~ever_dual & (n_holders >= _MIN_HOLDERS)
    kind_col = fill_kind_column(_OWNERSHIP)
    carried = before[kind_col] == FILL_CARRY if kind_col in before.columns else pd.Series(False, index=before.index)
    counts = {
        "insider_ownership_pct: ESTIMATED from director/officer holder rows": fills,
        "insider_ownership_pct: estimate REFUSED (ticker ever flagged dual-class)": candidate & ever_dual,
        f"insider_ownership_pct: estimate REFUSED (< {_MIN_HOLDERS} director/officer holders)": candidate & ~ever_dual & (n_holders < _MIN_HOLDERS),
        "insider_ownership_pct: estimate REFUSED (no point-in-time shares outstanding)": eligible & estimate.isna(),
        "insider_ownership_pct: estimate REJECTED (outside (0, 1])": eligible & estimate.notna() & ~((estimate > 0) & (estimate <= 1.0)),
        f"insider_ownership_pct: estimate REJECTED (> {_MAX_FILED_MULTIPLE:g}x the ticker's largest filed value)": eligible
        & (estimate > 0)
        & (estimate <= 1.0)
        & implausible,
        "insider_ownership_pct: carried value KEPT over a same-filing estimate": carried & usable,
    }
    tally.update({key: int(mask.sum()) for key, mask in counts.items() if int(mask.sum())})


def repair_ownership_basis(def14a_hist: pd.DataFrame, tally: dict[str, int] | None = None) -> pd.DataFrame:
    """Blank dual-class ownership percentages that use a per-class denominator."""
    need = {"insider_ownership_pct", "insider_voting_pct", "dual_class_shares"}
    if not need.issubset(def14a_hist.columns):
        return def14a_hist

    out = def14a_hist.copy()
    dual_values = cast(pd.Series, pd.to_numeric(frame_column(out, "dual_class_shares"), errors="coerce"))
    dual = dual_values > 0
    if "ticker" in out.columns:
        # One missed flag can accompany the bad value, so corroborate structure across history.
        dual = dual | out["ticker"].map(dual.groupby(out["ticker"]).max()).fillna(False).astype(bool)
    vote = pd.to_numeric(out["insider_voting_pct"], errors="coerce")
    own = pd.to_numeric(out["insider_ownership_pct"], errors="coerce")
    per_class = dual & vote.notna() & own.notna() & (own > vote + 1e-9)

    n = int(per_class.sum())
    if n:
        write_with_provenance(out, per_class, "insider_ownership_pct", None, None, None)
        if "ceo_ownership_pct" in out.columns:
            # the CEO leg was read off the SAME table and the same columns, so it inherits the
            # basis error on those filings whether or not its own pair is comparable
            write_with_provenance(out, per_class, "ceo_ownership_pct", None, None, None)
        if tally is not None:
            tally["insider_ownership_pct: blanked (per-class basis, own > vote)"] = n
            tally["insider_ownership_pct: blanked (per-class basis) — tickers"] = (
                int(out.loc[per_class, "ticker"].nunique()) if "ticker" in out.columns else 0
            )
    return out


def _given_parts(names: pd.Series, prefix: str = "") -> pd.DataFrame:
    """`<prefix>first` and `<prefix>initials` of each name cell (None when it names no given name)."""
    given = names.astype(object).map(given_names)
    return pd.DataFrame(
        {f"{prefix}first": given.map(lambda g: g[0] if g else None), f"{prefix}initials": given.map(lambda g: g[1] if g else None)},
        index=names.index,
    )


def _named_ceo(hit: pd.DataFrame, by: list[str]) -> pd.Series:
    """True on the SCT rows of `hit` that belong to the proxy's named CEO.

    Where relatives share the CEO's `person_key` within one `by` group, only the rows whose first
    name equals the CEO cell's are kept, else the rows whose initials single out one name; else none.
    """
    keys = [hit[c] for c in by]
    sig = (hit["first"] + "|" + hit["initials"]).where(hit["first"].notna())
    shared = sig.groupby(keys).transform("nunique") > 1
    first_eq = (hit["first"] == hit["ceo_first"]) & hit["first"].notna()
    init_eq = (hit["initials"] == hit["ceo_initials"]) & hit["initials"].notna()
    any_first = first_eq.groupby(keys).transform("any")
    init_names = sig.where(init_eq).groupby(keys).transform("nunique")
    keep = first_eq.where(any_first, init_eq & (init_names == 1))
    return ~shared | keep.astype(bool)


@dataclass(frozen=True)
class SctFrames:
    """The SCT lines and the proxies' named-CEO rows, built once for the pay growth and backfill.

    `rows`: one line per `def14a_executive_comp` row (ticker, accession, `_later` filing date,
    person key, given-name parts, fiscal year, `stated` total, the `_SCT_COLUMN` values present).
    `ceo`: one row per proxy in frame order (`_row`), with the CEO's key and given-name parts,
    `fy` (the filing's latest SCT year) and `fiscal_year` (`fy`, else the extracted year)."""

    rows: pd.DataFrame
    ceo: pd.DataFrame


def sct_frames(def14a: pd.DataFrame, exec_comp: pd.DataFrame) -> SctFrames:
    """Build the `SctFrames` of one cleaning pass (exec comp already imputed, so `total_imputed` is set)."""
    present = {c: c for c in ("ticker", "accession_number") if c in exec_comp.columns}
    rows = pd.DataFrame({k: exec_comp[c] for k, c in present.items()}, index=exec_comp.index)
    if "ticker" in rows.columns:
        rows["ticker"] = rows["ticker"].astype(str)
    if "as_of" in exec_comp.columns:
        rows["_later"] = pd.to_datetime(exec_comp["as_of"], errors="coerce")
    if "name" in exec_comp.columns:
        rows["person"] = exec_comp["name"].astype(object).map(person_key)
        rows = pd.concat([rows, _given_parts(exec_comp["name"])], axis=1)
    if "fiscal_year" in exec_comp.columns:
        rows["fiscal_year"] = pd.to_numeric(exec_comp["fiscal_year"], errors="coerce")
    imputed = exec_comp["total_imputed"] if "total_imputed" in exec_comp.columns else pd.Series(0.0, index=exec_comp.index)
    rows["stated"] = pd.to_numeric(imputed, errors="coerce").fillna(0.0).to_numpy() == 0.0
    for c in sorted(set(_SCT_COLUMN.values()) & set(exec_comp.columns)):
        rows[c] = pd.to_numeric(exec_comp[c], errors="coerce")

    names = def14a["ceo_name_proxy"] if "ceo_name_proxy" in def14a.columns else pd.Series(None, index=def14a.index, dtype="object")
    ceo = pd.concat(
        [
            pd.DataFrame(
                {
                    "_row": np.arange(len(def14a)),
                    "ticker": def14a["ticker"].astype(str).to_numpy() if "ticker" in def14a.columns else None,
                    "accession_number": def14a["accession_number"].to_numpy() if "accession_number" in def14a.columns else None,
                    "person": ceo_identity_series(names).to_numpy(),
                    "_as_of": _proxy_dates(def14a).to_numpy() if "as_of" in def14a.columns else pd.NaT,
                }
            ),
            _given_parts(names, "ceo_").reset_index(drop=True),
        ],
        axis=1,
    )
    if {"accession_number", "fiscal_year"}.issubset(rows.columns):
        ceo["fy"] = ceo["accession_number"].map(rows.groupby("accession_number")["fiscal_year"].max())
    else:
        ceo["fy"] = np.nan
    ceo["fiscal_year"] = ceo["fy"]
    if "fiscal_year_extract" in def14a.columns:
        ceo["fiscal_year"] = ceo["fy"].fillna(pd.Series(pd.to_numeric(def14a["fiscal_year_extract"], errors="coerce").to_numpy(), index=ceo.index))
    return SctFrames(rows=rows, ceo=ceo)


def _ceo_sct_totals(sct: SctFrames, index: pd.Index) -> tuple[pd.DataFrame, int]:
    """The named CEO's positive SCT totals per proxy row, from the proxy's own filing:
    columns `fy` (the filing's latest fiscal year), `t1` (CEO total for `fy`), `t0` (for
    `fy - 1`), `t2` (for `fy - 2`) and `s1` / `s0` (True where `t1` / `t0` is filer-stated
    rather than a sum of components), on `index`; plus the number of proxies whose CEO cannot
    be told apart from a relative sharing the key."""
    child = sct.rows[["accession_number", "person", "first", "initials", "fiscal_year", "total", "stated"]].dropna(
        subset=["accession_number", "fiscal_year"]
    )
    ceo = sct.ceo[["_row", "accession_number", "person", "ceo_first", "ceo_initials", "fy"]]
    hit = ceo.dropna(subset=["person", "fy"]).merge(child, on=["accession_number", "person"])
    named = _named_ceo(hit, ["_row"])
    ambiguous = int((~named.groupby(hit["_row"]).any()).sum())
    paid = (
        hit.loc[named & (hit["total"] > 0)]
        .groupby(["_row", "fy", "fiscal_year"], as_index=False)
        .agg(total=("total", "max"), stated=("stated", "min"))
    )
    positions = np.arange(len(index))
    out = pd.DataFrame({"fy": sct.ceo["fy"].to_numpy()}, index=index)
    for col, lag in (("t1", 0), ("t0", 1), ("t2", 2)):
        rows = paid.loc[paid["fiscal_year"] == paid["fy"] - lag].set_index("_row")
        out[col] = rows["total"].reindex(positions).to_numpy()
        if lag < 2:
            out[f"s{1 - lag}"] = rows["stated"].reindex(positions).fillna(False).astype(bool).to_numpy()
    return out, ambiguous


def sct_pay_growth(def14a_hist: pd.DataFrame, sct: SctFrames, tally: dict[str, int]) -> pd.DataFrame:
    """Write `ceo_comp_growth_sct` = log(total FY / total FY-1) of the named CEO from the proxy's
    own Summary Compensation Table, and fill a NULL `ceo_total_comp` from the same-FY SCT row.

    The CEO is matched on `ceo_identity(ceo_name_proxy) == person_key(name)`; FY is the filing's
    latest SCT year, FY-1 must be disclosed (a two-year step is refused) and both totals must be
    filer-stated. The total fill also takes a total summed from the SCT components.
    """
    if not {"accession_number", "ceo_name_proxy", "as_of"}.issubset(def14a_hist.columns):
        return def14a_hist
    if not {"accession_number", "person", "fiscal_year", "total"}.issubset(sct.rows.columns):
        return def14a_hist
    totals, ambiguous = _ceo_sct_totals(sct, def14a_hist.index)
    out = def14a_hist.copy()
    disclosed = totals["t1"].notna() & totals["t0"].notna()
    pair = disclosed & totals["s1"] & totals["s0"]
    out[SCT_GROWTH] = np.log(totals["t1"] / totals["t0"]).where(pair)

    total = (
        cast(pd.Series, pd.to_numeric(frame_column(out, _TOTAL), errors="coerce")) if _TOTAL in out.columns else pd.Series(np.nan, index=out.index)
    )
    carried = _carried_total(out)
    fill = (total.isna() | carried) & totals["t1"].notna()
    if int(fill.sum()):
        out[_TOTAL] = total
        # same filing, so the value is as old as the proxy itself
        write_with_provenance(out, fill, _TOTAL, totals["t1"], FILL_IDENTITY, _proxy_dates(out))
        _clear_comp_imputed(out, fill & carried)

    tally["growth from the filing's own SCT (FY vs FY-1, named CEO)"] = int(pair.sum())
    tally["growth refused: an SCT total is a sum of components, not filer-stated"] = int((disclosed & ~pair).sum())
    tally["growth refused: CEO has no FY-1 SCT row (FY-2 only)"] = int((totals["t1"].notna() & totals["t0"].isna() & totals["t2"].notna()).sum())
    tally["growth refused: CEO has FY only"] = int((totals["t1"].notna() & totals["t0"].isna() & totals["t2"].isna()).sum())
    tally["CEO not matched in the filing's own SCT (filing has SCT rows)"] = int((totals["fy"].notna() & totals["t1"].isna()).sum())
    tally["ceo_total_comp filled from the filing's own SCT row"] = int((fill & ~carried).sum())
    tally["ceo_total_comp built with a carried leg REPLACED by the filing's own SCT row"] = int((fill & carried).sum())
    tally["CEO ambiguous in the filing's SCT (relatives share the key)"] = ambiguous
    return out


def later_sct_values(sct: SctFrames, col: str, index: pd.Index) -> tuple[pd.Series, pd.Series]:
    """`(value, filing date)` of `col` for each proxy row's CEO and fiscal year, as disclosed by
    the EARLIEST later proxy of the same ticker whose SCT carries it, on `index`. NaN where none
    does. A total is taken only where the filer stated it; relatives sharing the key are told
    apart as in `_named_ceo`."""
    value_col = _SCT_COLUMN[col]
    child = sct.rows[["ticker", "accession_number", "person", "first", "initials", "fiscal_year", "_later", "stated", value_col]].dropna(
        subset=["ticker", "accession_number", "person", "fiscal_year", "_later"]
    )
    rows = sct.ceo[["_row", "ticker", "person", "fiscal_year", "_as_of", "ceo_first", "ceo_initials"]].dropna(
        subset=["ticker", "person", "fiscal_year", "_as_of"]
    )
    hit = rows.merge(child, on=["ticker", "person", "fiscal_year"])
    hit = hit.loc[hit["_later"] > hit["_as_of"]]
    hit = hit.loc[_named_ceo(hit, ["_row", "accession_number"])]
    usable = (hit[value_col] > 0) & hit["stated"] if col == _TOTAL else hit[value_col] >= 0
    hit = hit.loc[usable].sort_values(["_row", "_later"]).drop_duplicates("_row").set_index("_row")
    positions = np.arange(len(index))
    value = pd.Series(hit[value_col].reindex(positions).to_numpy(), index=index)
    later = pd.Series(hit["_later"].reindex(positions).to_numpy(), index=index)
    return value, later


def interpolate_in_stint(values: pd.Series, as_of: pd.Series, stint: pd.Series, evidence: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """`(value, evidence date, refused)` for each missing cell bridged by at most `_INTERP_MAX_GAP`
    proxies between two known values of the same stint: log-linear in time, 0 between two zeros.
    `refused` marks a bridgeable cell between one zero and one positive value. Inputs are sorted
    chronologically within ticker; the evidence date is the later neighbour's."""
    on = stint.notna()
    v, t, st, ev = values[on], as_of[on], stint[on], evidence[on]
    known = v.notna()

    def fwd(x: pd.Series) -> pd.Series:
        return x.where(known).groupby(st).ffill()

    def bwd(x: pd.Series) -> pd.Series:
        return x.where(known).groupby(st).bfill()

    pos = st.groupby(st).cumcount().astype("float64")
    prev_v, next_v = fwd(v), bwd(v)
    middle = ~known & prev_v.notna() & next_v.notna() & (bwd(pos) - fwd(pos) - 1 <= _INTERP_MAX_GAP)
    prev_t, next_t = fwd(t), bwd(t)
    w = (t - prev_t) / (next_t - prev_t)
    positive = (prev_v > 0) & (next_v > 0)
    zeros = (prev_v == 0) & (next_v == 0)
    lo, hi = np.log(prev_v.where(positive)), np.log(next_v.where(positive))
    value = cast(pd.Series, np.exp(lo + w * (hi - lo))).where(positive, 0.0).where(middle & (positive | zeros))
    later = bwd(ev).where(value.notna())
    refused = middle & ~positive & ~zeros
    return value.reindex(values.index), later.reindex(values.index), refused.reindex(values.index, fill_value=False).astype(bool)


def backfill_ceo_pay(def14a_hist: pd.DataFrame, sct: SctFrames, tally: dict[str, int]) -> pd.DataFrame:
    """Fill historical CEO pay holes from LATER filings (accepted lookahead): first the exact value
    a later proxy's SCT discloses for the same CEO and fiscal year (kind `backfill`), then a
    log-linear bridge inside one CEO stint (kind `interp`). Components first; the total then
    takes the backfill, the six-component identity and the bridge.

    A row with no later value is left to the past-only carry. Each written cell is dated at its
    own `as_of`, with the later evidence date in `<col>_evidence_as_of`.
    """
    need = {"ticker", "as_of", "accession_number", "ceo_name_proxy"}
    if not need.issubset(def14a_hist.columns) or not {"ticker", "accession_number", "_later", "person", "fiscal_year"}.issubset(sct.rows.columns):
        return def14a_hist
    out = def14a_hist.copy()
    out["_t"] = _proxy_dates(out)
    order = out.sort_values(["ticker", "_t"], kind="stable").index
    view = out.loc[order]
    stint = ceo_stint_ids(view["ceo_name_proxy"], view["ticker"], view["_t"])
    components = [c for c in CEO_COMP if c in out.columns]
    carried = _carried_total(out)
    replaced_leg = pd.Series(False, index=out.index)
    for col in components:
        replaced_leg |= _backfill_then_bridge(out, sct, col, order, stint, tally, backfill=_SCT_COLUMN[col] in sct.rows.columns)
    if _TOTAL in out.columns:
        replaced = _backfill_then_bridge(out, sct, _TOTAL, order, stint, tally, backfill="total" in sct.rows.columns, bridge=False, also=carried)
        rederive = carried & replaced_leg & ~replaced
        if len(components) == len(CEO_COMP):
            _total_identity(out, components, tally, rederive)
        _backfill_then_bridge(out, sct, _TOTAL, order, stint, tally, backfill=False)
        _clear_comp_imputed(out, replaced | rederive)
        bridged = bridged_total(out)
        out["comp_imputed"] = (pd.to_numeric(out.get("comp_imputed", 0.0), errors="coerce").fillna(0.0).astype(bool) | bridged).astype("float64")
    return out.drop(columns="_t")


def _backfill_then_bridge(
    out: pd.DataFrame,
    sct: SctFrames,
    col: str,
    order: pd.Index,
    stint: pd.Series,
    tally: dict[str, int],
    backfill: bool = True,
    bridge: bool = True,
    also: pd.Series | None = None,
) -> pd.Series:
    """Write the later-SCT value into the NULL and carried cells of `col` (and the `also` cells),
    then the in-stint bridge into the NULL cells. Returns the valued cells the backfill replaced."""
    out[col] = pd.to_numeric(out[col], errors="coerce")
    replaced = pd.Series(False, index=out.index)
    if backfill:
        value, later = later_sct_values(sct, col, out.index)
        kind_col = fill_kind_column(col)
        stale = (out[kind_col] == FILL_CARRY) if kind_col in out.columns else pd.Series(False, index=out.index)
        if also is not None:
            stale |= also
        fill = (out[col].isna() | stale) & value.notna()
        replaced = fill & out[col].notna()
        if int(fill.sum()):
            write_with_provenance(out, fill, col, value, FILL_BACKFILL, out["_t"], later)
            tally[f"backfill (later SCT, same CEO and FY): {col}"] = int((fill & ~replaced).sum())
        if int(replaced.sum()):
            tally[f"backfill REPLACED a carried value (later SCT, same CEO and FY): {col}"] = int(replaced.sum())
    if not bridge:
        return replaced
    ev_col = evidence_date_column(col)
    evidence = pd.to_datetime(out[ev_col], errors="coerce").fillna(out["_t"]) if ev_col in out.columns else out["_t"]
    value, later, refused = interpolate_in_stint(out.loc[order, col], out.loc[order, "_t"], stint, evidence.loc[order])
    value, later = value.reindex(out.index), later.reindex(out.index)
    fill = out[col].isna() & value.notna()
    if int(fill.sum()):
        write_with_provenance(out, fill, col, value, FILL_INTERP, out["_t"], later)
        tally[f"interp (same stint, <= {_INTERP_MAX_GAP} missing proxies): {col}"] = int(fill.sum())
    if int(refused.sum()):
        tally[f"interp refused (one zero and one positive neighbour): {col}"] = int(refused.sum())
    return replaced


def _total_identity(out: pd.DataFrame, components: list[str], tally: dict[str, int], rederive: pd.Series) -> None:
    """A NULL total, or a `rederive` one whose carried leg was replaced, becomes the sum of its
    six components, with their latest evidence date."""
    legs = out[components].apply(pd.to_numeric, errors="coerce")
    fill = (out[_TOTAL].isna() | rederive) & legs.notna().all(axis=1)
    if not int(fill.sum()):
        return
    dates = [pd.to_datetime(out[evidence_date_column(c)], errors="coerce") for c in components if evidence_date_column(c) in out.columns]
    later = pd.concat(dates, axis=1).max(axis=1) if dates else pd.Series(pd.NaT, index=out.index)
    bridged = pd.concat([out[fill_kind_column(c)] == FILL_INTERP for c in components if fill_kind_column(c) in out.columns], axis=1).any(axis=1)
    for kind, mask in ((FILL_IDENTITY, fill & ~bridged), (FILL_INTERP, fill & bridged)):
        if int(mask.sum()):
            write_with_provenance(out, mask, _TOTAL, legs.sum(axis=1), kind, out["_t"], later)
            tally[f"{kind}: ceo_total_comp = sum of components, {'one bridged' if kind == FILL_INTERP else 'none bridged'}"] = int(mask.sum())


def say_on_pay_fallback(def14a_raw: pd.DataFrame, votes: pd.DataFrame | None, tally: dict[str, int]) -> pd.DataFrame | None:
    """Append one synthetic say-on-pay vote row per proxy whose FILED support has no 8-K tally
    in the prior `_SOP_WINDOW_DAYS` days.

    Reads the raw (pre-impute) proxy frame, so a carried value never becomes a vote. The row
    is for = support, against = 1 - support, abstain 0, dated at the proxy `as_of`, keyed by
    the proxy's accession, with `source = proxy`; the 8-K rows get `source = 8k`.
    """
    # a row already labelled (a re-cleaned bundle) keeps its source
    out = None if votes is None else votes.assign(source=votes["source"].fillna(VOTE_SOURCE_8K) if "source" in votes.columns else VOTE_SOURCE_8K)
    if not {"ticker", "as_of", "accession_number", _SOP_SUPPORT}.issubset(def14a_raw.columns):
        return out
    support = cast(pd.Series, pd.to_numeric(def14a_raw[_SOP_SUPPORT], errors="coerce"))
    proxies = pd.DataFrame(
        {
            "ticker": def14a_raw["ticker"].astype(str),
            "accession_number": def14a_raw["accession_number"],
            "_as_of": pd.to_datetime(def14a_raw["as_of"], errors="coerce").astype("datetime64[ns]"),
            "_support": support,
        }
    ).dropna(subset=["_as_of", "_support"])
    in_domain = proxies["_support"].between(0.0, 1.0)
    if int((~in_domain).sum()):
        tally["say_on_pay: proxy value REFUSED (outside [0, 1])"] = int((~in_domain).sum())
    proxies = proxies.loc[in_domain]
    if proxies.empty:
        return out

    covered = _covered_by_8k(proxies, out)
    if int(covered.sum()):
        tally[f"say_on_pay: proxy value NOT used (8-K tally within {_SOP_WINDOW_DAYS} d)"] = int(covered.sum())
    use = proxies.loc[~covered]
    if use.empty:
        return out
    rows = pd.DataFrame(
        {
            "ticker": use["ticker"].to_numpy(),
            "accession_number": use["accession_number"].to_numpy(),
            "proposal_seq": np.nan,
            "filing_date": use["_as_of"].to_numpy(),
            "proposal_type": _SAY_ON_PAY,
            "votes_for": use["_support"].to_numpy(),
            "votes_against": 1.0 - use["_support"].to_numpy(),
            "votes_abstain": 0.0,
            "source": VOTE_SOURCE_PROXY,
        }
    )
    tally[f"say_on_pay: fallback vote rows ADDED from the proxy (no 8-K tally within {_SOP_WINDOW_DAYS} d)"] = len(rows)
    return rows if out is None else pd.concat([out, rows], ignore_index=True)


def _covered_by_8k(proxies: pd.DataFrame, votes: pd.DataFrame | None) -> pd.Series:
    """True where the ticker has a usable 8-K say-on-pay tally filed in the window before the proxy."""
    none = pd.Series(False, index=proxies.index)
    if votes is None or not {"ticker", "filing_date", "proposal_type", "votes_for"}.issubset(votes.columns):
        return none
    sop = votes.loc[(votes["proposal_type"].astype("object") == _SAY_ON_PAY) & votes["votes_for"].notna()]
    right = pd.DataFrame(
        {"ticker": sop["ticker"].astype(str), "_vote_date": pd.to_datetime(sop["filing_date"], errors="coerce").astype("datetime64[ns]")}
    ).dropna(subset=["_vote_date"])
    if right.empty:
        return none
    left = proxies[["ticker", "_as_of"]].assign(_key=range(len(proxies))).sort_values("_as_of")
    merged = pd.merge_asof(
        left,
        right.sort_values("_vote_date"),
        left_on="_as_of",
        right_on="_vote_date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=_SOP_WINDOW_DAYS),
    )
    hit = pd.Series(merged["_vote_date"].notna().to_numpy(), index=merged["_key"].to_numpy()).sort_index()
    return pd.Series(hit.to_numpy(), index=proxies.index)
