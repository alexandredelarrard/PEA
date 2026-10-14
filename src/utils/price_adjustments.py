"""Guarded corporate-action views for prices, vendor counts and actual quantities.

The register is configuration; resolution is pure and never changes source observations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT

log = logging.getLogger(__name__)
BUGFIX_FILENAME = "prices/yf_price_bugfix.json"


class SplitReport(Protocol):
    splits_rejected: list[str]
    splits_applied: list[str]


@dataclass(frozen=True)
class ShareBasis:
    """Resolved actual/vendor/price events and evidenced quote units for one source snapshot."""

    actual: pd.DataFrame
    vendor: pd.DataFrame
    price: pd.DataFrame
    policies: dict[str, dict]

    def factor(self, tickers: pd.Series, dates: pd.Series, kind: str) -> pd.Series:
        """Strict future-event product, or ordinary shares per quoted security for units."""
        if kind != "units":
            if kind not in {"actual", "vendor", "price"}:
                raise ValueError(f"Unknown share basis {kind!r}")
            return forward_split_factor(tickers, dates, getattr(self, kind))
        result = pd.Series(1.0, index=dates.index)
        stamps = pd.to_datetime(dates)
        for ticker, policy in self.policies.items():
            units = policy.get("quote_units")
            if units is None:
                continue
            hit = tickers.eq(ticker)
            result.loc[hit] = np.where(stamps.loc[hit] < pd.Timestamp(units["date"]), units["before"], units["after"])
        return result


def _closed(record: object, allowed: set[str], required: set[str], label: str) -> dict:
    if not isinstance(record, dict) or set(record) - allowed or required - set(record):
        raise ValueError(f"Invalid {label}: required={sorted(required)}, allowed={sorted(allowed)}")
    return record


def _positive(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")


def _date(value: object, label: str) -> None:
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _event_leg(value: object, label: str) -> dict:
    leg = _closed(value, {"date", "ratio"}, {"date", "ratio"}, label)
    _date(leg["date"], label)
    _positive(leg["ratio"], label)
    return leg


def _validate_policy(ticker: str, value: object) -> None:
    policy = _closed(value, {"anchors", "events", "quote_units"}, {"anchors", "events"}, f"share policy {ticker}")
    if not isinstance(policy["anchors"], list) or not policy["anchors"] or not isinstance(policy["events"], list) or not policy["events"]:
        raise ValueError(f"{ticker}: nonempty anchors and events are required")
    for anchor in policy["anchors"]:
        _closed(anchor, {"date", "ordinary_shares", "quoted_price", "rtol", "source"}, {"date", "ordinary_shares", "rtol", "source"}, "anchor")
        _date(anchor["date"], "anchor")
        for key in ("ordinary_shares", "rtol", "quoted_price"):
            if key in anchor:
                _positive(anchor[key], f"anchor {key}")
        if not isinstance(anchor["source"], str) or not anchor["source"].strip():
            raise ValueError("Share anchor needs a source")
    seen: set[str] = set()
    for event in policy["events"]:
        _closed(event, {"expect", "actual", "vendor", "price", "evidence"}, {"expect", "actual", "evidence"}, "share event")
        expected = _closed(event["expect"], {"yahoo", "sharadar"}, {"yahoo", "sharadar"}, "source expectation")
        for source, leg in expected.items():
            if leg is not None:
                _event_leg(leg, source)
        for key in ("actual", "vendor", "price"):
            if key in event:
                _event_leg(event[key], key)
        if event["actual"]["date"] in seen:
            raise ValueError(f"{ticker}: duplicate share event")
        seen.add(event["actual"]["date"])
        if not isinstance(event["evidence"], str) or not event["evidence"].strip():
            raise ValueError("Share event needs evidence")
        if "price" in event and not any(_price_anchor_discriminates(a, event) for a in policy["anchors"]):
            raise ValueError("Price override requires a quoted-price anchor in its affected prefix")
    units = policy.get("quote_units")
    if units is not None:
        _closed(units, {"date", "before", "after", "quoted_titles", "ordinary_titles"}, {"date", "before", "after", "quoted_titles"}, "quote units")
        _date(units["date"], "quote units")
        _positive(units["before"], "quote units before")
        _positive(units["after"], "quote units after")
        for key in ("quoted_titles", "ordinary_titles"):
            titles = units.get(key, [])
            if not isinstance(titles, list) or not all(isinstance(t, str) and t.strip() for t in titles):
                raise ValueError(f"Invalid {key}")
        if not units["quoted_titles"]:
            raise ValueError("Nonunitary quote convention requires explicit titles")


def _price_anchor_discriminates(anchor: dict, event: dict) -> bool:
    if "quoted_price" not in anchor:
        return False
    old = event["expect"]["yahoo"]
    new = event["price"]
    old_factor = old["ratio"] if old is not None and anchor["date"] < old["date"] else 1.0
    new_factor = new["ratio"] if anchor["date"] < new["date"] else 1.0
    return old_factor != new_factor


def validate_bugfix(blob: dict) -> None:
    """Validate new metadata while retaining the legacy register's accepted vocabulary."""
    policies = blob.get("share_events", {})
    if not isinstance(policies, dict):
        raise ValueError("share_events must map tickers to policies")
    for ticker, policy in policies.items():
        _validate_policy(ticker, policy)
    for entries in blob.get("return_seams", {}).values():
        for entry in entries:
            if "fields" not in entry:
                continue
            _closed(
                entry,
                {"date", "fields", "expect_step", "target_step", "expect_quote_step", "evidence"},
                {"date", "fields", "expect_step", "target_step", "expect_quote_step", "evidence"},
                "total-return seam",
            )
            _date(entry["date"], "return seam")
            if entry["fields"] != ["close_total"] or not isinstance(entry["evidence"], str) or not entry["evidence"].strip():
                raise ValueError("Total-return seam requires close_total only and evidence")
            for key in ("expect_step", "target_step", "expect_quote_step"):
                _positive(entry[key], key)
            if entry["expect_step"] == entry["target_step"]:
                raise ValueError("Return seam defect and target must differ")


def _near_event(frame: pd.DataFrame, ticker: str, event: dict, leg: dict | None = None) -> pd.Series:
    stamps = [pd.Timestamp(event["actual"]["date"])]
    stamps.extend(pd.Timestamp(leg["date"]) for leg in event["expect"].values() if leg is not None)
    stamps.extend(pd.Timestamp(event[key]["date"]) for key in ("price", "vendor") if key in event)
    if leg is not None:
        stamps.append(pd.Timestamp(leg["date"]))
    dates = pd.to_datetime(frame["date"])
    near = pd.Series(False, index=frame.index)
    for stamp in stamps:
        near |= dates.sub(stamp).abs().le(pd.Timedelta(days=SPLIT_MATCH_DAYS))
    return frame["ticker"].eq(ticker) & near


def _check_source(frame: pd.DataFrame | None, ticker: str, event: dict, source: str) -> None:
    if frame is None:
        raise ValueError(f"{ticker}: {source} evidence unavailable for applicable share policy")
    intended = event.get("price", event["actual"]) if source == "yahoo" else event["actual"]
    rows = frame.loc[_near_event(frame, ticker, event, intended)]
    expected = event["expect"][source]
    if expected is None:
        if rows.empty:
            return
        # A newly corroborating Yahoo row changes the represented price basis.
        if any(event["expect"].values()):
            raise ValueError(f"{ticker}: unexpected {source} event; remeasure the share policy")
        expected = intended
    matches = rows["date"].eq(pd.Timestamp(expected["date"])) & np.isclose(rows["value"], expected["ratio"], rtol=1e-12, atol=0)
    healed = rows["date"].eq(pd.Timestamp(intended["date"])) & np.isclose(rows["value"], intended["ratio"], rtol=1e-12, atol=0)
    if len(rows) != 1 or not (matches | healed).all():
        raise ValueError(f"{ticker}: changed {source} date/ratio; remeasure the share policy")


def _replace_event(frame: pd.DataFrame, ticker: str, event: dict, leg: dict) -> pd.DataFrame:
    retained = frame.loc[~_near_event(frame, ticker, event, leg)]
    added = pd.DataFrame({"ticker": [ticker], "date": pd.to_datetime([leg["date"]]), "value": [float(leg["ratio"])]})
    return added if retained.empty else pd.concat([retained, added], ignore_index=True)


def _check_anchors(basis: ShareBasis, shares: pd.DataFrame | None, quotes: pd.DataFrame | None) -> None:
    for ticker, policy in basis.policies.items():
        for anchor in policy["anchors"]:
            stamp = pd.Timestamp(anchor["date"])
            if shares is None:
                raise ValueError(f"{ticker}: share anchor evidence unavailable")
            rows = shares.loc[shares["ticker"].eq(ticker) & pd.to_datetime(shares["date"]).eq(stamp)]
            factor = basis.factor(pd.Series([ticker]), pd.Series([stamp]), "vendor").iloc[0]
            values = pd.to_numeric(rows["sharesbas"], errors="coerce").dropna()
            if values.empty or not np.isclose(values / factor, anchor["ordinary_shares"], rtol=anchor["rtol"], atol=0).all():
                raise ValueError(f"{ticker}: share anchor {stamp.date()} conflicts with vendor restatement; remeasure")
            if "quoted_price" not in anchor:
                continue
            if quotes is None:
                raise ValueError(f"{ticker}: price anchor evidence unavailable")
            quoted = quotes.loc[quotes["ticker"].eq(ticker) & pd.to_datetime(quotes["date"]).eq(stamp)]
            price_factor = basis.factor(pd.Series([ticker]), pd.Series([stamp]), "price").iloc[0]
            restored = quoted["close_split"] * price_factor * quoted.get("hidden_factor", 1.0)
            if (
                restored.empty
                or not np.isfinite(restored).all()
                or not np.isclose(restored, anchor["quoted_price"], rtol=anchor["rtol"], atol=0).all()
            ):
                raise ValueError(f"{ticker}: price anchor {stamp.date()} conflicts with repaired bars")


def _check_quantity_coverage(original: pd.DataFrame, basis: ShareBasis) -> None:
    """A quantity anchor must distinguish a changed vendor basis from the original view."""
    for ticker, policy in basis.policies.items():
        old = original.loc[original["ticker"].eq(ticker)]
        new = basis.vendor.loc[basis.vendor["ticker"].eq(ticker)]
        boundaries = pd.concat([old["date"], new["date"]]).drop_duplicates()
        probes = pd.to_datetime(boundaries).reset_index(drop=True) - pd.Timedelta(nanoseconds=1)
        names = pd.Series(ticker, index=probes.index)
        changed = forward_split_factor(names, probes, old).ne(forward_split_factor(names, probes, new)).any()
        if not changed:
            continue
        dates = pd.Series(pd.to_datetime([anchor["date"] for anchor in policy["anchors"]]))
        names = pd.Series(ticker, index=dates.index)
        if forward_split_factor(names, dates, old).eq(forward_split_factor(names, dates, new)).all():
            raise ValueError(f"{ticker}: share anchor required in the affected vendor interval")


def resolve_share_events(
    actions: pd.DataFrame | None,
    yf_splits: pd.DataFrame | None,
    bugfix: dict,
    *,
    shares: pd.DataFrame | None = None,
    quotes: pd.DataFrame | None = None,
    tickers: Sequence[str] | None = None,
    base_events: pd.DataFrame | None = None,
) -> ShareBasis:
    """Resolve default events and guarded overrides into separate actual/vendor/price views."""
    validate_bugfix(bugfix)
    actual = split_events(actions, yf_splits) if base_events is None else base_events.copy()
    original = actual.copy()
    vendor = actual.copy()
    price = (
        pd.DataFrame(columns=["ticker", "date", "value"])
        if yf_splits is None
        else yf_splits.rename(columns={"ratio": "value"})[["ticker", "date", "value"]].copy()
    )
    sources = {"yahoo": None if yf_splits is None else price.copy(), "sharadar": None}
    if actions is not None:
        sources["sharadar"] = actions.loc[actions["action"].eq(SHARADAR_ACTION_SPLIT), ["ticker", "date", "value"]].copy()
    for frame in sources.values():
        if frame is not None:
            frame["date"] = pd.to_datetime(frame["date"])
    observed = set(actual["ticker"]) | set(price["ticker"])
    for frame in (shares, quotes):
        if frame is not None:
            observed.update(frame["ticker"])
    names = set(tickers) if tickers is not None else observed
    policies = {ticker: policy for ticker, policy in bugfix.get("share_events", {}).items() if ticker in names}
    for ticker, policy in policies.items():
        for event in policy["events"]:
            for source, frame in sources.items():
                _check_source(frame, ticker, event, source)
            actual = _replace_event(actual, ticker, event, event["actual"])
            vendor = _replace_event(vendor, ticker, event, event.get("vendor", event["actual"]))
            if "price" in event:
                price = _replace_event(price, ticker, event, event["price"])
    basis = ShareBasis(
        actual.sort_values(["ticker", "date"]).reset_index(drop=True),
        vendor.sort_values(["ticker", "date"]).reset_index(drop=True),
        price.sort_values(["ticker", "date"]).reset_index(drop=True),
        policies,
    )
    _check_quantity_coverage(original, basis)
    _check_anchors(basis, shares, quotes)
    return basis


#: Calendar days within which a Sharadar and a yfinance split are the same event.
SPLIT_MATCH_DAYS = 7
#: Max distance from a small-integer fraction to read as a split; admits 5-dp rounding, rejects spinoff factors.
SPLIT_INTEGER_TOL = 1e-4
#: Relative ratio gap above which two vendors on one date describe different events (price vs share factor).
SPLIT_RATIO_CONFLICT_TOL = 0.01
#: Largest denominator a genuine split ratio may have (2:1, 3:2, 1:20, 21/20 stock dividend, ...).
SPLIT_MAX_DENOMINATOR = 20


def _is_split_shaped(ratio: float) -> bool:
    """Whether `ratio` is a fraction of small integers (a split or stock dividend), not a spinoff/merger price factor.

    Applied to both vendors: yfinance's `Stock Splits` also carries spinoff factors.
    """
    if not ratio or ratio <= 0 or not np.isfinite(ratio):
        return False
    frac = Fraction(float(ratio)).limit_denominator(SPLIT_MAX_DENOMINATOR)
    return abs(float(frac) - ratio) < SPLIT_INTEGER_TOL


def _is_simple_split(ratio: float) -> bool:
    """Whether `ratio` is split-shaped AND `n:1` or `1:n`; used only to decide what to flag for review."""
    if not _is_split_shaped(ratio):
        return False
    frac = Fraction(float(ratio)).limit_denominator(SPLIT_MAX_DENOMINATOR)
    return frac.denominator == 1 or frac.numerator == 1


def split_events(actions: pd.DataFrame | None, yf_splits: pd.DataFrame | None = None, *, report: SplitReport | None = None) -> pd.DataFrame:
    """The genuine share splits from both vendors, as `(ticker, date, value)`.

    `sharadar_actions` is incomplete per ticker, so it is cross-validated against yfinance (`prices_splits`) by
    `union_split_sources`. A Sharadar `split` co-dated with a `spinoff` is still a split; the spinoff's price
    factor is separated by the ratio-conflict rule, not by dropping the row. `yf_splits=None` uses Sharadar alone.
    """
    # the empty frame needs a datetime `date`, or the union's date subtraction raises TypeError
    empty = pd.DataFrame(
        {
            "ticker": pd.Series(dtype="object"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
            "label": pd.Series(dtype="object"),
        }
    )
    codated: set[tuple] = set()
    if actions is None or actions.empty:
        sharadar = empty
    else:
        frame = actions.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        candidates = frame[frame["action"] == SHARADAR_ACTION_SPLIT]
        kept = [
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "value": float(row["value"]),
                "label": f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']}",
            }
            for _, row in candidates.iterrows()
        ]
        sharadar = pd.DataFrame(kept, columns=["ticker", "date", "value", "label"]) if kept else empty
        # `spinoff` rows veto nothing; they only name co-dated splits for review
        codated = set(map(tuple, frame.loc[frame["action"] == SHARADAR_ACTION_SPINOFF, ["ticker", "date"]].to_numpy()))

    yf = pd.DataFrame(columns=["ticker", "date", "value"])
    if yf_splits is not None and not yf_splits.empty:
        yf = yf_splits.rename(columns={"ratio": "value"})[["ticker", "date", "value"]].copy()
        yf["date"] = pd.to_datetime(yf["date"], errors="coerce")
        yf = yf.dropna(subset=["date", "value"])
        yf = yf[yf["value"] != 0.0]

    out = union_split_sources(sharadar, yf, report=report)
    _log_codated(out, codated)
    return out


def _log_codated(out: pd.DataFrame, codated: set[tuple]) -> None:
    """Warn on kept splits co-dated with a spinoff, and separately on those that are not `n:1` / `1:n`."""
    if out.empty or not codated:
        return
    hit = [
        f"{r.ticker} {pd.Timestamp(cast(Any, r.date)).date()} x{r.value}"
        for r in out.itertuples()
        if (r.ticker, pd.Timestamp(cast(Any, r.date))) in codated
    ]
    if not hit:
        return
    log.warning(
        "%d split(s) co-dated with a spinoff are now KEPT (the veto that dropped them was wrong -- see `split_events`): %s",
        len(hit),
        ", ".join(sorted(hit)),
    )
    odd = [h for h in hit if not _is_simple_split(float(h.rsplit("x", 1)[1]))]
    if odd:
        log.warning(
            "of those, %d is/are split-shaped but NOT n:1 or 1:n -- the shape test "
            "is the only filter left, so confirm each against the filer's own "
            "disclosure: %s",
            len(odd),
            ", ".join(odd),
        )


def _resolve_ratio_conflict(yf_row: pd.Series, near: pd.DataFrame) -> tuple[float, str | None]:
    """`(ratio to keep, warning line or None)` for one corroborated event.

    yfinance's ratio unless the vendors differ by more than `SPLIT_RATIO_CONFLICT_TOL` and only Sharadar's is
    split-shaped; any material disagreement yields a warning line.
    """
    yf_value = float(yf_row["value"])
    label = f"{yf_row['ticker']} {pd.Timestamp(yf_row['date']).date()}"
    for _, other in near.iterrows():
        sh_value = float(other["value"])
        if not np.isfinite(sh_value) or sh_value <= 0:
            continue
        if abs(yf_value / sh_value - 1.0) <= SPLIT_RATIO_CONFLICT_TOL:
            continue
        yf_ok, sh_ok = _is_split_shaped(yf_value), _is_split_shaped(sh_value)
        if sh_ok and not yf_ok:
            return sh_value, (
                f"{label}: yfinance x{yf_value} vs sharadar x{sh_value} "
                f"-> KEPT x{sh_value} (only the sharadar ratio is "
                f"split-shaped; the yfinance one is the spinoff's PRICE "
                f"factor)"
            )
        return yf_value, (
            f"{label}: yfinance x{yf_value} vs sharadar x{sh_value} "
            f"-> KEPT x{yf_value} (yfinance, "
            f"{'both' if yf_ok and sh_ok else 'neither'} split-shaped -- "
            f"no evidence to prefer the other)"
        )
    return yf_value, None


def _reject_yfinance_only(row: pd.Series, report: SplitReport | None) -> None:
    """Record an uncorroborated, non-split-shaped yfinance event as rejected on `report`, if any."""
    if report is not None:
        report.splits_rejected.append(f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']} (yfinance-only, not split-shaped)")


def union_split_sources(sharadar: pd.DataFrame, yf: pd.DataFrame, *, report: SplitReport | None = None) -> pd.DataFrame:
    """Merge two cleaned split lists into one, sorted by `(ticker, date)`.

    In both (within `SPLIT_MATCH_DAYS`): kept on the yfinance DATE (so it steps with `close_split`), ratio per
    `_resolve_ratio_conflict`. One vendor only: kept iff split-shaped (Sharadar-only kept with a warning).
    """
    matched_sharadar: set[int] = set()
    events: list[dict] = []
    conflicts: list[str] = []
    kept_both = kept_yf = kept_sharadar = 0

    for _, row in yf.iterrows():
        near = sharadar.index[(sharadar["ticker"] == row["ticker"]) & ((sharadar["date"] - row["date"]).abs() <= pd.Timedelta(days=SPLIT_MATCH_DAYS))]
        matched_sharadar.update(near)
        value = float(row["value"])
        # corroboration overrides shape; uncorroborated, the shape test decides
        if len(near):
            kept_both += 1
            value, note = _resolve_ratio_conflict(row, sharadar.loc[near])
            if note:
                conflicts.append(note)
        elif _is_split_shaped(value):
            kept_yf += 1
        else:
            _reject_yfinance_only(row, report)
            continue
        events.append({"ticker": row["ticker"], "date": row["date"], "value": value})

    uncorroborated = []
    for idx, row in sharadar.iterrows():
        if idx in matched_sharadar:
            continue
        label = row["label"] if "label" in row else (f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']}")
        if _is_split_shaped(float(row["value"])):
            events.append({"ticker": row["ticker"], "date": row["date"], "value": float(row["value"])})
            uncorroborated.append(label)
            kept_sharadar += 1
        elif report is not None:
            report.splits_rejected.append(f"{label} (uncorroborated, not split-shaped)")

    if uncorroborated:
        log.warning(
            "%d split event(s) in sharadar_actions but NOT in yfinance, kept because the ratio is split-shaped -- review: %s",
            len(uncorroborated),
            ", ".join(uncorroborated),
        )
    if conflicts:
        log.warning("%d date(s) where the two vendors report DIFFERENT ratios -- resolved: %s", len(conflicts), "; ".join(conflicts))
        if report is not None:
            report.splits_rejected.extend(conflicts)

    out = (
        pd.DataFrame(events, columns=["ticker", "date", "value"])
        .drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    log.info("split events: %d corroborated, %d yfinance-only, %d sharadar-only -> %d total", kept_both, kept_yf, kept_sharadar, len(out))
    if report is not None:
        report.splits_applied.extend(f"{r.ticker} {pd.Timestamp(cast(Any, r.date)).date()} x{r.value}" for r in out.itertuples())
    return out


def forward_split_factor(tickers: pd.Series, dates: pd.Series, splits: pd.DataFrame) -> pd.Series:
    """Product of every split dated STRICTLY AFTER each row's filing date (1.0 if none).

    This is the factor Sharadar applied retroactively: divide a count (multiply a per-share figure) by it to
    recover the as-filed basis.
    """
    factor = pd.Series(1.0, index=dates.index)
    if splits.empty:
        return factor
    stamps = pd.to_datetime(dates, errors="coerce")
    for _, split in splits.iterrows():
        hit = (tickers == split["ticker"]) & (stamps < split["date"])
        factor.loc[hit] = factor.loc[hit] * float(split["value"])
    return factor


def suffix_factor(events: pd.DataFrame | None, tickers: Sequence[str], stamps: np.ndarray) -> dict[str, np.ndarray]:
    """Per ticker, `PROD(value : event date > d)` evaluated at every `d` in `stamps`.

    A suffix product plus a `searchsorted`, not a loop over events: the products are formed
    RIGHT-TO-LEFT over dates sorted ascending, so two identical event lists produce
    bit-identical floats in `level_factor`'s division and cancel to exactly 1.0. A left-to-
    right accumulation would not guarantee that.
    """
    out: dict[str, np.ndarray] = {}
    if events is None or events.empty:
        return out

    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])
    frame = frame[np.isfinite(frame["value"]) & (frame["value"] > 0)]
    wanted = set(map(str, tickers))

    for ticker, group in frame.groupby(frame["ticker"].astype(str), sort=False):
        if ticker not in wanted:
            continue
        group = group.sort_values("date")
        values = group["value"].to_numpy(dtype="float64")
        # suffix[i] = prod(values[i:]), with suffix[n] = 1.0 for a date after every event.
        suffix = np.ones(values.size + 1, dtype="float64")
        suffix[:-1] = np.cumprod(values[::-1])[::-1]
        # `right`: an event dated exactly d restates the bars BEFORE d and leaves d itself
        # alone, matching `field_map.forward_split_factor`'s strict `<`.
        pos = np.searchsorted(group["date"].to_numpy(dtype="datetime64[ns]"), stamps, side="right")
        out[ticker] = suffix[pos]
    return out


def load_bugfix(config_dir: str | Path) -> dict:
    """Read `configs/prices/yf_price_bugfix.json`. Returns `{}` when it is absent, so a
    checkout without the register builds normally instead of failing.

    ⚠ REFUSED WITHOUT AN `_APPROVED` BLOCK, exactly as the two Sharadar registers are and for
    the same reason: a regenerated proposal is byte-identical to a reviewed decision, so
    without the check "human-approved" is a sentence in a docstring. These entries rewrite
    prices, which is the strongest case in the repo for demanding the block."""
    path = Path(config_dir) / BUGFIX_FILENAME
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    if "_APPROVED" not in blob:
        raise ValueError(
            f"{path} has no `_APPROVED` block -- refusing to apply a price repair nobody signed off. Add one stating who measured it and how."
        )
    validate_bugfix(blob)
    return blob
