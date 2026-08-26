"""
reason_codes.py  (src/data_extract/utils/fundamentals/reason_codes.py)
--------------------------------------------------------------------------------------------
The ONE place the `dc_code` vocabulary is written down.

Before this module the set existed in no single place: `periods.py` declared three codes,
`xbrl_linkbase.py` three more, `fetch_fundamentals_sec.py` emitted `"not_disclosed"` as a
bare literal, and the two rebuild plans listed DIFFERENT supersets of all of it. A code that
only one producer knows about cannot be validated, and `fundamentals_reason_codes` exists
precisely so that "why is this value absent or qualified?" has a single answer.

So the existing codes are IMPORTED here rather than restated -- there is still exactly one
definition of each, next to the mechanism that emits it -- and the codes the history layer
introduces are declared here, next to the table they are written to.

Two kinds of code live in the same vocabulary, deliberately:

  * an ABSENCE code says why a cell is NULL (`insufficient_quarters`, `not_disclosed`, ...);
  * a QUALIFIER says a cell HAS a value that is not on the field's nominal basis
    (`basis_ex_iprd`, `period_intersection_partial`, `zero_only_retained`, `regime_break`).

Splitting them into two tables was rejected: a consumer asking "can I trust this number?"
must not have to know in advance which of two tables the answer is in. `IS_QUALIFIER` names
the second set, so a null-gate can still ask the narrower question.

One reconciliation, recorded rather than silently resolved: the plan lists a code
`unresolved` and notes that v1's `no_usable_period` "folds in here". They are one slot,
and the name already in the tree -- and already measured, at 1 row in 144,131 -- is
`no_usable_period`, so that is the name kept. `unresolved` is a `resolution_method`
VALUE, and a reason code spelled the same as a method value would carry no information
the row does not already have: every unresolved resolution reaches
`fundamentals_facts` with a more specific code than "no route" already attached.
"""
from __future__ import annotations

from src.data_extract.utils.fundamentals.periods import (
    AMBIGUOUS_DURATION, DERIVED_BASIS_MISMATCH, DERIVED_SIGN_IMPLAUSIBLE,
    INSUFFICIENT_QUARTERS, SPLIT_BASIS_MISMATCH)
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    INCOMPLETE_ROLL_UP, NO_USABLE_PERIOD, PARTIAL_LEAF_SUM, SEGMENT_ONLY_CONCEPT)

# --------------------------------------------------------------- absence codes ---

#: The filer tags nothing for this field anywhere in the filing. Already emitted as a bare
#: literal by `_compose` and by `resolve_field`'s final fallback; named here so the two
#: producers and the validator share one spelling.
NOT_DISCLOSED = "not_disclosed"

#: `fundamentals_exceptions.json` declares the field `expected_absent` for this filing's
#: regime, i.e. the statement template the filer is REQUIRED to use has no such caption
#: (a bank has no `AccountsPayableCurrent`). Applied by the history build, not by the facts
#: layer: the facts layer stays regime-agnostic about absence so that the register can be
#: re-measured against it rather than being assumed by it.
NOT_APPLICABLE_FOR_REGIME = "not_applicable_for_regime"

#: Structural, and NOT driven by the regime register: the catalogue's own `dc_code` /
#: `dc_code_when_absent` keys (bank `capex`, insurer `capex` when absent) and the fields
#: that inherit from them -- bank `freeCashflow` is null BY DESIGN because its subtrahend is.
NOT_APPLICABLE = "not_applicable"

#: The filer folds this caption into another one, so the number is not missing, it is inside
#: a sibling field. `combined_into` on the row carries the destination field.
#:
#: The MECHANISM is wired (an exceptions-register `combined_into` key on a
#: `by_regime` / `by_ticker` cell) but NO cell declares one yet, so it fires zero times
#: today. That is deliberate: §B.6.4 writes the register cells the validator's findings
#: demand, and inventing them here would be the speculative sweep that section refuses.
COMBINED_INTO = "combined_into"

#: A definitional discontinuity in the field itself, not a data defect: the catalogue's own
#: `regime_break` block (ASU 2016-18 for `cash`, ASC 842 for the lease liabilities, LDTI for
#: insurers). The value is real on BOTH sides of the date and comparable on neither.
REGIME_BREAK = "regime_break"

# ------------------------------------------------------------- qualifier codes ---

#: Route 3b's strict period intersection dropped this period: the filer declares the leaves
#: and reported only SOME of them for this window, so the sum would be short by a whole leg.
#:
#: §B.6.6, measured: **128 rows** across 5 (ticker, field) pairs -- EQIX capex 40, EQIX
#: depAmort 40, SCHW cash 34, NEE ppeNet 8, VRT depAmort 6. Until Phase 5 those periods were
#: dropped SILENTLY. For a duration field that surfaces as a null a gate can see; for an
#: INSTANT field (SCHW `cash`) it surfaces as a stale forward-filled value, which no null
#: gate can ever catch -- which is why the code attaches to the (ticker, as_of, field) where
#: the refusal happened rather than to the null it may or may not produce.
PERIOD_INTERSECTION_PARTIAL = "period_intersection_partial"

#: The value survives only because a concept the filer reports as ZERO in every period was
#: put back in play -- the zero is the filer's whole answer (VRT's pre-merger shell), not a
#: tagging artefact. Rides `Resolution.zero_only_retained`.
ZERO_ONLY_RETAINED = "zero_only_retained"

#: `researchAndDevelopment` resolved on the `...ExcludingAcquiredInProcessCost` element
#: rather than the aggregate. Measured on 21 both-tagged pairs: 0.0% agree within 1%, mean
#: aggregate/ex-IPR&D ratio 1.675 -- so the two are a SUPERSET and a SUBSET, and a
#: cross-sectional z-score that mixes them is comparing different measures.
BASIS_EX_IPRD = "basis_ex_iprd"

#: The cell was not resolved from a tagged fact at all: it is an ACCOUNTING IDENTITY over two
#: cells that were. Today's only user is `totalLiabilities` -- see
#: `build_history._total_liabilities_identity` for the measurement that put it here rather
#: than in the facts layer, and for why `Liabilities` is absent often enough to need it.
#:
#: A qualifier and not an absence: the number is right, it is simply not evidence. Phase 5b's
#: `cross_identity` check must treat a row carrying this code as an INPUT and never as
#: independent corroboration of the identity it was computed from.
DERIVED_IDENTITY = "derived_identity"

#: The same identity, PLUS one further inference: the filer's equity resolved on the EX-NCI
#: element, `minorityInterest` did not resolve, and the filer has never tagged a valued NCI in
#: anything visible at that `as_of` -- so a NULL there means zero rather than unknown, and
#: ex-NCI equity IS total equity.
#:
#: Its own code rather than plain `derived_identity` because it rests on TWO inferences and a
#: consumer must be able to drop it separately. Measured on the 54-ticker roster: 121 rows
#: across MCD / NVDA / GOOGL / EOG, which tag no NCI at all; LLY (134 valued NCI facts), TMO
#: (38), NEE (97), ETN (120) and CVS (100) stay refused, and a blanket zero would have been
#: wrong for exactly those. The test is per-FILER and point-in-time, never per-regime -- an
#: asserted rule here is what nearly claimed UNH earns no premiums.
DERIVED_IDENTITY_NCI_ZERO = "derived_identity_nci_assumed_zero"

#: The newest trailing-twelve window this field could assemble ends in a DIFFERENT fiscal
#: quarter from the row's own `fiscal_end`, so carrying it would date another period's number
#: to this one. `build_history._latest` had no freshness bound at all until this code existed:
#: it returned the newest TTM row that had EVER been computed, which is a forward-fill with no
#: cap and no trace.
#:
#: Measured on the 54-ticker roster before the cap: **27 (ticker, field) pairs frozen for 5+
#: years** and 49 for 2+. ORCL `grossProfit` sat at 24,238,000,000 from 2018-11-30 to
#: 2026-05-31 -- 32 consecutive rows -- while `grossMargins` divided it by a growing revenue
#: and manufactured a margin collapse from 0.609 to 0.360 that never happened. BRK-B
#: `operatingIncome` was frozen for 54 of its 57 rows, XOM `dilutedShares` for all 51.
#:
#: The module comment claimed duration fields were "NOT forward-filled". That was true only
#: of a REFUSED TTM, which stays null with its own code; a TTM that simply stops being
#: computable because its input quarters dried up was carried silently, which is the same
#: staircase wearing the other guard's uniform.
STALE_TTM = "stale_ttm"

#: The cell is the catalogue's own `derived_fallback` arithmetic over cells that WERE
#: resolved, applied because no filer tag gave the field. A qualifier, not an absence: the
#: number is right, it is simply not independent evidence -- the same contract as
#: `derived_identity`, and Phase 5b's `cross_identity` must treat a row carrying it as an
#: INPUT, never as corroboration of the identity it came from.
#:
#: Measured on the as-filed facts, where no forward-fill can confound it: of 13 tickers that
#: tag `grossProfit`, `totalRevenue` and `costOfRevenue` in the same filing, **11 satisfy
#: `revenue - cost = grossProfit` EXACTLY in 100% of rows** (AAPL 195, NVDA 267, ADM 211,
#: MSFT 184, SMCI 192, BA/CSCO 195, CVS 95, SWKS 95, JNJ 101, EQIX 2). CAT (24 rows, +22.5%)
#: and COST (6 rows, +20.3%) contradict it, which is why the derivation is gated on the
#: filer's OWN arithmetic, point-in-time, rather than applied as a blanket rule.
#:
#: Only `grossProfit` uses it. `operatingIncome` declares a `derived_fallback` too and it is
#: NOT an identity -- measured on the same substrate, `revenue - cost - SG&A - R&D - D&A`
#: lands within 1% of the filed figure in **0.5% of 550 rows**, mean absolute error 29.3%,
#: mean signed bias -18.1%, because it silently omits restructuring, impairment,
#: acquisition-related and intangible-amortisation lines. Enabling it would inject exactly
#: the class of plausible-but-wrong number this pipeline keeps having to remove.
DERIVED_FALLBACK = "derived_fallback"

#: Reserved for Phase 5b Layer A: a physically-impossible value the validator nulled.
#: Declared now so the validator adds no vocabulary of its own.
FAILED_HARD_GUARD = "failed_hard_guard"

# ------------------------------------------------------------------ the closed set ---

#: Codes that describe a value that IS present. Everything else means the cell is NULL.
IS_QUALIFIER: frozenset[str] = frozenset({
    PERIOD_INTERSECTION_PARTIAL, ZERO_ONLY_RETAINED, BASIS_EX_IPRD, REGIME_BREAK,
    DERIVED_IDENTITY, DERIVED_IDENTITY_NCI_ZERO, DERIVED_FALLBACK})

#: Every legal `dc_code`. A code outside this set is a typo, and a typo in a reason code is
#: worse than no code at all: the null-gate's `LEFT JOIN` still finds a row, so the cell
#: reads as explained while nothing can interpret the explanation. `build_history` asserts
#: membership on every row it writes.
ALL_CODES: frozenset[str] = frozenset({
    # periods.py
    INSUFFICIENT_QUARTERS, SPLIT_BASIS_MISMATCH, AMBIGUOUS_DURATION,
    DERIVED_BASIS_MISMATCH, DERIVED_SIGN_IMPLAUSIBLE,
    # xbrl_linkbase.py / the facts layer
    INCOMPLETE_ROLL_UP, PARTIAL_LEAF_SUM, NO_USABLE_PERIOD, SEGMENT_ONLY_CONCEPT,
    NOT_DISCLOSED, PERIOD_INTERSECTION_PARTIAL, ZERO_ONLY_RETAINED, BASIS_EX_IPRD,
    # the history build
    NOT_APPLICABLE_FOR_REGIME, NOT_APPLICABLE, COMBINED_INTO, REGIME_BREAK,
    DERIVED_IDENTITY, DERIVED_IDENTITY_NCI_ZERO, STALE_TTM, DERIVED_FALLBACK,
    # Phase 5b
    FAILED_HARD_GUARD,
})
