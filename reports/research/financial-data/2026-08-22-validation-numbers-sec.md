# Research: verification pass on the fundamentals rebuild's pending questions

**Date**: 2026-08-22
**Research Phase**: 1 of 3 (FIC workflow) — this is an audit/verification pass, not a new
implementation area. The user has explicitly asked to skip `/plan` for this request.
**Request**: [specs/2026-8-22/2026-08-21_research_validation_numbers_sec.md](../../../specs/2026-8-22/2026-08-21_research_validation_numbers_sec.md)
**Inputs read in full**: [2026-08-21-fundamentals-extraction.md](2026-08-21-fundamentals-extraction.md)
(Parts 1-6, 1,169 lines), [2026-08-21-fundamentals-rebuild-plan.md](../../planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md)
(2,154 lines, Phases 1-3c), [2026-08-22-phase3b-resolution-audit.md](2026-08-22-phase3b-resolution-audit.md),
`configs/fundamentals/fundamentals_kpis.json` (548 lines), `fundamentals_regimes.json`,
`fundamentals_exceptions.json`, and the current (uncommitted) implementation —
`xbrl_linkbase.py`, `entity_scope.py`, `kpi_catalogue.py`, `fetch_fundamentals_sec.py`, and
their four test files.

---

## Why this document exists, and the one finding that shapes everything else

The spec (written 2026-08-21) asks for research to close two specific open items and to
verify three JSON configs. **Both named items were already closed by work done on
2026-08-22**, one day after the spec was written — a second research pass
([-part2.md](2026-08-21-fundamentals-extraction-part2.md)) and Phase 3c of the rebuild plan.
This document's job therefore shifted from "answer the open questions" to **"verify the
closures are real, not merely claimed"** — check the JSON against its own citations, and
check the JSON's resolution logic against the actual Python that runs it, since a contract
file can assert a fix that the code never implements.

**The verification result: every mechanism the JSON claims is real, backed by code, and
covered by a test with a synthetic known-truth fixture.** This was not assumed — each claim
below was traced to a specific function or test. One genuine gap survives (§4), and it is
narrower than anything the spec asked about.

**A second finding, not asked for but material**: the git status shows only "phase 1 and 2"
committed. `xbrl_linkbase.py`, `entity_scope.py`, `fetch_fundamentals_sec.py`, and all four
of their test files are **untracked**, and `fundamentals_kpis.json` / `kpi_catalogue.py` are
**modified-uncommitted**. The rebuild plan's own checkbox state under Phase 3c.9 (several
items still shown `[ ]`) is **stale relative to the working tree** — the code and tests for
those items already exist and pass. This document reports what is actually in the working
tree today, not what the plan's checkboxes say.

---

## 1. The 17 UNVERIFIED-authority fields — verified closed

**Spec's ask**: "17 of 53 fields carry authority: UNVERIFIED... I will need you to produce a
new research .md answering those questions pending."

**Verified state**: `configs/fundamentals/fundamentals_kpis.json` contains the literal string
`"UNVERIFIED"` **zero times** (`grep -c UNVERIFIED` → 1 match, and that one match is the
schema documentation in `_README.field_keys.authority` describing what the placeholder
*would* mean, not an occurrence of it on a field). All 53 fields carry a populated
`authority` string.

This is not merely a JSON claim — it is machine-enforced in two places:

- `kpi_catalogue.py:342-345` (`_build_field`) raises `ValueError` at load time if any field's
  `authority == "UNVERIFIED"` without a matching `authority_note`. This runs on every process
  start (`load_catalogue` is `functools.cache`d, so once per process), not just in a test.
- `tests/data_extract/test_kpi_catalogue.py:43` declares `EXPECTED_UNVERIFIED: frozenset[str]
  = frozenset()` and asserts (`:126-133`) that `cat.unverified_fields == EXPECTED_UNVERIFIED`.
  The assertion message is explicit about direction: *"Closing one is good news — shrink
  EXPECTED_UNVERIFIED. Adding one needs a matching authority_note, not a silent drop."* So a
  future field that regresses to UNVERIFIED will fail the suite, not merely fail to be noticed.

**What actually closed each of the 17** (`accountsPayable`, `accountsReceivable`,
`accumulatedDepreciation`, `dilutedShares`, `goodwill`, `incomeTaxExpense`,
`intangiblesExGoodwill`, `interestExpense`, `inventory`, `minorityInterest`, `ppeGross`,
`ppeNet`, `pretaxIncome`, `retainedEarnings`, `sellingGeneralAdmin`, `stockBasedComp`,
`stockholdersEquity`) — read directly from the current JSON, not from the plan's summary of
it:

| field | citation now in the JSON | grade |
|---|---|---|
| `stockholdersEquity` | FASB doc labels partition `StockholdersEquity` (parent-only) vs `...IncludingPortionAttributableToNoncontrollingInterest`; Reg S-X 5-02(29)/(30)/(31) has no single incl-NCI caption, confirming the roll-up from BOTH sides | `authority` + `authority_caveat` (ASC 810-10-45-16 prose is secondary) |
| `inventory` | Reg S-X **5-02(6)(c) itself** (LIFO-reserve disclosure), not the guessed ASC 330-10-50 tie the pre-closure note had proposed | full `authority`, no caveat |
| `incomeTaxExpense` | `IncomeTaxExpenseBenefit` → ASC 740-10-50-10/-12; the deprecated `IncomeTaxExpenseBenefitContinuingOperations` candidate is **kept**, with the tag-ledger's measured window (276 facts / 82 tickers, 2011-06 to 2013-12) recorded so it is not deleted as "not existing" | full `authority` + `fallback_concept_notes` |
| `interestExpense` | Reg S-X Rule 9-04 caption chain (6/7/8/9) mapped to `InterestExpenseDeposits`/`...ShortTermBorrowings`/`...LongTermDebt`/`InterestExpenseOperating`; generic `InterestExpense` has **no Reg-S-X tie at all** (verified absent from the reference linkbase) | full `authority`, with a measured correction to the first research's proposed leg names (§below) |
| `pretaxIncome`, `ppeGross`, `accumulatedDepreciation`, `intangiblesExGoodwill`, `accountsReceivable`, `accountsPayable`, `goodwill`, `retainedEarnings`, `minorityInterest`, `dilutedShares`, `stockBasedComp` | each cites FASB doc-label text distinguishing superset/subset pairs, plus a Reg-S-X caption number from `us-gaap-ref-2025.xml` or a direct eCFR fetch | full, several with `authority_caveat` for a secondary-sourced ASC prose string |

**Two corrections the closure made to the *first* research's own guesses**, both material
enough to record so nobody re-introduces them: `inventory`'s ASC 330-10-50 hypothesis was
wrong (the operative cite is the Reg-S-X rule itself); `interestExpense`'s proposed bank legs
(`InterestExpenseDeposits + InterestExpenseBorrowings`) would have covered only 2 of 14 banks
— `InterestExpenseBorrowings` was replaced with the two elements filers actually use,
`InterestExpenseShortTermBorrowings` (10/14) and `InterestExpenseLongTermDebt` (11/14).

**Three citation grades exist, not two**, and this is itself worth understanding before
treating "sourced" as binary: `authority` (a primary quote/cite), `authority_caveat` (the
field is fully verified but one ASC paragraph's *prose*, not its *number*, came from a
login-walled secondary source — 7 fields carry one: `stockholdersEquity`, `dilutedShares`,
`stockBasedComp`, `ppeNet`, `goodwill`, `intangiblesExGoodwill`, `accountsReceivable`), and
`authority_inherits_from` (a tier-0 calculation input has no citation of its own because it
exists only to feed a cited field — `kpi_catalogue.py:369-375` walks the chain at load time
and raises if the named parent field doesn't exist).

**Conclusion on item 1**: closed, verified, and load-bearing (a code path enforces it, not
just a document). No further research action needed here.

---

## 2. The regime-vs-GICS-sector absence register — verified closed, and the exact
   reasoning the spec quoted is now in the shipped config

**Spec's ask** (quoting verbatim from a document the user had read): *"The absence register
is measured per regime, not per GICS sector... Assets present for 441/441 in every regime;
bank/insurer 100% absent for currentAssets/currentLiabilities/inventory/R&D... utility/energy
0% absent for currentAssets... bank.capex is .43 absent, not ~1.0..."*

**Verified**: every one of those figures is in `fundamentals_exceptions.json` today, with one
correction already applied and documented in the file itself — the denominator is **442**,
not 441 (`_README.how_measured.scope`, `:9`): *"An earlier draft of this README said 441,
which was off by one against this file's own by_regime block; the block itself was always
right and sums exactly to 442 (bank 14 + insurer 16 + broker_dealer 6 + real_estate 19 +
utility 28 + energy 18 + hybrid 1 + industrial 340)."*

Why the register has to be per-regime and cannot be translated from a GICS-sector matrix —
this is the mechanism, read from `_README.primary_signal` in `fundamentals_regimes.json` and
confirmed against the `force_regime` block:

- **GICS "Financials" is not one filing behaviour, it is five.** The reconciliation in
  `fundamentals_regimes.json:121` is exact: `bank 16 + insurer 17 + broker_dealer 6 +
  industrial-via-force_regime 36 + hybrid 1 (BRK-B) = 76`, matching a live query of
  `sp500_tickers WHERE sector = 'Financials'`. A sector-level absence rate for "Financials"
  would average a bank's 100%-absent `currentAssets` against Visa/Mastercard's 0%-absent
  `currentAssets` (they are in the `force_regime` industrial bucket) and produce a number that
  describes no filer.
- **The primitive is the filer's own statement role URI, GICS only a tiebreak** — stated
  directly (`fundamentals_regimes.json:4`): *"A filer that presents a deposit-based balance
  sheet has told us it is a bank, in its own filing, for that period. That is evidence; a
  GICS label is a third-party classification applied to the whole company today and
  back-dated over its entire history."* This is also why `bank`/`insurer` (unclassified
  statement roles, FASB roles 108000/108200) are 100% structurally absent for
  `currentAssets`/`currentLiabilities`/`inventory`/`R&D` — `17 CFR 210.1-02(bb)(1)(i)`, quoted
  in both files — while `utility`/`energy` file **classified** balance sheets under ordinary
  Article 5 and are measured at **0.00** absence for the same fields
  (`fundamentals_exceptions.json:109-113,126-128`). Grouping them with banks would have been
  wrong precisely because "Financials" and "unclassified balance sheet" are not the same set.

**bank.capex's 0.43 vs the plan's earlier ~1.0 "not reliably reconstructible" language**:
verified in `fundamentals_exceptions.json:44-46`. The register does not merely lower the
number — it replaces a single intermittency claim with a measured three-way partition of the
8 banks that ever tag `PaymentsToAcquirePropertyPlantAndEquipment`: **always** (AXP, C, COF,
FITB, HBAN — capex is fully usable for these 5), **sporadic** (CFG 1/13 years, KEY 3/16, TFC
6/16 — a TTM here really would mix tagged and untagged quarters), **never** (BAC, JPM, MTB,
PNC, RF, SYF — the 6 that make the 0.43 rate). `expected_absent: true` is kept with a written
`override_reason` because the *sporadic* group's risk (a silently-wrong mixed-basis TTM) is
worse than a clean null — matching the user's own stated reasoning in the spec almost word
for word. `kpi_catalogue.py` does not special-case this cell: `expected_absent()` and
`measured_absent_rate()` (`:307-319`) read `override_reason` and `measured_absent_rate` as
plain data, so the reasoning lives in the config, not in a code branch that could drift from
it.

**Conclusion on item 2**: closed, and the register's own README documents its own prior
error (441→442) rather than silently correcting it — worth preserving as a pattern.

---

## 3. Verification of the three JSON configs against their own citations and against
   the code that consumes them

The spec asked to "verify first information given there is correct." Beyond the two items
above, three checks were run:

### 3.1 Internal consistency (schema-level)

`kpi_catalogue.py`'s loader (`load_catalogue`, `:353-407`) performs four checks at import
time, all currently passing (`test_kpi_catalogue.py`, 8 tests):
- every field has `tier/kind/sign/unit/definition/authority` (`:338-341`);
- every `authority_inherits_from` names a field that exists in the same file (`:371-375`);
- every field named in `fundamentals_exceptions.json`'s `by_regime` blocks exists in the KPI
  catalogue (`:393-396`) — catches a typo that would otherwise silently excuse nothing;
- every regime-keyed override inside a KPI entry (`fundamentals_kpis.json`'s `"regimes":
  {...}` blocks) names a regime that actually exists in `fundamentals_regimes.json`
  (`:399-403`).

Tier census: 11 Tier-1 + 12 Tier-2 + 17 Tier-3 (16 + `researchAndDevelopment`, carried as
`tier: 3` + `regime_gated` rather than inventing a fourth tier value) + 13 Tier-0 inputs = 53,
matching `_README.counts` and the plan's Phase 2 contract exactly.

### 3.2 The resolution mechanism the JSON *describes* is the mechanism the code *runs*

This was the main open question this pass had to close, because a contract file can assert
a fix in prose while the resolver still does the old thing. Each JSON claim was traced to a
specific function:

| JSON claim | code location | test |
|---|---|---|
| `linkbase_root_discovery: true` on `totalRevenue` only | `xbrl_linkbase.py:733` (`resolve_field`, route 2 gated on the flag); `REVENUE_ANCHORS`/`NOT_A_TOP_LINE`/`discover_root` ranking (`:548-623`) | `test_linkbase_resolution.py`, `test_linkbase_history.py::test_revenue_never_resolves_to_something_off_the_income_statement` |
| `shortTermDebt.total_adjustment._only_when_test: "declared_descendant"` vs `ppeNet`'s default `"not_a_declared_sibling"` | `xbrl_linkbase.py:470-537` (`_resolve_subtractions`, `ONLY_WHEN_DESCENDANT`/`ONLY_WHEN_SIBLING`) | `test_linkbase_resolution.py:420-437` (`3c.5 conditional subtraction`) |
| Zero-only guard (`Revenues=0` loses to a real number, survives alone) | `entity_scope.zero_only_concepts` + `resolve_field`'s two-pass retry (`xbrl_linkbase.py:626-663`) | `test_linkbase_history.py:289` (`test_a_zero_revenue_row_means_the_filer_reported_nothing_else`) |
| `totalDebt.roll_up.require_any`, `ppeNet.roll_up.require_all` | `fetch_fundamentals_sec.py:181-207` (`_compose`) | `test_linkbase_resolution.py:471-499` (`test_a_composed_field_refuses_to_stand_in_for_its_missing_legs`) — synthetic fixture proves a lease-only sum returns `{}` + `incomplete_roll_up`, never a plausible wrong number |
| `_linkbase_weights` trusts a leg's sign only when its declared parent is the field's OWN total (the MSFT SG&A fix) | `xbrl_linkbase.py:784-823` | `test_linkbase_resolution.py:440-468` (`test_a_leg_weight_is_only_trusted_against_this_fields_own_total`) — synthetic MSFT fixture asserts `[1.0, 1.0]` when untrusted, `[-1.0, -1.0]` when trusted |
| `menucat` union with a hardened role-URI test | `xbrl_linkbase.py:250-299` (`statement_arcs`), `NON_STATEMENT_ROLE` regex (`:109-111`) | `test_linkbase_history.py::test_the_linkbase_drives_resolution_in_every_year` |
| `employees` excluded from the XBRL walk (`"source": "text:10-K"`) | `kpi_catalogue.py:92-104` (`FieldSpec.is_extracted` returns `False` for a `text`-sourced field) | `tests/data_extract/test_fundamentals_employees.py` |
| Entity scoping — filter on the axis, never the member | `entity_scope.consolidated_facts` (`:76-114`), keeps only undimensioned facts | `test_entity_scope.py` (8 tests) |

**Every mechanism named in the JSON exists in the code and has a dedicated test**, several
with synthetic known-truth fixtures specifically built to reproduce the historical bug
(MSFT's -$34.7bn SG&A, BRK-B's lease-liability-as-total-debt). This is stronger evidence than
the plan document's own checkbox state suggests — see the next section.

### 3.3 One inconsistency found and worth flagging: the plan's checkboxes lag the code

Phase 3c.9 in `2026-08-21-fundamentals-rebuild-plan.md` lists three items as unchecked `[ ]`:
the `linkbase_sum` wrong-parent bug (MSFT), the `_compose` zero-fill bug (`totalDebt`), and
`employees` in the XBRL field list. **All three are fixed and tested in the current working
tree**, per §3.2 above. This is not a defect in the fix — it is the plan document's narrative
not having been updated after the fix landed (consistent with the file being tracked by git
but the fix files being untracked/uncommitted — see the header note). Anyone picking up the
plan document cold would under-estimate how much of Phase 3c.9 is actually done.

---

## 4. What is genuinely still open, with trade-offs (per the spec's "no invention"
   instruction, decisions are flagged, not made)

Everything the spec explicitly asked about is closed (§1, §2). The full read of the JSON,
the code, and Phase 3c.9's own "further observations" surfaced four items that remain open.
None was asked for by name in the spec, but they are the honest answer to "verify... is
correct" — the register is correct as far as it goes, and here is where it does not yet go.

### 4.1 The expected-absence register does not scope the balance-sheet DETAIL fields

**Verified directly against the full text of `fundamentals_exceptions.json`**: the
`by_regime` blocks for `bank`, `insurer`, `broker_dealer`, and `real_estate` cover
`currentAssets`, `currentLiabilities`, `inventory`, `researchAndDevelopment`, `grossProfit`,
`sellingGeneralAdmin`, `costOfRevenue`, `operatingIncome`, and `capex` — but **none of the
four blocks mentions `accountsPayable`, `accountsReceivable`, `ppeGross`,
`accumulatedDepreciation`, `intangiblesExGoodwill`, or `minorityInterest`**. All six are
Tier-3 fields whose absence for an unclassified-balance-sheet filer is structurally identical
to `currentAssets`'s (same Reg S-X 5-02 "when appropriate" mechanism, same
`17 CFR 210.1-02(bb)(1)(i)` exception, which the register already cites for the fields it
does cover). Phase 3c.9's own out-of-sample note flags this directly: *"346 of the in-sample
'holes' are structural. Declare them before Phase 7 builds a gate on them."*

**Trade-off, not a decision made here**: extending the register to these six fields is
config-only work (add cells to `fundamentals_exceptions.json`, following the exact pattern
already used for `currentAssets`) with no code change, and the authority is already
established (same CFR citation, same Reg-S-X mechanism). The only reason to hold off is
sequencing — Phase 7 (the validator) is what actually consumes `expected_absent`, and it has
not been built yet, so there is no functional urgency. The cost of doing it now is small; the
cost of deferring it is that Phase 7 will need to re-derive these six cells from the same
7.8M-fact substrate anyway, which is measurement work already known how to do.

### 4.2 `totalLiabilities`'s derived fallback is specified, not yet implemented

`fundamentals_kpis.json`'s `totalLiabilities` entry declares `"derived_fallback":
"totalAssets - stockholdersEquity"`, and the authority for it (the balance-sheet identity
itself) is sound and already verified (§Phase 3c.6 in the plan: the earlier claim that this
formula "overstates liabilities by the NCI" was checked and **withdrawn** — `stockholdersEquity`
already resolves on the incl-NCI basis first). But this is a *cross-field* derivation, and the
facts layer (`fetch_fundamentals_sec.py`) resolves one field at a time from one filing's own
concepts — it does not currently compute a field from two *other resolved fields'* history.
Verified: `totalLiabilities` has zero coverage for APA, DTE, EOG, ETN, VLO (in-sample) plus
DUK, LLY, MCD, ORCL, TMO, WMT (out-of-sample) — 11 tickers, ~20% of the swept universe,
confirming this is systemic (Reg S-X 5-02 genuinely has no "Total liabilities" caption) and
not roster-specific.

**Trade-off**: the plan already assigns this derivation to Phase 5 (the history layer, which
has cross-field access), and that sequencing is architecturally correct — this is not a
facts-layer bug to rush. Flagging only because the *0% coverage* is currently indistinguishable
from a resolver defect until Phase 5 lands; anyone auditing the facts layer in isolation
before then will see a real gap that is not evidence against the current design.

### 4.3 AXP's bank-basis revenue break is a router question, not a resolver defect

Confirmed in the out-of-sample sweep (3c.9): American Express reports revenue on
`TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses` for 2011-2018 (91 rows), then
switches to the ASC-606 element. AXP is currently routed to the `bank` regime in
`fundamentals_regimes.json` (`"Consumer Finance": ["AXP", "COF", "SYF"]`, `:23-30`), but its
`totalRevenue` resolution goes through the *default* (non-bank) fallback path because the
concept it tags pre-2018 is not in any `never_use` list for the bank regime — it is simply
absent from the industrial fallback list too, so it currently resolves via the generic
priority walk. This is the same *class* of basis break as MTB's (§3c.4, already fixed), not
yet measured with the same rigor for AXP specifically.

**Trade-off, explicitly the user's call per the spec's "I will make the decision" instruction**:
(a) add `TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses` to the bank regime's
`never_use` for `totalRevenue`, forcing AXP onto the two-leg `InterestIncomeExpenseNet +
NoninterestIncome` roll-up like every other bank in the roster (consistent, but AXP's
post-provision figure may already be closer to what card-lender analysts expect); or
(b) leave it as a card-issuer-specific basis, documented as a known break rather than
unified with deposit-taking banks. Both are defensible; this is not resolved by any prior
measurement in the corpus.

### 4.4 Re-registration silently truncates a ticker's history (universe-layer, not
   resolver-layer)

Confirmed live 2026-08-22 (plan §3c.6): `Company(ticker)` resolves through
`company_tickers.json`, which maps a ticker only to its *current* registrant CIK. A
**rename** (CVS Caremark→CVS Health, Facebook→Meta) keeps the CIK and is harmless. A
**re-registration** (new holding company, foreign domestication) does not: APA Corp
(CIK 1841666, since 2021-05) has no visible link to predecessor Apache Corp
(CIK 6769, 2011-02 to 2024-11), and GOOGL (Alphabet, CIK 1652044, since 2015-10) has none to
Google Inc (CIK 1288776, 2011-02 to 2016-02). Critically, **the fix is not a union of the two
CIKs**: Apache Corp kept filing its own 10-K/10-Q through 2024-11-07 (it retains registered
public debt), so 2021-2024 is double-covered by two *different legal entities*, and
concatenating would duplicate filings and mix a subsidiary's statements with the parent's. A
correct repair needs a **dated cutover** at the reorganisation date (2021-03-01 for APA,
2015-10-02 for Alphabet), which is universe-construction logic, not something the per-ticker
resolver can discover on its own.

**Trade-off**: this was explicitly called out in the plan as "out of scope for Phase 3c... a
named gap, not an APA footnote, because GOOGL shows it recurs." No config or code exists yet
to declare a CIK-cutover table. This is a genuine open decision about where such a table
would live (a new `configs/fundamentals/*.json`, or a column on `sp500_tickers`) and is
deferred here rather than invented.

---

## 5. Summary answer to the spec's three numbered asks

1. **"Review the first research result and the plan's first two parts"** — done; see the
   header's list of files read in full, and the header note on stale-checkbox risk.
2. **"Research the 17 UNVERIFIED fields"** — already closed by the 2026-08-22 second-pass
   research; **this pass independently re-verified the closure is real** (zero `UNVERIFIED`
   strings, a load-bearing test, each of the 17 traced to its current citation) — §1.
3. **"Research the regime-vs-GICS reasoning, and verify the three JSONs"** — the quoted
   reasoning is verified present, correct, and consistent with the exceptions register's own
   documented self-correction (441→442) — §2. All three JSONs pass their schema-level
   invariants, and — going beyond a JSON-only check — every resolution mechanism the JSON
   *claims* was traced into the actual Python and found implemented with a synthetic-fixture
   test — §3. Four genuinely open items survive, none of them the two named in the spec,
   each stated with its trade-off rather than decided — §4.

**No implementation, no plan, and no code was written or changed in the course of this
research**, per the spec's constraints. The four open items in §4 are candidates for the
next `/plan` invocation, or for direct config edits if the user wants to close 4.1 (the
lowest-effort, best-evidenced one) without a planning pass.
