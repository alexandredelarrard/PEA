# Research: how data providers (SEC, XBRL US, Compustat, Bloomberg) actually validate fundamentals

**Date**: 2026-08-22
**Research Phase**: 1 of 3 (FIC workflow) — feeds the eventual Phase 7 (validator) plan.
**Request**: follow-up to [2026-08-22-validation-numbers-sec.md](2026-08-22-validation-numbers-sec.md),
in conversation, not a new spec file. User asked: how do S&P/Bloomberg/etc. run checks, and
what does that imply for validating a large volume of fundamentals in this repo.
**Method**: WebSearch + WebFetch against primary/authoritative sources (SEC.gov, XBRL US,
peer-reviewed citations already in the corpus). Sources listed at the end; two fetches
403'd (a GitHub docs subdirectory, one blog) and are noted as unreached rather than silently
dropped.

---

## Summary — the one pattern every source shares

**Nobody validates by re-deriving the value and checking it matches itself, and nobody
claims 100%.** Every entity examined — the SEC's own EDGAR acceptance pipeline, the XBRL US
Data Quality Committee, and Compustat — runs a **cheap, automated, deterministic layer
first, on data independent of how the number was derived**, and reserves expensive
review (human analyst, external cross-check) for whatever that layer flags. None of them
block or silently correct on a statistical flag; they escalate it. This directly confirms
the two-layer design already decided (flag-only + hard-impossible-null) and gives it three
independent precedents rather than one internal argument.

---

## 1. The SEC's own posture: calculation inconsistencies are *warnings*, not errors

**Verified from the EDGAR Filer Manual and SEC.gov structured-data guidance.** An EDGAR
submission with XBRL attachments passes through several validation levels — file-format
syntax, EDGAR-specific syntax restrictions, and semantic consistency checks — but a
calculation-linkbase inconsistency (the filer's own footing math not reconciling) **produces
a warning, not a rejection**. The filing is accepted; the warning stays visible to every
downstream data consumer, and the SEC's Office of Structured Disclosure (and the Division of
Corporation Finance, via comment letters) treats it as a *quality signal* to follow up on,
not a blocking error.

**Rule 6.6.30**, verbatim as returned by search of the Filer Manual: *"Invert the sign of a
numeric fact whose element has an `xbrli:balance` value that is inconsistent with the
reporting concept being reported."* This is the SEC's own formal statement of the sign
problem — the `balance` attribute (`debit`/`credit`) on an element is supposed to determine
sign, and getting that inversion wrong is common enough to have its own numbered rule.

**Why this matters for us**: the SEC — the entity that actually receives and could reject
these filings — chose warn-and-flag over reject-and-block for exactly the identity checks we
discussed as "meaningful but not certain" (footing, sign). That is direct precedent for the
flag-only layer, from the strictest possible party in this chain.

## 2. The SEC's own named error taxonomy includes exactly the check you'd expect to be
   invented, not borrowed: "inconsistent element selection across periods"

A practitioner summary of the categories SEC review actually watches for names five buckets;
the one worth quoting because it is **the same failure mode this repo calls
`tag_switch_break`**, independently named by the regulator's own review process: *"You have
used different XBRL elements to tag the same reported line item on the income statement from
period to period"* — detected, per that source, by comparing the element used across
consecutive filings, not by any property of the value itself. The other four buckets: missing
required tags, custom-extension-over-standard-element substitution (flagged when a filer's
custom-tag rate is unusually high), sign/calculation errors (Rule 6.6.30 sign, and a paired
calculation-linkbase consistency rule), and context/unit/period mismatches (point-in-time vs.
duration, wrong units, misaligned dates) — the last caught automatically at submission time,
before the filing is even accepted.

**Academic corroboration, already partly in this repo's corpus**: Debreceny, Farewell,
Piechocki, Felden & Gräning, *"Does it add up? Early evidence on the data quality of XBRL
filings to the SEC,"* *Journal of Accounting and Public Policy* 29(3), 296-306 (2010) —
confirmed as a real, findable citation (not confabulated) via ScienceDirect/ResearchGate
listings, and it is the source the first research report already cited for "sign convention
is the dominant cause of XBRL arithmetic failure." **One number could not be independently
re-verified this pass**: a secondary source (blocked on fetch, 403) attributed a
"more than 50% negative-where-positive-expected, more than 10% positive-where-negative-expected"
split to XBRL calculation errors generally. Treat that specific split as **unverified** —
the *direction* of the finding (sign errors dominate) is corroborated by Debreceny et al. and
by Rule 6.6.30 existing at all; the exact percentages are not confirmed against a primary
source in this pass.

**Further corroboration on filer heterogeneity**, from a study in *The Accounting Review*
(found via search, not independently re-verified against the journal directly this pass):
XBRL error rates are **systematically higher for smaller filers, first-year adopters, and
filers using in-house tagging rather than a specialist filing agent**; filers with high
custom-tag rates show lower comparability and draw more XBRL-related comment letters. This
is a genuinely new, actionable signal not in the prior research: **filer characteristics are
themselves a prior on how much to trust a given filing's raw tags**, independent of any
check run on the values.

## 3. XBRL US's Data Quality Committee (DQC): the free, executable, independent rule set

**Mechanism, verified from the GitHub repo and the XBRL US rules-guidance page**: DQC rules
are written in a domain-specific assertion language (**Xule**) and executed by a plugin to
**Arelle**, an open-source XBRL processor — i.e. **a completely separate engine from
whatever resolver produced our stored value**, run directly against the filer's own XBRL
instance document. This is the single most actionable finding for the provenance-independence
problem raised earlier in this conversation: running DQC rules is *by construction* not
circular with our own resolution code, because it never touches our code path at all — it
re-parses the filing from scratch with a different, independently-maintained tool.

**Version 30.0.3 (approved June 2026) ships 196 rules**, organized into families (mapped from
the fetched rules-guidance categorization; rule-ID ranges are as returned by that fetch and
should be treated as approximate groupings rather than an exhaustive index, since the full
rule-by-rule catalogue lives in a `/docs` subdirectory this pass could not reach — 403):

| family | rule IDs (approx.) | mechanism |
|---|---|---|
| Calculation & footing | 0004, 0009, 0043-0062, 0084, 0093, 0118, 0126, ... | Assets = Liabilities + Equity and similar statement-level arithmetic; cash-flow reconciliation; hierarchical roll-up checks — run against the filer's own calculation linkbase, exactly the primitive this repo's resolver already reads |
| Negative value / sign | 0013-0015, 0080, 0092, 0147, 0174 | elements declared non-negative must not carry a negative value except under stated conditions; flags inverted elimination accounts |
| Dimension / member | 0001, 0041, 0052-0055, 0079, 0104, 0166 | axes carry only permitted members; custom members must not duplicate a standard taxonomy member; hypercube violations |
| Context / date | 0005, 0006, 0033, 0036 | date contexts align with the stated reporting period; subsequent-event dates don't improperly extend past the fiscal period |
| Presentation / hierarchy | 0018, 0045-0049, 0099, 0105, 0117, 0127 | deprecated elements; elements missing an expected calculation relationship |
| Scale / decimals | 0091, 0095, 0103, 0139, 0157 | percentage values outside a plausible range; scale mismatches between related metrics (the `dcml` idea already used in `num.tsv`) |
| Extension controls | 0079, 0107, 0144 | illegitimate custom extensions; monetary items missing a `balance` attribute |
| Domain-specific | 0067-0078, 0082-0090, 0107-0188 | revenue recognition (ASC 606), lease accounting (ASC 842), tax reconciliation, dividends, acquisitions — i.e. rules that encode exactly the kind of standard-specific knowledge this repo's KPI catalogue already carries in prose |

**DQC_0118** specifically (fetched directly): *"identifies inconsistent calculations in
financial statements covering the Cash Flow Statement, Statement of Financial Position, and
Income Statement by evaluating each line representing an aggregation based on elements
defined in the calculation linkbase."* Its own documentation notes the practical problem this
repo already discovered independently: *"Calculation inconsistencies reported to XBRL filers
can be overwhelming as many don't represent real errors, so validation rules filter out false
inconsistencies to help filers quickly identify valid calculation issues"* — i.e. even the
DQC itself had to learn that a naive footing check over-fires and needed calibration, the
same lesson this repo's Phase 3c drew from its own menucat/role-URI and zero-guard work.

**Output format**: a Xule assertion evaluated against an instance produces a **validation
message** — a rule ID, a severity, the specific fact(s) and concept(s) involved, and
human-readable text. Structurally, this is exactly the `(check_name, severity, evidence,
message)` shape already proposed for `fundamentals_quality` — DQC's output schema is not a
new design, it's confirmation of the one already sketched.

## 4. Compustat: automated volume-check + mandatory human review, never automation alone

**Verified from Compustat's own user-guide language (found via WRDS/Fidelity/user-guide
documents), consistent with the ~2,500-per-company figure this repo's first research already
cited** (the "14,000 checks/company" figure from the prior document could not be
re-confirmed to a specific primary source this pass; the recurring, more consistently sourced
number across multiple hits is **"more than 2,500 validity checks... performed on each
company entered into the Compustat database"** — treat 14,000 as the vendor's own marketing
claim for the *platform*, and 2,500 as the *per-company* figure with more independent
corroboration).

The mechanism, stated plainly in Compustat's own materials: **every report is reviewed by an
analyst** for adherence to presentation format; relevant numbers are *extracted from
footnotes and narrative text* by that analyst, not by pure automation; data may be
*"adjusted or reapportioned"* by the analyst where necessary; and **automated systems perform
internal consistency checks on top of, not instead of, that human step.** Academic
research on Compustat data quality (footnote-item miscoding, e.g. NOL carryforwards
sometimes coded zero/missing when a disclosed value exists) confirms the highest error rates
sit precisely in the **footnote-sourced, judgment-requiring items** — the same tier this
repo's own Tier-3 fields occupy, and the same reason the KPI catalogue treats those fields'
absence as needing a structural register rather than a blanket assumption.

**Why this matters for "100%"**: even the largest, longest-running vendor in this space does
not claim automation alone reaches full correctness — its stated process is
automation-plus-mandatory-analyst-review for every single report, forever. That is a direct,
concrete answer to the user's "100%" framing: **it is not achieved by better automation, it
is achieved by automation narrowing what a human has to look at, applied continuously.** A
validator that tries to replace that human step entirely is solving a different, harder
problem than any existing vendor has solved.

## 5. Bloomberg — governance framing, not a published check catalogue

Bloomberg does not publish a fundamentals-specific validation methodology comparable to
Compustat's or the SEC's (searched directly; found only the *Quality Indices* factor
methodology, which is a portfolio-construction document, not a data-QA one, and a general
"Open Data Principles" data-governance white paper). The governance white paper's only
transferable idea: Bloomberg's data-management group classifies its own quality metrics into
three explicit tiers — **essential, sufficient, best-in-class** — rather than a single pass/
fail bar. That maps onto the severity tiers already proposed for `fundamentals_quality`
(critical/high/medium/low), and is worth noting as independent confirmation that "graded
confidence" rather than "binary pass/fail" is the norm among data vendors generally, not
something specific to XBRL. No further primary detail on Bloomberg's actual fundamentals
check logic was reachable in this pass — flagged as unresearched rather than assumed absent.

---

## 6. What this changes about the validator design for this repo

Nothing here overturns the two-layer decision already made; it sharpens *how* to reach
"validate a large volume" without pretending to certainty nowhere earns it.

**A concrete new layer, not previously proposed**: run the **XBRL US DQC ruleset (via Arelle)
directly against each cached filing**, independent of and prior to our own resolver. This is
the one mechanism in this whole research pass that is *provenance-independent by
construction* — it doesn't touch our Python at all, so a DQC finding is never circular with
whatever our `resolve_field` did. It also catches a category of defect our resolver
architecturally cannot see: an error in the **filer's own linkbase**, upstream of anything we
read from it. Cost/effort trade-off: it is a new dependency (Arelle + the DQC rules package)
and a per-filing run cost on top of `xbrl()`/`calculation_linkbase()`, so it is additive
infrastructure, not a small tweak — a genuine option to weigh, not a decided addition.

**A triage/waterfall, which is the actual answer to "how do you validate at scale"**: every
source examined here — SEC's own acceptance pipeline, DQC, Compustat — puts a **cheap,
fully-automated, provenance-independent pass first across the entire volume**, and reserves
**expensive, slow methods (human review, external vendor cross-check) only for what that
first pass flags**. Concretely, for this repo that reads as four tiers of increasing cost,
each operating only on the prior tier's output:

1. **Per-filing, independent of our resolver** (DQC/Arelle, if adopted) — runs once per
   filing, catches source-XBRL defects.
2. **Per-resolved-value, deterministic** (sign, footing gated on provenance, coverage vs the
   exceptions register) — cheap, vectorizable across the whole `fundamentals_facts` table.
3. **Per-series, statistical** (MAD outlier, mean-shift, peer-relative, frozen-staircase) —
   still cheap at this volume, but probabilistic, so it produces *candidates* not verdicts.
4. **External corroboration** (Tiingo/Yahoo cross-check) — reserved for whatever tier 2-3
   flagged, because it's the slowest and rate-limited, mirroring Compustat's "automate first,
   send the residual to a human/second source" pattern exactly.

**On filer-sophistication as a prior**: the *Accounting Review* finding (smaller filers,
first-year adopters, in-house taggers → higher error rates) suggests severity thresholds in
tier 3 could be *filer-aware* rather than universal — the same absolute deviation is more
suspicious for a large, mature filer with an established tagging history than for a small-cap
first-year filer, mirroring how DQC_0118's own documentation says a naive footing check
"overwhelms" without calibration. This is a refinement worth carrying into the eventual plan,
not a decision made here.

---

## What this pass did not close

- The exact DQC rule-ID-to-family index (the repository's `/docs` subdirectory 403'd via
  WebFetch) — would need `gh` or a direct clone to enumerate all 196 rules precisely rather
  than the approximate groupings above.
- The "50%/10%" sign-error split's primary source (the page carrying it 403'd).
- Compustat's "14,000 checks" figure's origin — likely a different metric (platform-wide vs.
  per-company) than the more consistently sourced "2,500+ per company."
- FactSet's, Worldscope's, and Moody's specific validation *mechanisms* (as opposed to their
  standardization templates, already covered in the first research pass) — this pass found
  no new primary material on their check methodology beyond what was already in the corpus.
- Bloomberg's actual fundamentals-check logic, as distinct from its portfolio-index and
  general data-governance materials.

---

## Sources

- [Approved Validation Rules - XBRL US](https://xbrl.us/home/priorities/data-quality/rules-guidance/)
- [FASB proposes updates to 2026 Data Quality Rules taxonomy | XBRL](https://www.xbrl.org/news/fasb-proposes-updates-to-2026-data-quality-rules-taxonomy/)
- [GitHub - DataQualityCommittee/dqc_us_rules](https://github.com/DataQualityCommittee/dqc_us_rules)
- [Financial Statement Tables Calculation Check of Required Context - DQC_0118](https://xbrl.us/data-rule/dqc_0118/)
- [FS Calculation Check with Non Dimensional Data - DQC_0126](https://xbrl.us/data-rule/dqc_0126/)
- [SEC.gov | Staff Observations From Review of Interactive Data Financial Statements](https://sec.gov/structureddata/osd_staffobs_06-15-11.html) (fetch 403'd; cited via search snippet only)
- [EDGAR XBRL Guide Prepared by SEC Staff, June 2026](https://www.sec.gov/files/edgar/filer-information/specifications/xbrl-guide.pdf)
- [Filer Manual – Volume II EDGAR Filing, March 2026 (v77)](https://www.sec.gov/files/edgar/filermanual/edgarfm-vol2-v77.pdf)
- [Inline XBRL Tagging Errors in 10-K Filings: 2026 Practitioner Guide | Finrep Blog](https://www.finrep.ai/blog/inline-xbrl-tagging-errors-in-10-k-filings-2026-practitioner-guide)
- [XBRL Tagging Errors That Trigger SEC Review | Finrep Blog](https://www.finrep.ai/blog/xbrl-tagging-errors-that-trigger-sec-review)
- [Does it add up? Early evidence on the data quality of XBRL filings to the SEC — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0278425410000219)
- [Standard & Poor's COMPUSTAT User's Guide](https://sites.bu.edu/qm222projectcourse/files/2014/08/compustat_users_guide-2003.pdf)
- [Introduction to Standard & Poor's Compustat - Fidelity](https://www.fidelity.com/learning-center/trading-investing/fundamental-analysis/introduction-to-compustat)
- [Data Quality Problems Troubling Business and Financial Research (footnote miscoding)](https://digitalcommons.wcupa.edu/cgi/viewcontent.cgi?article=1013&context=lib_facpub)
- [Bloomberg Quality Indices Methodology, Oct 2022](https://assets.bbhub.io/professional/sites/10/Bloomberg-Quality-Indices-Methodology.pdf)
- [Applying Open Data Principles to Financial Data Governance - Bloomberg](https://data.bloomberglp.com/promo/sites/12/750171296-FinDataGovernance.pdf)
