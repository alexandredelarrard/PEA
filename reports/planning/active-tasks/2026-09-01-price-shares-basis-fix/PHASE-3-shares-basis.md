# Phase 3 — Shares: vendor basis for features, a separate PIT column ✅

**Parent**: [PLAN.md](PLAN.md) · **Depends on**: P2 · **Blocks**: P4 · **Estimate**: 2h code + fundamentals rebuild

## Goal

`sharesOutstanding` becomes the vendor's split-adjusted `sharesbas`, verbatim. The
point-in-time count moves to its own column, consumed only by ownership ratios.

> ⚠ This **reverses the de-adjustment approved on 2026-08-26**, for the feature columns only.
> The PIT series is not lost — it moves to `sharesOutstandingPit`. Decision D3 in
> [PLAN.md](PLAN.md).

## Why

`deadjust_splits` computes `sharesOutstanding = sharesbas / F(d)`. Three measured problems:

1. It makes market cap depend on an event list with 9 holes, so **AAPL is de-adjusted and GOOGL
   is not** — the same column on different bases across the cross-section. ANET is worse still:
   *partially* de-adjusted (2021 applied, 2024 missing), so the residual factor is neither the
   PIT count nor the vendor count.
2. It gets share-change features **outright wrong**: a 4:1 split reads as +300% issuance in the
   quarter the factor steps. That hits `shares_growth`, `diluted_shares_growth`,
   `buyback_yield`/`shareholder_yield` and one of nine Piotroski legs.
3. On the split-adjusted basis `F(d)` cancels against `close_split`, so none of this is needed
   for any product or ratio involving a price.

10,996 of 51,255 rows (21.5%) across 277 of 489 tickers are currently de-adjusted; 601 upward
(reverse splits — GE x8, C x10, AIG x20, TMUS x2), which produces mcap **overstated** 8-20x.

## Changes

### 1. `configs/sharadar/sharadar_field_map.json`

- [x] Drop `"split_basis"` from all four entries: `sharesOutstanding` (`sharesbas`),
      `basicShares` (`shareswa`), `dilutedShares` (`shareswadil`), `dividendsPerShare` (`dps`).
      These are the only four in the whole config
      ([lines 173-176, 238](configs/sharadar/sharadar_field_map.json#L173-L176)).
- [x] Add `sharesOutstandingPit`: `{"kind": "direct", "from": "sharesbas",
      "split_basis": "count", "pit": true}` — or whatever key the emitter uses to mark "de-adjust
      into a new column rather than in place".
- [x] **The config is hand-formatted.** A `json.dumps` round-trip reformats all 545 lines
      (`fundamentals-config-json-formatting` memory). Use a validated emitter or a text splice,
      and check the diff is confined to the five entries.
- [x] Update the `_SPLIT_ADJUSTMENT` rationale block
      ([lines 73-107](configs/sharadar/sharadar_field_map.json#L73-L107)) to state the new rule:
      *features on the vendor basis because F(d) cancels against `close_split`; PIT only for
      ownership ratios.* Keep the HON spinoff-trap note — it still applies to `sharesOutstandingPit`.

### 2. `src/data_extract/utils/fundamentals_sharadar/field_map.py`

- [x] `deadjust_splits` ([524-582](src/data_extract/utils/fundamentals_sharadar/field_map.py#L524-L582)):
      **keep the machinery unchanged**, but write to a target column instead of overwriting in
      place. It is correct code solving the wrong problem for the feature columns.
- [x] It must consume the P2 unioned event list, not `sharadar_actions` alone.
- [x] Preserve the documented ordering (TTM aggregate first, then de-adjust — the reasoning is at
      [field_map.py:546-557](src/data_extract/utils/fundamentals_sharadar/field_map.py#L546-L557)).

### 3. Schema

- [x] Add `sharesOutstandingPit DOUBLE PRECISION` to `fundamentals_history`. Hand-splice
      `sql/schema.sql`; purely additive.
- [x] `epsDiluted` is derived *after* de-adjustment as `netIncome / dilutedShares`, so it follows
      automatically onto the vendor basis. `optionOverhang` is split-invariant either way. No code
      change for either — but assert both in verification.

### 4. Rebuild

- [x] Full `fundamentals_history` rebuild. Per the `thorough-rebuild-over-incremental` memory, a
      full rebuild is the right call here rather than an in-place migration.

## Verification

- [x] `SELECT count(*) FROM fundamentals_history h JOIN fundamentals_sharadar s USING (...)
      WHERE h."sharesOutstanding" IS DISTINCT FROM s.sharesbas` == **0** on all 51,255 rows
- [x] `sharesOutstandingPit` is non-null wherever `sharesOutstanding` is
- [x] `sharesOutstandingPit` within ±3% of `fundamentals_history_sec.sharesOutstanding` on the
      5,553-row / 96-ticker overlap. **P0 recorded 24 of 96 tickers failing; target is 0-2.**
      The known offenders and their pre-fix ratios: AVGO 10.0, ANET 4.0, CMG 49.7-50,000,
      APH 2.0, CCL 0.0012-1.4032, BKNG 25.0, WTW 2.649, SJM 1.058. AMCR's 0.2 was already
      **correct** (a real 1:5 reverse split) and must stay at 0.2.
- [x] `shares_growth` shows **no** ±300% step in AAPL 2020Q3, AMZN 2022Q2, GOOGL 2022Q3,
      NVDA 2024Q2 — the split quarters. This is the free-fix check.
- [x] `optionOverhang` digest matches P0's `option_overhang_digest` **exactly**. It is
      split-invariant (both legs carry the same factor), so it is the control that proves this
      change touched only what it should. A move here means the rebuild changed something
      unrelated to the basis.
- [x] `epsDiluted` moves onto the vendor basis and reconciles against Sharadar's `epsdil`
      (which is `netinccmn`-basis — expect a level difference, not a split-factor one; see the
      `sharadar-db-measured-2026-08-26` memory)

## Risks

| risk | mitigation |
|---|---|
| The JSON splice reformats 545 lines | Diff-gate the config change; reject anything beyond the five entries |
| A consumer silently keeps reading the PIT semantics under the old name | P4 routes the two real PIT consumers explicitly; P5 invariant 1 catches the rest |
| The rebuild drops rows | Row-count and per-ticker-coverage diff against P0 before accepting |

## Rollback

Restore `split_basis` on the four entries, drop `sharesOutstandingPit`, rebuild. The vendor data
is unchanged throughout, so the rebuild is deterministic in both directions.
