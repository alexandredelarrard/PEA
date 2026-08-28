# Phase 2 — Efficiency A: delete the redundant work ✅

**Goal**: remove work that is provably independent of the loop variable it sits inside. No
algorithm changes, no semantic changes. Every item is "this is computed N times and the answer
never varies".

**Gate**: the Phase 0 harness must report **0 differing cells** on the 8-ticker sample (tier A per item, tier B at the end) after this
phase. Not "within tolerance" — zero.

**Outcome**: all 18 items settled — 14 implemented as written, 3 implemented differently (7, 8,
18) and 1 found to need no change (10). **0 differing cells on both tiers, in every comparison
run.** Wall clock **1.19x**, not the 2–3x the profile shape suggested; the honest number and
what it means are in [Verification](#verification).

## 2.1 The period engine — redundant coercion and re-sorting

Every one of these is already done upstream. `_normalise_facts` (`build_history.py:914`) sorts by
`filing_date` at `:930` and coerces the date columns at `:920-921`.

| # | Site | Redundancy | Fix | Done |
|---|---|---|---|---|
| 1 | `periods.py:569-571` | `pd.to_datetime` on `period_start`, `period_end`, `filing_date`, **E·K times** | Coerce once in `_normalise_facts`; assert the dtype at the `build_periods` entry instead of re-coercing. | ✅ *(guard, not assert — see below)* |
| 2 | `build_history.py:599` | `visible.sort_values("filing_date")` per event, on an already-sorted frame | Drop. Take `.iloc[-1]` on the prefix. | ✅ |
| 3 | `build_history.py:552` | `pd.to_datetime(period_of_report)` over the whole prefix, per event | Precompute the column once per ticker. | ✅ *(dtype guard; the vectorisation is Phase 3's)* |
| 4 | `build_history.py:604` + `periods.py:967` | `fiscal_year_ends` **twice per event** (2·E) | Compute once per event in `build_ticker`/`_snapshot` and pass into both. Signature already accepts it (`quarterize(..., year_ends=...)`). | ✅ |
| 5 | `periods.py:219` | `_latest_per_window` re-sorts the group, **5x per (event, field)** | `quarterize:569-571` already sorted. Sort once per field per event; document the required order in `_latest_per_window`'s contract instead of re-establishing it. | ✅ |
| 6 | `periods.py:719` | `_fiscal_bounds` re-`sorted()`s `year_ends`, E·K + E times | `fiscal_year_ends:685` already returns them sorted. Cache `_fiscal_bounds` on the `year_ends` tuple. | ✅ |
| 7 | `periods.py:860` | `_annual_by_end` re-derives the annual shape, E·K times | `quarterize:587` already computed `annual = _shape(frame, ANNUAL)`. Pass it in. | ⚠ **done differently** |
| 8 | `periods.py:564, :816` | `load_guards()` called E·K times | Already masked by `@lru_cache(4)`, but the call is noise: guards are a parameter on both functions. Pass them. | ⚠ **already free** |

**Item 1, as built.** `quarterize` is called directly by 15 tests with synthetic frames, so a
hard assert would convert a working fixture into a failure. It coerces only where the dtype is
not already `datetime64`, which costs an `is_datetime64_any_dtype` call in place of a full
`to_datetime` and still refuses to sort a string column lexicographically. The plan's stated
purpose — never re-convert on the production path, never silently mis-sort — is met; the
failure mode it asked for is not, and does not need to be.

**Item 7 could not be done as written, and the reason is a real trap.** `quarterize`'s `annual`
frame is NOT the same object `_annual_by_end` needs. It is taken after the `notna` filter and
after `_drop_annual_masquerading_as_quarter`, and — decisively — **for a non-additive field its
`value` column is in SHARE-DAYS**, because the share-day transform runs before the `_shape`
calls. `trailing_twelve` reads `reported_annual[end]["value"]` as the filer's twelve-month
weighted *average*. Passing quarterize's frame in would have multiplied every as-reported annual
share count by ~366. What was done instead removes strictly more work with no semantic surface
at all: `reported_annual` is consulted **only** on the `not spec.is_additive` branch, so it is
now built only there. 45 of the 48 fields stop paying for a `_latest_per_window` whose result
was never read.

**Item 8 was already free on the production path.** `guards or load_guards()` short-circuits,
and `build_ticker:801` resolves the guards before the event loop, so `load_guards()` was never
reached per (event, field). The gap was a caller that passes `None` into `build_periods`;
that is now resolved once at `build_periods`' own entry.

**Item 5's safety is provable, not just harness-checked.** pandas' multi-key `sort_values` is a
`np.lexsort`, which is stable, and a boolean filter preserves relative order — so
*sort-then-filter* and *filter-then-sort* produce the identical sequence, ties included. That is
what makes reusing one sort across the four `_shape` reads a no-op rather than a gamble. The
required order is now a named constant (`_WINDOW_ORDER`) and `_latest_per_window`'s docstring
states it as a contract, with `presorted` as the opt-in.

## 2.2 `build_history` — repeated scans of `visible`

| # | Site | Redundancy | Fix | Done |
|---|---|---|---|---|
| 9 | `build_history.py:304, :355, :360` | `_facts_code`, `_has_valued_fact`, `_qualifiers` each re-filter `visible["field"] == field` — **E x 52** filters | One `visible.groupby("field")` per event, handed to all three. Or a `dict[str, DataFrame]` built once per event. | ✅ |
| 10 | `build_history.py:413` | `_contradicts_gross_profit` builds a full `pivot_table` per event | Build it once per event (it already is) but **hoist it out of the field loop** if it is being called per field; confirm the call count in the profile first. | ⚠ **no change needed** |
| 11 | `build_history.py:619` | `catalogue.history_fields` re-evaluated once per publication event per ticker | Fixed by 2.4. | ✅ |

**Item 9, as built.** `_split_by_field` cuts `visible` once per event with
`groupby("field", sort=False)` — which keeps each group in the parent's filing-date order, the
order the `.iloc[-1]` reads in `_facts_code` and `_total_liabilities_identity` depend on.
`_facts_code`, `_has_valued_fact`, `_qualifiers`, `_deduced_nci` and
`_total_liabilities_identity` all now take the split instead of the frame.
`test_build_history.py`'s NCI-bridge test was updated in the same edit, per decision 10.

**Item 10: the call count was checked first, as the plan asked, and it does not fire per field.**
`_contradicts_gross_profit` has exactly one caller, `_gross_profit_identity` (`:449`), which is
itself called once per event and only when `row["grossProfit"] is None`. So it is already
≤ 1 `pivot_table` per event and there is nothing to hoist. Recorded rather than "fixed".

## 2.3 `fetch_fundamentals_sec` / `xbrl_linkbase` — per-filing waste

| # | Site | Redundancy | Fix | Done |
|---|---|---|---|---|
| 12 | `fetch_fundamentals_sec.py:697` + `xbrl_linkbase.py:502` (via `statement_arcs`) | `calculation_linkbase()` called **twice per filing**; `edgar/xbrl/xbrl.py:304` has no cache decorator. Honest size: 0.003–0.006 s x 23k filings ~= **70–140 s** total | Call once, pass the object down. Cleanliness with a small real win. | ✅ |
| 13 | `fetch_fundamentals_sec.py:625` | `getattr(filing, "period_of_report")` **per row** (hundreds per filing) — a plain `@property` calling `self.sgml()` | Read once per filing into a local. | ✅ |
| 14 | `fetch_fundamentals_sec.py:618, :623` | `pd.Timestamp(filing.filing_date)` twice per row | Once per filing. | ✅ |
| 15 | `fetch_fundamentals_sec.py:713` | `catalogue.extracted_fields` — a `sorted()` over 53 fields, per filing; answer never varies (48) | Fixed by 2.4. | ✅ |
| 16 | `xbrl_linkbase.py:1065, 1161, 1452` | `_candidates` rebuilt up to **7x** per (filing, field) | Build once per (filing, field), pass down. | ✅ |
| 17 | `xbrl_linkbase.py:815, 1165, 1451` | `spec.never_use(regime)` builds a fresh merged dict up to **10x** per (filing, field) | Fixed by 2.4 (memoise on `(field, regime)`). | ✅ |
| 18 | `xbrl_linkbase.py:1198-1200` | `_leaf_sum` prologue + `catalogue.filer_leaves()` runs for **all 48 fields**; only **3** (`capex`, `costOfRevenue`, `depAmort`) declare the `roll_up.any_of` that makes route 3b applicable | Gate the prologue on `spec.roll_up_any_of` being non-empty. Also correct `:1196-1197`'s claim that it "is free when it does not apply" — it is not. | ⚠ **gate widened** |

Items 13 and 14 land as one `_FilingStamp`, built once in `rows_from_xbrl` and threaded into
`_row` and `_period_end`: accession, form, filing date, period of report and the `/A` test.

**Item 18's gate is `any_of` OR a filer register entry, not `any_of` alone.** Route 3b's
`groups` is the union of the catalogue's `roll_up.any_of` and the `by_ticker` leaf register, so
gating on the declaration alone would silently disable the route for a filer whose leaves are
declared per-ticker against a field with no catalogue `any_of`. The gate now returns on
`_leaf_sum`'s first statement when **both** are empty. `test_per_filing_reuse.py` asserts the
entering population is exactly `{capex, costOfRevenue, depAmort}` with no regime, and that no
regime drops any of the three — the plan's own risk-row mitigation, as a test.

The `:1196-1197` comment claiming the route "is free when it does not apply" is corrected; it
now says what the gate actually does.

## 2.4 `kpi_catalogue` — the uncached derived views

`Catalogue.field(name)` is O(1) (`:332`), but **every derived view is an uncached linear scan +
sort**: `history_fields` (`:295`), `side_table_fields` (`:283`), `history_columns` (`:319`),
`scored_fields`, `input_fields`, `extracted_fields`, `unverified_fields`, `by_tier`, `never_use`
(fresh dict per call), `filer_leaves` (fresh frozenset per call). The docstring at `:258` claims
"with lookups precomputed"; only `fields` is.

- [x] Converted to `functools.cached_property`: `all_column_names`, `side_table_fields`,
      `history_fields`, `history_derived_columns`, `history_columns`, `regime_names`,
      `scored_fields`, `input_fields`, `extracted_fields`, `unverified_fields`.
      `all_column_names` returns a `frozenset` now that the object is shared.
- [x] `by_tier(tier)` reads a `cached_property` map built in ONE pass over `fields`, instead of
      a scan and a sort per call.
- [x] `never_use(regime)` and `filer_leaves(ticker, field)` memoise in per-instance dicts, not
      `lru_cache`: `Catalogue` and `FieldSpec` are frozen dataclasses carrying `dict` fields, so
      the generated `__hash__` raises and `self` cannot be a cache key. Both hand back
      immutable objects — `MappingProxyType` and tuples — so a caller cannot mutate a shared
      answer.
- [x] `:258` docstring rewritten to say what is actually precomputed and why it must be.
- [x] **Hashability / mutation**: nothing writes to a `Catalogue` attribute or to its nested
      registers anywhere in `src/` or `tests/` (grepped), and the dataclass is frozen, so the
      caches cannot go stale.

**One regression this caused, found by the suite and fixed.** `cached_property` stores into the
instance `__dict__`, so `FieldSpec(**real.__dict__)` — the clone idiom in
`test_periods_q4.py:40` — starts failing with `unexpected keyword argument
'_never_use_by_regime'` as soon as anything has asked for `never_use`. The test now uses
`dataclasses.replace`, which passes only declared fields; the failure was order-dependent and
would otherwise have surfaced at a random later date.

## 2.5 The double catalogue parse

`functools.cache` keys on the **argument**, so `load_catalogue()` and `load_catalogue("./configs")`
are distinct cache entries. Both conventions exist: no-arg at `fetch_fundamentals_sec.py:872`,
`build_history.py:796, :1050`; explicit at `field_map.py:326`, `validator.py:244`. One
`StepExtractAllData.run()` therefore parses the 169 KB catalogue (3 JSON files) **twice** and
runs all six validation passes each time.

- [x] `resolve_config_dir` in `kpi_catalogue.py` is the single normalisation: `None`, a relative
      and an absolute path resolve to one absolute key. Each loader is now a thin wrapper over a
      cached `_*_at(config_dir: str)`.
- [x] Same treatment for `load_cutovers` (`cik_cutover.py`) and `load_guards` (`periods.py`).
- [x] `load_field_map` (`fundamentals_sharadar/field_map.py`) had **no cache at all** while
      calling the cached `load_catalogue` inside itself. It is cached on the normalised key.
- [x] Test: `tests/data_extract/fundamentals/test_config_dir_cache.py` — 5 tests. Each loader is
      asked for all three spellings; asserts `cache_info().misses == 1` and that the three
      results are the **same object**, and prints the miss count. Measured: **1 miss each** for
      all four loaders.

Phase 5 removes the ambiguity at source by threading one `config_dir` from `Context`, but the
key normalisation is worth having regardless — it is what makes a mistake cheap instead of
doubling a parse.

## 2.6 Not in this phase

- `edgartools`' own noise (301 cosmetic `SGML header declares ...` warnings, 50 `SGML fetch
  failed`). These are library-side (`edgar/sgml/sgml_common.py:238`, `edgar/_filings.py:2013`)
  and cost up to **7 extra GETs** each on the fallback path. Suppressing the cosmetic one is a
  one-line `logging.getLogger("edgar.sgml.sgml_common").setLevel(WARNING+1)` in
  `configs/logging.yml`, but the fallback GETs are a network-reliability question (no retry,
  backoff, timeout or `EDGAR_*` env var is set anywhere in the repo) and belong in their own
  task, not in a refactor. **Recorded, not fixed.**
- The 49 GB / 94,468-file `~/.edgar/_tcache/` with no TTL, size cap or eviction. Same reasoning:
  a real decision, not a refactor.

## Verification

- [x] **Phase 0 harness, tier A frozen mode: 0 differing cells, 0 dtype changes, 0 code
      deltas** — in every one of four independent comparisons:
      | comparison | scope | result |
      |---|---|---|
      | gate 1: 2.4 + items 1, 6, 7, 8 | 8 tickers x 16 filings | **0** cells, 0 codes |
      | gate 2: + items 2, 3, 4, 5, 9 | same | **0** cells, 0 codes |
      | final tree vs the pre-phase snapshot | same | **0** cells, 0 codes |
      | **HEAD `e8740ad` vs the final tree**, both replayed this session | same | **0** cells, 0 codes |
      The last one is the strongest available: it compares the phase's output against the
      committed code's own output on the same machine, so it does not rest on a snapshot taken
      earlier under different conditions.
- [x] Phase 1's changes were confirmed not to move the replay at all (Phase 0's pre-Phase-1
      tier A snapshot vs this session's post-Phase-1 one: **0** cells). That is what makes
      Phase 0's tier B snapshot a valid baseline for this phase.
- [x] Phase 0 harness, **tier B** (8 tickers, full history) once at the end of the phase:
      **0 cells, 0 codes, all 8 tickers** (APA/BA/BAC/BRK-B 69 rows, KR 70, MCD 69, ORCL 68,
      VRT 34), against Phase 0's own tier-B snapshot. 21m 51s.
- [x] Phase 0 harness, **db mode** (read-only) once, on the tier-B freeze. Moving-target
      guard `moved=[]`, so the freeze still matches live `fundamentals_facts`. Drift against
      the stored table: KR 0, VRT 0, MCD 38, ORCL 49, BA 46, BAC 46, BRK-B 32, APA 31 —
      **every one of them `stored=NaN -> rebuilt=<value>` at that ticker's EARLIEST `as_of`**
      (2011-09-23 to 2011-11-08). This is the pre-existing stored-history staleness
      `baseline.md` §5 already recorded for MCD, now visible on six more tickers because
      Phase 0 only ran db mode on two.

      **It is not attributable to this phase, and that is provable rather than argued**:
      `diff_against_stored(stored, rebuilt)` is a pure function of its two arguments, and
      the tier-B check above shows this phase's `rebuilt` is cell-identical to HEAD's. So the
      drift is a property of the stored table alone. Fixing it means re-running
      `build_fundamentals_history` for those tickers, which is post-run work.
- [x] `"$PY" -m pytest tests/data_extract/fundamentals -q` — **231 passed, 0 failed**
      (15m 49s). The first full run gave 215 passed / 13 failed; both causes were found and
      fixed (the `__dict__` clone above, and this phase's own edit to the NCI-bridge test).
- [x] The consumers outside `data_extract/fundamentals` that read the changed return types
      (`never_use` -> `MappingProxyType`, `filer_leaves` -> tuples, the cached `load_field_map`):
      `tests/validate`, `tests/data_store`, `tests/data_extract/sharadar` — 210 passed,
      11 skipped, **5 failed**. All 5 are in `test_sharadar_field_map.py` /
      `test_sharadar_merge.py` and **fail identically at HEAD `e8740ad`** in the worktree, so
      they are pre-existing and outside this phase. `tests/data_store/test_store_boundary.py`
      passes.
- [x] Re-measured tier A wall clock. **The measured gain is 1.19x, not the 2–3x this plan
      pre-registered**, and per the plan's own instruction that is stated plainly rather than
      hunted:

      | build | tier A runs (s) | mean |
      |---|---|---|
      | HEAD `e8740ad` (worktree) | 262, 246, 238 | **248.7 s** |
      | Phase 2 (working tree) | 214, 208, 207 | **209.7 s** |

      Runs alternated head/phase2 on an otherwise idle machine, three pairs. Both triples are
      tight (σ ≈ 12 s and 4 s), so the ratio is trustworthy even though single readings on this
      machine are not: an early "baseline" of **789 s** and a later post-phase reading of
      **156 s** were both outliers, and quoting either would have produced a fabricated 5x. The
      machine takes 3x excursions from background load, and the only defensible protocol is
      paired, alternating, repeated runs.

      **Why the profile shape over-promised.** `baseline.md` §3 attributed 31.7 % of cumulative
      time to `DataFrame.__getitem__` and 19.0 % to `_getitem_bool_array`, which is what
      suggested 2–3x. But cumulative time counts a caller's whole subtree, and the boolean masks
      this phase deletes are the *cheap* ones — `visible["field"] == field` over a 27-column
      prefix — while the expensive indexing is inside `quarterize`/`_ladder`/`trailing_twelve`,
      per window, and is not redundant. Deleting provably-redundant work was never going to
      reach the irreducible part; that is Phase 3's memoisation and Phase 4's pool.
- [ ] **Not done: one commit per numbered item.** Phases 0 and 1 were left uncommitted in the
      working tree, and their edits sit in the same files and sometimes the same functions as
      this phase's, so an 18-way split would have meant reconstructing someone else's
      uncommitted work by hand. The bisection this was for is moot anyway — every gate passed
      first time. The phase is one reviewable diff over 7 `src/` files plus 2 new test modules.

## Risks

| Risk | Mitigation | Outcome |
|---|---|---|
| A "redundant" sort was actually establishing a different order, changing `drop_duplicates(keep=...)` | Assert the required order on entry rather than assuming it; the harness catches the rest. Item 5 is the one to be most careful with. | Item 5 turned out to be provably order-identical (`np.lexsort` is stable); item 2's single-key sort is not, so it rested on the harness — 0 cells on both tiers. |
| `cached_property` on `Catalogue` changes behaviour if anything mutates it after load | Grep for writes to `Catalogue` attributes outside `load_catalogue` before converting. If any exist, freeze first. | Nothing mutates it; it is already frozen. But `cached_property` DID break a `FieldSpec(**__dict__)` clone in a test — fixed, and worth remembering. |
| Caching `never_use(regime)` returns a shared dict a caller then mutates | Return an immutable mapping (`MappingProxyType`) or a frozen dict from the cached path. | `MappingProxyType`; `filer_leaves` likewise returns tuples. |
| The `_leaf_sum` gate skips a field that needed it | Gate on the declared `roll_up.any_of` only — the same condition route 3b already tests internally. Assert the 3 known fields (`capex`, `costOfRevenue`, `depAmort`) still enter it. | Gate widened to `any_of` OR a filer register entry, because `groups` is their union. The 3-field assertion is `test_per_filing_reuse.py`. |

## What this leaves for Phase 3

The fetch-path items (12–18) are **not covered by the replay harness**, which replays
`build_ticker` off frozen facts and never calls the resolver. Their gate is the test suite plus
the three call-count pins in `test_per_filing_reuse.py`. A fetch-path acceptance belongs with
the post-run wide check.

Item 3's "precompute the column once per ticker" was met only as a dtype guard: the O(E²)
shape of `_latest_period_known` (a full-prefix scan per event) is still there, and removing it
is the same vectorisation Phase 3 does for `carry_latest_known`.
