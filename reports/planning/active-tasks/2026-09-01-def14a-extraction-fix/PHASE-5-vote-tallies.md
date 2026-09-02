# Phase 5 — Shareholder vote tallies from 8-K Item 5.07 ✅

**Goal**: parse the **6,657 Item 5.07 narratives already stored in `sec_8k.item_text`** into a new
`sec_8k_votes` table — one row per proposal, with director elections collapsed to one row carrying
per-role-category vote columns.

**Depends on**: Phase 3 (the role categorisation joins to `def14a_executive_comp` and
`def14a_director_comp`).

---

## Why this is cheap (measured)

- **We already fetch it.** `fetch_8k_edgar.py:57` tags `_HIGH_SIGNAL_ITEMS["5.07"]`. Live DB:
  **6,657 rows, 405 tickers, 2010-03-01 → 2026-08-19**, and 99.2% of them have `item_text` > 200
  chars. Nothing parses them.
- **No HTML re-fetch is needed.** `item_text` contains zero HTML tags (0 of 5,984 rows checked) but
  is edgartools' *rendered* view and **column alignment survives**: 84.6% carry a `U+2500` rule under
  the header, the rest are whitespace-aligned, one filing row per text line.
- Vote counts are in Form 8-K Item 5.07 **and nowhere else**. Item 5.07 began **March 2010**
  (Rel. 33-9089 moved the disclosure out of 10-Q Part II Item 4). **No vote number is ever in
  XBRL** — Apple's 2025 annual-meeting 8-K carries 21 facts, **100% `dei:` cover-page tags**.
  There is no SEC dataset, no bulk route, and **GitHub has no 8-K Item 5.07 vote parser or dataset
  at all**.
- **LLM-only, per D3.** Head-to-head on 60 filings / 690 hand-read rows:

| | deterministic parser | gpt-4o-mini |
|---|---|---|
| rows fully correct | 82.8% | 82.5% |
| missed | 39 | **1** |
| **filings fully clean** | 40% | **65%** |
| cost | free | **$0.00063/filing → ~$4 for the corpus** |

  The parser's failures are layout-driven and unbounded: 12 distinct header vocabularies for 34
  filings, 41% combined-table layouts, 20.0% using "Withheld" not "Against", a **vertical
  `For: 1,234` layout that it misses 100% of the time**, and a dropped-header case that produces a
  silent **column permutation** on 5 filings.

---

## The two hard rules

Both are non-negotiable — they are the LLM's two measured failure modes.

### Rule 1 — the fabrication guard
The LLM **fabricated an entire table** for the truncated HWM 2019 filing ("John Doe" / "Jane Smith",
250,000,000 votes) and a fake row for BSX.

- [x] **Emit nothing when `item_text` contains no comma-grouped number** (`\d{1,3}(?:,\d{3})+`).
      Vote tallies are **always shares** and always comma-grouped in these filings.
- [x] Also emit nothing when `item_text` is shorter than a floor (~400 chars). HWM 2019 stores 508
      chars where the live filing has 10,922 and four tables, because the filer numbered proposals
      `Item 1.` / `Item 2.` and edgartools' item splitter cut there. **62 rows corpus-wide (1.0%)** —
      accepted as a loss (README), not repaired by re-fetching HTML.
- [x] Additionally reject any extracted row whose name/description does not appear as a substring of
      `item_text`. This is the cheap, general form of the guard and it also catches the BSX case.
      (Precedent: the research verified **99.74% of `directors[]` names appear verbatim in the
      source** — a verbatim check is a valid hallucination test on this data.)
- [x] A legitimately tally-free filing exists: **8.8% have no HTML vote table at all**
      (HUM `0000049071-23-000042` is a real tally-free Item 5.07(d) board-response filing; ES 2010
      and IBKR 2012 narrate in prose). "No rows" must be a normal outcome, logged as such, not an
      error.

### Rule 2 — never "latest wins" on amendments
**190 multi-filing meetings** on `(ticker, period_of_report)`. In **135 (71%) the amendment carries
no vote numbers at all**, so "latest wins" is correct on only **33/190 (17%)**; **"union the group"
is correct on 173/190 (91%)**.

- [x] The PK `(ticker, accession_number, proposal_seq)` stores **every** filing's rows, so the union
      is the table's natural state. Do not dedup across accessions.
- [x] Carry `is_amendment` and `period_of_report` so a reader can union on
      `(ticker, period_of_report)`.
- [x] **224 of 6,657 (3.4%) are 8-K/A and 167 of ~5,400 meetings (3.1%) have >1 filing**, but only 14
      filings say "preliminary" — so it is **not signposted**. Add a `mentions_preliminary` flag
      (regex on `preliminary` / `certified` / `final` in `item_text`) which resolves 8 of the 17
      genuine restatements. The remaining **9 of 190 (4.7%) have no signal beyond the form type** —
      accepted loss.
- [x] The trap worth naming in the docstring: **in contested elections the first 8-K is
      PRELIMINARY.** DIS `0000950157-24-000595` states *"estimated preliminary voting results …
      do not include shares voted on the blue proxy card distributed by Trian"*; the 8-K/A
      `-000623` carries the Inspector of Election's final results **and switches from
      Against to Withhold**.

---

## Changes

### 1. `src/data_extract/utils/structure/vote_schema.py` (new)

```python
class NomineeVote(BaseModel):
    name: str
    votes_for: Optional[float]
    votes_against: Optional[float]        # "Against" OR "Withheld" -- see vote_standard
    votes_abstain: Optional[float]
    votes_broker_non_votes: Optional[float]

class ProposalVote(BaseModel):
    proposal_number: Optional[str]        # as PRINTED ("1", "2a", "Item 3")
    description: str
    proposal_type: str                    # same vocabulary as the retired sec_def14a_votes
    vote_standard: Optional[str]          # 'against' | 'withheld'
    votes_for / votes_against / votes_abstain / votes_broker_non_votes: Optional[float]
    nominees: list[NomineeVote] = []      # populated ONLY for director_election

class Item507Extract(BaseModel):
    meeting_date: Optional[str]
    is_preliminary: Optional[bool]
    proposals: list[ProposalVote] = []
```

- [x] `proposal_type` vocabulary is **reused verbatim** from the retired `sec_def14a_votes`:
      `director_election`, `say_on_pay`, `say_on_pay_frequency`, `auditor_ratification`,
      `equity_plan`, `shareholder_proposal`, `company_proposal`. Nothing new invented.
- [x] `vote_standard` exists because **20.0% of filings use "Withheld" not "Against"** and conflating
      them silently would corrupt every support percentage.
- [x] `votes_broker_non_votes`: **75.2% of filings mention broker non-votes, encoded three ways when
      absent** (column omitted / `N/A` / `0`). The description must say: null when the filing does
      not report them; `0` only when the filing prints a zero.
- [x] `say_on_pay_frequency` carries **4 buckets** (1yr / 2yr / 3yr / abstain) in 25.4% of filings.
      Store the frequency proposal as one row; the bucket detail is not worth a fifth vote column —
      it lands in `nominee_votes_json` if the model returns it.

### 2. `src/data_extract/utils/structure/fetch_8k_votes_llm.py` (new)

- [x] Reads `sec_8k` where `item = '5.07'`, projected to
      `(ticker, cik, accession_number, filing_date, period_of_report, form, is_amendment, item_text)`
      and filtered `where={"item": "5.07"}` — **never** an unprojected read (AGENTS.md).
- [x] Incremental via `existing_filings(context, Tables.sec_8k_votes)` — accession-only dedup, the
      same convention as every other per-filing fetcher, so a re-run costs nothing.
- [x] `LLMExtractor(model=config.data_extract.llm_model)` with a task-tailored `instructions` string
      (cached per `(model, schema)` via the existing `prompt_cache_key`).
- [x] Per-ticker upsert, like `fetch_def14a_llm` — LLM calls are paid for, persist as you go.
- [x] `_strip_nul` before saving (Postgres TEXT rejects `\x00`).
- [x] The prompt must explicitly correct the **one systematic, prompt-fixable LLM error measured**:
      a `broker_non_votes → votes_withheld` mislabel on ~62 rows. State that "Withheld" is a
      substitute for "Against" in director elections and is **never** the broker-non-vote column.
- [x] Also instruct: numbers are **share counts, not percentages**; a `% For` / `% Against` column is
      an extra to ignore (21.1% of filings carry one, and XOM **transposes** non-director proposals
      as `Votes Cast For: | 3,495,486,371 | 96.8 %`).
- [x] Do not attempt fractional-vote precision: the measured LLM error included fractional votes
      truncated 1000× (CVNA). Round to whole shares.

### 3. Director role categorisation (categories VALIDATED -- see README)

- [x] `_role_map(context, ticker, meeting_date) -> dict[str, str]` — build once per (ticker, meeting)
      from the **nearest prior proxy**:
      - `def14a_llm.ceo_name_proxy` → `ceo`
      - `def14a_executive_comp.name` (that ticker, most recent fiscal year ≤ meeting) minus the CEO →
        `exec_officer`, and collect their `title` values into `exec_officer_titles`
      - `def14a_director_comp.name` → `non_employee` (Item 402(k) ⇒ non-employee by construction)
      - anything else → `unmatched`
- [x] Name matching: normalise via the existing `clean_person_name` (casing left alone, footnote
      suffixes stripped, whitespace collapsed), then compare on a
      `last name + first initial` key so `"Timothy D. Cook"` matches `"Tim Cook"` and
      `"Mr. Cook"`. Log the per-filing unmatched count — that is the join's error rate and it must be
      visible, not hidden.
- [x] Sum the four vote fields within each bucket; count nominees per bucket.
- [x] Compute `min_support_pct = min(for / (for + against))` over all nominees, `min_support_name`,
      and `n_nominees_below_70pct`.
- [x] Store the raw per-nominee tallies in `nominee_votes_json` so **recategorisation is free** —
      the same property `def14a_json` gives the proxy path.

### 4. `src/data_store/schema.py` (risk zone)

```python
# Shareholder-meeting vote tallies parsed out of the ALREADY-STORED `sec_8k` Item 5.07
# narratives -- one row per proposal. Item 5.07 is the ONLY source of certified vote counts
# (Rel. 33-9089, from 2010-03); no XBRL tag, no SEC dataset, no vendor publishes a free parse.
# Director elections collapse to ONE row whose per-role-category columns are summed across
# nominees; the raw per-nominee tallies stay in `nominee_votes_json`. Amendments are stored as
# their own rows and UNIONED by the reader -- "latest wins" is correct only 17% of the time.
sec_8k_votes = Table("sec_8k_votes", ("ticker", "accession_number", "proposal_seq"),
                     date_col="filing_date",
                     date_type_cols=("filing_date", "period_of_report"))
```

Columns: keys + `form`, `is_amendment`, `mentions_preliminary`, `proposal_seq`, `proposal_number`,
`proposal_type`, `description`, `vote_standard`, the 4 proposal-level vote columns, the
20 director-category columns (4 categories × for/against/abstain/broker/n), `exec_officer_titles`,
`n_nominees`, `min_support_pct`, `min_support_name`, `n_nominees_below_70pct`, `nominee_votes_json`.

The 20 category columns are NULL on non-director rows (~85% of rows). Accepted: you asked for
columns rather than rows, and a director election is exactly one row per meeting.

### 5. Wiring

- [x] `src/data_extract/transformers/step_extract_structure.py` — call `fetch_8k_votes_llm` **after**
      `fetch_8k_edgar` (it reads that table) and **after** `fetch_def14a_llm` (it reads the role map).
- [x] `src/data_extract/cli.py` — new command `sec-8k-votes`, following the existing `def14a`
      command's shape (config + tickers options, model from `config.data_extract.llm_model`).
- [x] `sql/schema.sql` — splice the block by hand.

### 6. No validation gate exists — say so in the docstring

- [x] A vote table prints **no independent total**, and the dominant error is a column permutation,
      which is **invariant under sums**. Measured gates: total ≤ shares outstanding computable on
      only 9/57 (16%); per-nominee totals computable 96% and hold 91%; meeting-level totals
      computable 100% and hold 81%. **Combined: recall 0.56 / precision 0.56 — 7 of 16 known-bad
      filings pass clean.** Do not build a gate and do not imply one exists.
- [x] What we do instead: the two hard rules above, plus the per-nominee-sum check as a **flag**
      (`nominee_sum_matches`) rather than a filter — it is computable on 96% of filings and holds on
      91%, which makes it a useful monitor and a bad gate.
- [x] Prior-year say-on-pay in the proxy narrative is **not** an independent check either: it is
      lossy (Merck says "approximately 94%" where its 8-K says 93.50%) and the denominator
      convention varies by state of incorporation (XOM states New Jersey excludes abstentions from
      votes cast). Useful as a sanity comparison, never as a gate.

---

## RESULTS — code complete, verified free; the paid probe is OUTSTANDING

New: `vote_schema.py`, `fetch_8k_votes_llm.py`, `tests/.../test_8k_votes.py`,
`fixtures/item507_texts.json`, and `utils/common/frame_sanitize.py`. Touched:
`data_store/schema.py`, `sql/schema.sql`, `docs/data_schema.md`, `cli.py`,
`step_extract_structure.py`, and `fetch_def14a_llm.py` (the `strip_nul` move).

`sec_8k_votes` is **46 columns**, and the DDL was proved against the flatten name-for-name AND
in declaration order — `test_the_ddl_declares_exactly_what_the_flatten_writes` compares the two
in both directions, so a column the code writes but the table lacks (an insert failure after the
tokens are paid) and a column the DDL declares but nothing writes (permanently NULL, which reads
downstream as non-disclosure) both fail here. `sql/schema.sql` diff is **+63 / −0**, purely
additive; indexes 32 → 34, exactly the two new ones. **No live-Postgres type-check this phase**
(no DB credentials in the session), so the DDL is verified against the code, not against the
server; Phase 6 touches the DB anyway.

**Tests: 171 passed, 2 skipped** in `tests/data_extract/structure` (was 141 + 2), 30 of them new
and every one of them free. The two `tests/data_store/test_read_equivalence.py` failures are
**pre-existing and live-DB-dependent** — confirmed by stashing this phase's diff and re-running:
they fail identically without it.

### Fixtures: the plan's filings weren't all reachable, so real equivalents were substituted

The plan named DIS / HUM / XOM / HWM, none of which are in the 23-ticker Phase-0 baseline, and
there is no DB access this session. Every geometry was re-sourced from the 333 real Item 5.07
narratives that ARE in the baseline (19 tickers, 2010-03-01 → 2026-06-11):

| Geometry the plan wanted | What the plan named | Real substitute used |
|---|---|---|
| 4-col per-nominee + a 3-col auditor table | AAPL `...25-005876` | **the same filing** |
| a `%` row stacked inside the table + `N/A` broker | JPM `...25-000485` | **the same filing** |
| `Withheld` instead of `Against` | (unnamed) | **TDG 2020**, `FOR / WITHHELD` |
| the vertical `For: 1,234` layout | (unnamed) | **TDG 2020** proposal 2, one label per line |
| `Non-Votes` vs `Broker Non-Votes`, same filer | GE 2019 / 2024 | **confirmed in both**; one column carries both |
| a truncated `item_text` -> zero rows | HWM 2019 | **AAPL 2020** (0 chars) + a synthetic short stub |
| a genuine tally-free 5.07(d) -> zero rows | HUM `...23-000042` | **NKE 2011** (8-K/A, 553 chars, board response) |
| the say-on-pay frequency, 4 buckets | (unnamed) | **JPM 2017**, `One Year / Two Years / Three Years / Abstain / Broker Non-Votes` |
| an amendment that is not deduped | DIS `...24-000595` + `-000623` | AAPL 2025 + a synthetic 8-K/A accession, **through a real SQLite upsert** |
| the dropped-header column permutation | (unnamed) | **AEE 2026**, whose header row is the single word `Name` |

**XOM's transposed non-director layout has no baseline equivalent.** It is addressed in the
prompt and by nothing else — no test covers it. Recorded rather than glossed.

### Corrections to the plan, found by measuring

1. **The length floor cannot be the truncation defence.** The plan puts the floor at ~400 and
   cites HWM 2019's 508-char stub as what it catches — but 508 > 400, and raising the floor is
   impossible: the **shortest GENUINE tally in the baseline is 554 chars** (TDG's 2014 and 2019
   special meetings, each a complete `FOR / AGAINST / ABSTAIN` block on one stock-option plan).
   A floor high enough to reject a 508-char stub destroys two correct filings. So the floor
   stays at 400 and catches only the emptiest stubs; what actually catches a truncation is the
   **comma-grouped-number rule** plus the per-row grounding check — a stub cut before the tables
   carries no share count to read. `_MIN_ITEM_TEXT_CHARS`'s docstring records this and
   `test_the_shortest_genuine_tally_in_the_corpus_is_not_mistaken_for_a_stub` pins it.
2. **The verbatim guard applies to nominee NAMES and to vote NUMBERS, not to descriptions.** The
   plan says "name/description". A `description` is a model paraphrase of the filing's prose
   ("Election of directors"), so requiring it verbatim would reject nearly every row. Numbers
   are both the stronger and the more meaningful check — a fabricated tally is the actual defect.
3. **The number check is ANY-of-four, not all-of-four.** All-must-match would cost a whole row
   over one cell the model rounded, and the measured error set includes fractional votes
   truncated 1000x. "None of this row's numbers appear in the source" is exactly the shape of
   the BSX invented row, and that is what gets rejected.
4. **A per-label tally is not a per-person one.** Found while building the JPM 2017 fixture:
   `is_election` was initially `proposal_type == "director_election" OR bool(nominees)`, which
   would have built a role map, an `exec_officer_titles` and a `min_support_pct` for a bucket
   called **"One Year"**, and nulled the frequency proposal's real `votes_abstain` /
   `votes_broker_non_votes`. `is_election` is now the `proposal_type` alone; the frequency
   buckets land in `nominee_votes_json`, which is where the plan puts them, and recategorising
   out of the JSON stays free.
5. **`nominee_sum_matches` is null off an election row**, for the same reason: the four frequency
   buckets are four different counts and will never agree, so a `0` there would read as a defect
   rather than a category error.

### Deviations

- **`_strip_nul` was PROMOTED, not duplicated.** It became `strip_nul` in the new
  `utils/common/frame_sanitize.py`; `fetch_def14a_llm` and `test_def14a_nul` now import it from
  there. It is a Postgres correctness invariant (a NUL in a TEXT column takes down the whole
  INSERT), and two copies of that is how one copy later drifts.
- **`_role_source` reads once per TICKER, not per (ticker, meeting).** The plan's signature was
  `_role_map(context, ticker, meeting_date)`; a ticker with 17 annual meetings would have
  re-read the same three tables 17 times. `_role_source` does the three narrow reads and
  `_role_map(source, meeting_date)` derives the map per meeting off those frames.
- **Three columns beyond the plan's list**: `cik` (carried from `sec_8k`, free), `meeting_date`
  (the model's own reading, so a disagreement with `period_of_report` is visible instead of
  assumed away — an 8-K/A's `period_of_report` is not always the meeting date), and
  `is_preliminary_stated` (the model's reading of "are these numbers provisional", next to
  `mentions_preliminary`'s regex; the signal is weak either way, and two independent weak reads
  that disagree are more informative than one).
- **Column order is pinned.** `_director_columns` returns in `_DIRECTOR_COLS` order, matching the
  NULL fill a non-election row uses. Without it a frame's column order depended on whether its
  first proposal happened to be an election — and on a first-run `CREATE TABLE` the frame's
  column order IS the DDL's.
- **`max_chars=40_000`**, not the DEF 14A path's 130k: the input is one already-carved item, and
  the longest Item 5.07 narrative in the baseline is 17,032 chars.

### Carried into Phase 6

- [ ] **The live LLM probe (~20 calls, real spend).** Everything free is done and green; the
      model's *reading* accuracy on `gpt-5-mini` is the one thing that cannot be tested without
      paying, and the research's 82.5% row / 65% clean-filing numbers were measured on
      `gpt-4o-mini`, so they must be **re-measured, not assumed**. The same run produces the
      unmatched-rate-over-20-filings number the role map is judged on (the fetcher already logs
      it per ticker).

## Verification

- [x] **Fixture tests** (`tests/data_extract/structure/test_8k_votes.py`), synthetic-from-real:
      real `item_text` strings captured from the DB, hand-read expected rows. Cover the geometries
      that broke the deterministic parser, because they are the LLM's inputs too:
      - AAPL `0001140361-25-005876` — per-nominee 4-col `For|Against|Abstained|Broker Non-Vote`,
        with the auditor table dropping to 3 columns.
      - JPM `0000019617-25-000485` — a `91.45 | % | 8.14 | %` row stacked *inside* the table, and
        `N/A` for broker non-votes.
      - XOM `0000034088-25-000030` — interleaved `% For`/`% Against` and **transposed** non-director
        proposals.
      - GE — `Non-Votes` (2019) vs `Broker Non-Votes` (2024), the same filer switching vocabulary.
      - A "Withheld" filer → `vote_standard == 'withheld'` and the count in `votes_against`.
      - A say-on-pay-frequency filing (4 buckets).
- [x] **Guard tests**:
      - HWM 2019's truncated 508-char `item_text` → **zero rows**, and specifically no "John Doe".
      - HUM `0000049071-23-000042` (a genuine tally-free 5.07(d)) → zero rows, logged as normal.
      - A row whose extracted nominee name is absent from `item_text` → rejected.
- [x] **Amendment test**: DIS 2024's preliminary `0000950157-24-000595` and its 8-K/A
      `-000623` both produce rows; `mentions_preliminary` is True on the first; nothing is deduped
      away.
- [x] **Role categorisation test** on a real meeting where the CEO is on the ballot: `n_nominees_ceo
      == 1`, `n_nominees_non_employee` ≈ board size − 1, `n_nominees_unmatched == 0`. Print the
      unmatched rate across a 20-filing sample — that number is the join's honest error rate.
- [ ] **Live LLM probe** (~20 calls): 20 filings spread 2010→2026, printing rows produced, guard
      rejections and the `nominee_sum_matches` rate. Compare against the research's 82.5% row
      accuracy and 65% fully-clean-filings on `gpt-4o-mini` — this run is `gpt-5-mini`, so the
      numbers are expected to differ and must be **re-measured, not assumed**.
- [x] `"$PY" -m pytest tests/data_extract/structure/test_8k_votes.py -v -s` — prints the sanity
      conclusion.

## Rollback

Entirely additive: one new table, one new fetcher, one new CLI command, one call site. Drop the table
and revert.

## Notes

- Discovery is free: the SEC submissions JSON `items` array already carries the codes and
  `list_filings` already surfaces them, so no extra downloads. (For the record, EDGAR full-text
  search's `items=` parameter is **silently ignored** by the production API — filtering must be
  client-side on `_source.items` — and it caps at 10,000 results. We do not need it.)
- Do not pursue N-PX or vendor data. Settled empirically: **~51% of N-PX filings contain zero vote
  records**, 13F managers report **say-on-pay only** (Rule 14Ad-1), funds hold ~33% of US equity, and
  **2,684 N-PX filings in 2025 mention Apple's CUSIP** — that is the parsing bill to *partially*
  reconstruct one meeting whose own 8-K gives the complete tally in one document.
- `sec_8k` item 5.07 grew 5,965 → 6,251 → 6,657 rows across the research and planning sessions. The
  fetcher is incremental, so this converges on its own; don't treat a moving count as a bug.
