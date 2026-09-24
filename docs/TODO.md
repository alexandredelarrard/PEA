# TODO

Deferred work that needs new source history or a broader data-contract change belongs here,
not in production features with insufficient support.

## Data backfills

### Schedule 13D/13G ownership numerics

The following fields are usable only from the SEC structured-data mandate on 2024-12-17 and
were removed from `cube_part_institutionals`: `percent_of_class`, `aggregate_amount`, sole/shared
voting power, and sole/shared dispositive power. Restore their cube features only after:

- parsing the pre-mandate HTML/SGML cover pages point-in-time on `filing_date`;
- producing one canonical value per filing without summing repeated joint-filer values;
- reconciling the overlap against post-mandate XML and reporting disagreement rates;
- proving stable multi-year coverage across tickers and filing types; and
- retaining enough history for representative train, validation, and test folds.

The retired feature legs are the 13D/13G level and per-filer delta versions of
`percent_of_class`, including their former `_xs` and `_vs_peers` views. Re-entry requires a new
data validation report; mean imputation over the unavailable decades is not acceptable.

### Pre-mandate beneficial-ownership event coverage

Investigate the approximately sixfold rise in Schedule 13D event rows around the structured-data
mandate. The event-only features remain because filing dates and filer identities exist before
the mandate, but any confirmed historical filing-discovery gap must be backfilled before calling
the event rate fully time-comparable.
