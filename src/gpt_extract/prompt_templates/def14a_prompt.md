## Guidelines

- SUMMARY COMPENSATION TABLE: return EVERY (executive x fiscal year) row the table shows — 
it normally carries three years per executive. Take `fiscal_year` from the Year column. A 
'-', '—' or blank cell is 0. There are SEVEN dollar components: salary, bonus, stock awards, 
option awards, non-equity incentive, CHANGE IN PENSION VALUE, all other compensation. Do NOT 
confuse this table with the Pay-versus-Performance table (its column says 'compensation 
actually paid') or with the DIRECTOR compensation table.
- CEO pay: the CEO's SCT row for the MOST RECENT fiscal year.
- DIRECTOR COMPENSATION TABLE: one row per director. `fees_earned_usd` is the cash-retainer 
column whatever it is labelled ('Fees Earned or Paid in Cash', 'Cash Fees', 'Retainer'); 
`stock_awards_usd` covers 'Stock Awards', 'Restricted Stock Units' or 'Share Awards'.
- Board composition: read the governance/board 'highlights' summary for board_size, 
n_independent_directors and n_women_directors (e.g. '7 of our 8 directors are independent').
- Directors: resolve `gender` from the proxy's own statement, else the HONORIFIC used for 
that director, else the PRONOUNS in their bio, else the first name — and set `gender_basis` 
to whichever of 'stated'/'honorific'/'pronoun'/'name' you used. Never leave `gender_basis` 
null when `gender` is set.
- Provisions: classified_board and dual_class_shares are STRUCTURALLY always disclosed, so 
return FALSE when the proxy does not indicate them. For poison_pill and 
majority_voting_for_directors return TRUE or FALSE only when the proxy STATES the 
provision's status, and null when the proxy is SILENT — do not infer FALSE from silence.
- Ownership: insider_ownership_pct = the 'all directors and executive officers AS A GROUP' 
percent; ceo_ownership_pct = the CEO's own row; both as decimals (a '*' or '<1%' -> null). 
Both percents must come from the PERCENT OF THE CLASS of shares outstanding (economic 
ownership). Dual-class issuers print a '% of total voting power' / 'combined voting power' 
column beside it — NEVER take that one. n_five_percent_holders = count of owners 
holding >=5%.
- ownership_holders: one entry per HOLDER row across both ownership blocks. Exclude subtotal 
and 'as a group' rows, and exclude a row whose name is only a street address. 
`percent_of_class` is null for '*' / '<1%'.
- Auditor: `auditor_name` is the accounting FIRM NAME only, never a sentence. Report the four 
fee categories for the CURRENT year plus the prior-year TOTAL. Every fee must be WHOLE USD — 
apply any '(in thousands)' or '($ in millions)' note in the table header or the sentence 
before it (a table reading 57.6 under '($ in millions)' is 57,600,000).
- say_on_pay_support_pct as a decimal (92% -> 0.92); ceo_pay_ratio as a number (533:1 -> 533).
Only use values stated in the text; use null when genuinely absent (except 
classified_board / dual_class_shares above).

## Input 

Input to read is {query}