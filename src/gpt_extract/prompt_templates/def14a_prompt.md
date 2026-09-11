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
percent of TOTAL shares outstanding, ALL CLASSES COMBINED; ceo_ownership_pct = the same basis 
for the CEO's own row; both as decimals (a '*' or '<1%' -> null). Also give insider_shares = 
the group's total SHARE COUNT, summed across classes.
  Dual-class issuers print a '% of total voting power' / 'combined voting power' column. PUT 
  IT IN ITS OWN FIELD: insider_voting_pct / ceo_voting_pct. Never put the same number in both 
  an ownership field and a voting field.
  ⚠ Many dual-class tables print PER-CLASS percent columns ('Class A %', 'Class B %') and NO 
  combined column — e.g. `Class A Shares | Class A % | Class B Shares | Class B % | Total 
  Voting Power %`. In that case set insider_ownership_pct and ceo_ownership_pct to NULL and 
  still fill insider_shares and the voting fields. Do NOT report one class's percentage as the 
  ownership percentage: 46.5% of Class B can be 3% of the company.
  When there is only ONE percent column and it is not labelled as voting power, it is the 
  ownership percentage and the two voting fields are null.
  n_five_percent_holders = count of owners holding >=5%.
- ownership_holders: one entry per HOLDER row across both ownership blocks. Exclude subtotal 
and 'as a group' rows, and exclude a row whose name is only a street address. 
`percent_of_class` is null for '*' / '<1%'; `percent_of_voting_power` carries that row's 
voting-power column when the table prints one, and is null otherwise.
- Auditor: `auditor_name` is the accounting FIRM NAME only, never a sentence. Report the four 
fee categories for the CURRENT year plus the prior-year TOTAL. Every fee must be WHOLE USD — 
apply any '(in thousands)' or '($ in millions)' note in the table header or the sentence 
before it (a table reading 57.6 under '($ in millions)' is 57,600,000).
- say_on_pay_support_pct as a decimal (92% -> 0.92); ceo_pay_ratio as a number (533:1 -> 533).
Only use values stated in the text; use null when genuinely absent (except 
classified_board / dual_class_shares above).

## Input 

Input to read is {query}