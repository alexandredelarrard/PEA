## Guidelines

You are reading Item 5.07 of a Form 8-K: the certified results of a shareholder 
meeting. Extract every matter voted on and its vote counts.

- READ THE NUMBERS OFF THE PAGE. Every value you return must appear in the text. Do 
not compute, sum, average or infer any vote count. If a number is not printed, 
return null for it.

- The numbers are SHARE COUNTS, not percentages. Many filings also print a '% For' / 
'% Against' column, or stack a percentage row inside the table -- ignore every 
percentage. Some filings TRANSPOSE a non-director proposal into label/value lines 
('Votes Cast For: | 3,495,486,371 | 96.8 %'); read the share count, drop the 
percentage. Round fractional votes to whole shares.

- 'WITHHELD' (or 'Withhold') is the column some filers use INSTEAD OF 'Against' in a 
director election. Put its count in `votes_against` and set `vote_standard` to 
'withheld'. It is NEVER the broker-non-vote column. Broker non-votes are printed as 
'Broker Non-Votes' or 'Non-Votes'; when the filing omits that column or prints 
'N/A', return null -- return 0 only when it prints a zero.

- A director election is ONE proposal with one entry per nominee in `nominees`. Leave 
the proposal-level vote fields null for it; do not add the nominees up.

- Some Item 5.07 filings report no tallies at all -- they only state the board's 
response to an earlier say-on-pay frequency vote. Return an empty `proposals` list 
for those. Never invent a table, a nominee or a number that is not in the text.

## Input 

Input to read is {query}