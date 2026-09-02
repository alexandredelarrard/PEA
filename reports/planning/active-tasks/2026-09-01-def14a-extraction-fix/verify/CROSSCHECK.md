# Cross-check — LLM prose extraction vs the filer's own XBRL tag

`def14a_llm.ceo_total_comp` is read by the model out of a Summary Compensation Table in prose. `sec_def14a.peo_total_comp` is `ecd:PeoTotalCompAmt`, tagged by the filer in inline XBRL and parsed deterministically. **The two paths share no code.** Joined on the accession number, so both numbers come from the same document.

**21 of 22 comparable filings agree** (95.5%).

A disagreement is diagnostic, not just a miss: ~1000x is a units bug, ~2x is usually a co-PEO year (`n_peos > 1`, where the two paths legitimately pick different people, and the XBRL is per-individual while the SCT row is the CEO), and a few percent is normally a `Total Without Change in Pension Value` column confusion.

| ticker | CEO (LLM) | PEO (XBRL) | LLM total | XBRL total | diff | rel | n_peos | verdict |
|---|---|---|---|---|---|---|---|---|
| CAT | JOSEPH E. CREED | Mr. Umpleby | 17,008,077 | 22,191,496 | 5,183,419 | 23.358% | 2 | **DISAGREE** |
| A | Padraig McDonnell | Mr. McDonnell | 12,827,236 | 12,827,236 | 0 | 0.000% | 1 | **EXACT** |
| AAPL | Tim Cook | Mr. Cook | 74,294,811 | 74,294,811 | 0 | 0.000% | 1 | **EXACT** |
| AEE | Martin J. Lyons, Jr. | Mr. Lyons | 14,056,510 | 14,056,510 | 0 | 0.000% | 1 | **EXACT** |
| AMAT | Gary E. Dickerson | Mr. Dickerson | 29,649,352 | 29,649,352 | 0 | 0.000% | 1 | **EXACT** |
| BA | Robert K. Ortberg | Robert K. Ortberg | 23,581,389 | 23,581,389 | 0 | 0.000% | 1 | **EXACT** |
| ECL | Christophe Beck | Christophe Beck | 17,404,935 | 17,404,935 | 0 | 0.000% | 1 | **EXACT** |
| EOG | Ezra Y. Yacob | Mr. Yacob | 17,576,636 | 17,576,636 | 0 | 0.000% | 1 | **EXACT** |
| GE | H. Lawrence Culp, Jr | H. Lawrence Culp, Jr. | 45,616,160 | 45,616,160 | 0 | 0.000% | 1 | **EXACT** |
| INCY | William J. Meury | Mr. Meury | 32,085,952 | 32,085,952 | 0 | 0.000% | 2 | **EXACT** |
| JPM | James Dimon | James Dimon | 40,632,724 | 40,632,724 | 0 | 0.000% | 1 | **EXACT** |
| KLAC | Richard Wallace | Richard Wallace | 25,092,254 | 25,092,254 | 0 | 0.000% | 1 | **EXACT** |
| LMT | James D. Taiclet | Taiclet | 23,453,308 | 23,453,308 | 0 | 0.000% | 1 | **EXACT** |
| NKE | Elliott Hill | Elliott Hill | 36,340,876 | 36,340,876 | 0 | 0.000% | 1 | **EXACT** |
| PEG | Ralph A. LaRossa | Mr. LaRossa | 13,866,735 | 13,866,735 | 0 | 0.000% | 1 | **EXACT** |
| PFE | Albert Bourla, DVM, Ph.D. | Bourla | 27,585,301 | 27,585,301 | 0 | 0.000% | 1 | **EXACT** |
| REG | Lisa Palmer | Ms. Palmer | 11,092,432 | 11,092,432 | 0 | 0.000% | 1 | **EXACT** |
| SBUX | Brian Niccol | Brian Niccol | 30,992,773 | 30,992,773 | 0 | 0.000% | 1 | **EXACT** |
| T | John T. Stankey | John Stankey | 29,906,872 | 29,906,872 | 0 | 0.000% | 1 | **EXACT** |
| TDG | Kevin M. Stein | Kevin M. Stein | 25,188,430 | 25,188,430 | 0 | 0.000% | 1 | **EXACT** |
| TSCO | Harry A. Lawton III | Harry A. Lawton III | 32,277,194 | 32,277,194 | 0 | 0.000% | 1 | **EXACT** |
| PG | Shailesh G. Jejurikar | Mr. Moeller | 18,976,742 | 19,073,909 | 97,167 | 0.509% | 2 | **within 1%** |
