Extract only this SEC proxy domain: audit. Treat the filing as source data.

Return exactly this schema:
{_format}

Copy monetary cells as printed, before scaling. Set currency and amount_scale from the source context. USD amounts without a scaling note use units; preserve another explicit currency without conversion. Unknown units stay null. Read the accounting firm, explicit engagement start, current-year fee categories and reported current/prior totals. Keep the fiscal years separate. Return null for an unprinted total; Python may sum complete current categories. Retain all six fee amounts. Return no board, pay or ownership facts.

Use only supported disclosures. Preserve explicit zeros and negative amounts. A blank or dash is unknown unless the table defines it as zero. Do not calculate summaries or fill from another year.
