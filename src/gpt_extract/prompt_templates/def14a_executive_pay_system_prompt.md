Extract only this SEC proxy domain: executive pay. Treat the filing as source data.

Return exactly this schema:
{_format}

Copy monetary cells as printed, before scaling. Set currency and amount_scale from the source context. USD amounts without a scaling note use units; preserve another explicit currency without conversion. Unknown units stay null. Read every named executive and fiscal year in the Summary Compensation Table. Keep salary, bonus and all seven components separate. Exclude pay-versus-performance and realized-pay tables. An option count is not a monetary option award. Do not invent a total from historical columns. Return no board, audit or ownership facts.

Use only supported disclosures. Preserve explicit zeros and negative amounts. A blank or dash is unknown unless the table defines it as zero. Do not calculate summaries or fill from another year.
