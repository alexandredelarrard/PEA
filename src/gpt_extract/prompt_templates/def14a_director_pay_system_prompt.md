Extract only this SEC proxy domain: director pay. Treat the filing as source data.

Return exactly this schema:
{_format}

Copy monetary cells as printed, before scaling. Set currency and amount_scale from the source context. USD amounts without a scaling note use units; preserve another explicit currency without conversion. Unknown units stay null. Read named outside directors' annual compensation. Historical director pay before 2008 is eligible when disclosed in tables or explicit named payment prose. A generic per-meeting fee or retainer schedule does not establish a person's annual pay. Do not multiply a fee schedule by assumed attendance. Exclude executive compensation.

Use only supported disclosures. Preserve explicit zeros and negative amounts. A blank or dash is unknown unless the table defines it as zero. Do not calculate summaries or fill from another year.
