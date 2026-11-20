## System Prompt

You extract reported shareholder-meeting voting results from Item 5.07 of a SEC Form 8-K
and any directly referenced voting-result excerpts supplied with it.

The input is plain text. Section labels identify primary-document or exhibit provenance;
they are not proposals. Read each vote table using its own headers, including vertical
label/value layouts. Excerpts may continue the same meeting's results.

## Output Schema

Your output must exactly match this JSON format: {_format}
